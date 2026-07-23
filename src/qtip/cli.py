from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MODEL = "MossFormer2_SE_48K"
VIDEO_LUFS = -14.0
AUDIO_LUFS = -16.0
TRUE_PEAK = -1.5
LOUDNESS_RANGE = 11.0


class QtipError(Exception):
    pass


def run(
    command: list[str], *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        raise QtipError(detail) from error


def require_ffmpeg() -> None:
    missing = [
        command for command in ("ffmpeg", "ffprobe") if shutil.which(command) is None
    ]
    if missing:
        raise QtipError(f"missing required command: {', '.join(missing)}")


def has_video(path: Path) -> bool:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture=True,
    )
    return bool(result.stdout.strip())


def extract_audio(input_path: Path, output_path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def denoise(input_path: Path, output_path: Path) -> None:
    from clearvoice import ClearVoice

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "qtip"
    cache.mkdir(parents=True, exist_ok=True)

    with working_directory(cache):
        model = ClearVoice(task="speech_enhancement", model_names=[MODEL])
        device = model.models[0].device
        print(f"denoising on {device}...")
        enhanced = model(input_path=str(input_path), online_write=False)
        model.write(enhanced, output_path=str(output_path))


def parse_loudnorm(stderr: str) -> dict[str, str]:
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end < start:
        raise QtipError("ffmpeg did not return loudness measurements")

    try:
        return json.loads(stderr[start : end + 1])
    except json.JSONDecodeError as error:
        raise QtipError("could not parse ffmpeg loudness measurements") from error


def normalize(input_path: Path, output_path: Path, target_lufs: float) -> None:
    target = f"{target_lufs:g}"
    base_filter = f"loudnorm=I={target}:TP={TRUE_PEAK}:LRA={LOUDNESS_RANGE}"
    measurement = run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-i",
            str(input_path),
            "-af",
            f"{base_filter}:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    values = parse_loudnorm(measurement.stderr)
    second_pass = (
        f"{base_filter}"
        f":measured_I={values['input_i']}"
        f":measured_LRA={values['input_lra']}"
        f":measured_TP={values['input_tp']}"
        f":measured_thresh={values['input_thresh']}"
        f":offset={values['target_offset']}"
        ":linear=true:print_format=summary"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-af",
            second_pass,
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )


def write_output(
    input_path: Path, audio_path: Path, output_path: Path, video: bool
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-i",
        str(audio_path),
    ]

    if video:
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map_metadata",
                "0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
            ]
        )
    else:
        command.extend(["-map", "1:a:0", "-map_metadata", "0"])

    command.append(str(output_path))
    run(command)


def process(input_path: Path, output_path: Path, target_lufs: float | None) -> None:
    require_ffmpeg()
    if not input_path.is_file():
        raise QtipError(f"input file not found: {input_path}")
    if input_path == output_path:
        raise QtipError("input and output must be different files")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video = has_video(input_path)
    if target_lufs is None:
        target_lufs = VIDEO_LUFS if video else AUDIO_LUFS

    with tempfile.TemporaryDirectory(prefix="qtip-") as temporary:
        temp = Path(temporary)
        extracted = temp / "input.wav"
        cleaned = temp / "cleaned.wav"
        normalized = temp / "normalized.wav"

        print("extracting audio...")
        extract_audio(input_path, extracted)
        denoise(extracted, cleaned)
        print(f"normalizing to {target_lufs:g} LUFS...")
        normalize(cleaned, normalized, target_lufs)
        print("writing output...")
        write_output(input_path, normalized, output_path, video)


def lufs(value: str) -> float:
    parsed = float(value)
    if not -70 <= parsed <= -5:
        raise argparse.ArgumentTypeError("must be between -70 and -5")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qtip",
        description="denoise voice audio with MossFormer2 and normalize its loudness",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--lufs",
        type=lufs,
        default=None,
        metavar="TARGET",
        help="target integrated loudness (default: -14 for video, -16 for audio)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="overwrite the output"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.force:
        print(
            f"qtip: output already exists: {output_path} (use --force)", file=sys.stderr
        )
        raise SystemExit(2)

    try:
        process(input_path, output_path, args.lufs)
    except (QtipError, OSError, KeyError, ValueError) as error:
        print(f"qtip: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"done: {output_path}")
