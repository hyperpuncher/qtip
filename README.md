# qtip

clean and normalize voice audio in audio and video files with [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio)'s `MossFormer2_SE_48K` model.

## requirements

- [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe`

qtip runs on cpu and supports Nvidia CUDA and AMD ROCm GPUs. the model is downloaded to the qtip cache on first use.

## usage

```sh
qtip input.mp4 output.mp4
```

qtip uses two-pass loudness normalization with conventional defaults:

- `-14 LUFS` for video
- `-16 LUFS` for audio-only files
- `-1.5 dBTP` maximum true peak

override the automatic target when needed:

```sh
qtip --lufs -12 input.mp4 output.mp4
```

existing outputs are protected unless `--force` is passed.

### options

| option            | description                            |
| ----------------- | -------------------------------------- |
| `--lufs <target>` | override the automatic loudness target |
| `-f`, `--force`   | overwrite an existing output           |
| `-h`, `--help`    | show help                              |

## install

cpu (smallest download):

```sh
uv tool install --torch-backend cpu git+https://github.com/hyperpuncher/qtip
```

gpu (automatically selects Nvidia CUDA or AMD ROCm):

```sh
uv tool install --torch-backend auto git+https://github.com/hyperpuncher/qtip
```

gpu installations are several gigabytes. for short files, cpu processing may be faster due to gpu startup overhead.

qtip extracts audio at 48 khz, denoises it with MossFormer2, normalizes it, and replaces the original audio. video is copied without re-encoding.
