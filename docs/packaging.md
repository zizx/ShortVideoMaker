# pip Package and PyPI Releases

The distribution package name is `mpt`; the project name remains MoneyPrinterTurbo.
The version currently pending release is `1.3.9`. `1.3.8` was published to
[PyPI](https://pypi.org/project/mpt/1.3.8/) by `harry0703` on 2026-10-16 as both a
wheel and an sdist. Published files cannot be overwritten, so later code changes
require a version bump, a rebuild and fresh verification.

## Build and install

Python 3.11 or newer is required. Dependencies are resolved and installed with the
wheel. FFmpeg prefers the system installation, but the binary provided by
imageio-ffmpeg can also be used.

Users install from PyPI and start the application:

```sh
python -m pip install mpt
mpt webui
mpt cli --help
```

Build and verify locally:

```sh
uv build --out-dir storage/package/mpt-dist
python -m pip install ./storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl
mpt --help
mpt webui
mpt cli --help
```

Before uploading, confirm the PyPI account, target version and artifact hashes, and
avoid batch-uploading from a directory that still contains old packages. Every
release requires explicit user authorization; a commit or push never publishes a
release automatically.

## Pre-release checks

```sh
uvx twine check --strict \
  storage/package/mpt-dist/mpt-1.3.9-py3-none-any.whl \
  storage/package/mpt-dist/mpt-1.3.9.tar.gz
```

Check the PyPI description page, package metadata and dependency installability, and
complete a video generation in a fresh virtual environment. Publish only after
explicit authorization; account passwords, verification codes and API tokens are
never written to the repository, command examples or logs. Each language README
provides the user installation entry point, while the PyPI description is maintained
separately in `docs/pypi-description.md`; updating the repository documentation does
not change the already published PyPI page.

`mpt webui` listens on `127.0.0.1` only by default, picks a free port starting from
8501, and opens a browser automatically. Use `--host` and `--port` to choose the
address, and `--headless` to prevent the automatic browser launch:

```sh
mpt webui --port 8503 --headless
```

The CLI reuses the arguments, task pipeline, exit codes and JSON output of
`python cli.py`. Full offline example:

```sh
mpt cli --video-script "A short video from local materials." \
  --video-source local --video-materials /absolute/path/clip.mp4 \
  --custom-audio-file /absolute/path/narration.wav \
  --no-subtitle-enabled --bgm-type none
```

Online script, material and TTS features still need the corresponding service
configuration, credentials and network access. Whisper models are loaded on demand
and are not bundled in the wheel. CLI logs go to stderr; on success, stdout prints
JSON containing the video path.

## Configuration and data

Source runs keep using the project root. The installed package does not depend on the
current working directory; its default runtime directory is:

| Platform | Runtime directory |
| --- | --- |
| macOS | `~/Library/Application Support/MoneyPrinterTurbo` |
| Windows | `%LOCALAPPDATA%\MoneyPrinterTurbo` |
| Linux | `$XDG_DATA_HOME/MoneyPrinterTurbo`, defaulting to `~/.local/share/MoneyPrinterTurbo` |

Set `MPT_HOME` to choose the runtime directory explicitly (the source version also
supports it). Within it:

- `config.toml`: initialized on first run from a template that contains no
  credentials; later starts never overwrite an existing configuration.
- `storage/tasks`: tasks and generated videos; other caches and uploaded files live
  under `storage`.
- `resource/fonts`, `resource/songs`: built-in resources are copied on first use; you
  can add your own files.
- `models`: the place for manually downloaded Whisper models.

User resources are only filled in where files are missing; upgrades never overwrite
existing files. The installed package includes two default Chinese fonts, a
Vietnamese font and built-in background music; the fallback `STHeitiLight.ttc` and
`MicrosoftYaHeiNormal.ttc` are excluded from the distribution to control its size.
Add those files to `resource/fonts` in the runtime directory if you need those
weights. The source version keeps its original resources and appearance.

## Local verification requirements

Rebuild the wheel from the sdist, install it into a fresh virtual environment that
does not inherit system packages, leave the source directory, clear `PYTHONPATH`, and
verify `mpt cli --help`, the actual WebUI pages and video generation. The output video
must decode fully with FFmpeg, and confirm that the configuration and artifacts were
not written into site-packages. The distribution must not contain a local
`config.toml`, secrets, caches, task artifacts or local collaboration instructions.
