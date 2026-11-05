<h2 id="en-v1.3.9">Highlights</h2>

- Added official **`mpt` pip packaging**. Start the WebUI with `mpt webui` or generate videos with `mpt cli`.
- Added **static and slow zoom image effects** for local and AI-generated images. Slow zoom remains the default.
- Added Kimi Code subscription access with browser authorization, alongside improvements to the sign-in dialog, status updates, and cancellation.
- Added an optional **local SDXL image server** for the OpenAI-compatible image material workflow, with idle GPU offloading.
- Added Opper, onomeo, Muse Spark, and EmpirioLabs AI LLM providers, and completed Brazilian Portuguese WebUI translations.
- Added **diagnostic logging controls** in the WebUI, including runtime log-level changes and opt-in API request tracing with redaction and size limits.

## Fixes and Improvements

- Fixed grayscale image brightness, transparent image backgrounds, zoom mask alignment, and subtitle background colors. Improved transitions on short clips, background music fade-out, and special-character handling in concatenation paths.
- Isolated intermediate image files across concurrent jobs, improved image cache identity, and validated cached downloads. Completed videos are retained when another video in the same batch fails.
- Fixed speech speed, volume, and audio container handling across several providers. Improved remote audio validation and preserved existing audio until replacement output is ready.
- Improved subtitle timestamp rounding, long-word wrapping, Unicode handling, and pause-tag recognition. Symbol-only lines and separators no longer distort estimated subtitle alignment.
- Fixed Chinese and Japanese fallback social titles using the entire script. Improved JSON extraction, reasoning removal, Markdown cleanup, and generated search-term validation.
- Expanded supported local upload formats, including multi-picture camera JPEGs. Improved task history checks, deletion state, subtitle previews, and model connection error messages in the WebUI.
- Added HEAD support for generated artifacts and improved byte-range downloads, validation errors, and Redis task updates and deletion safeguards.
- Reduced unsafe speech resubmissions after lost responses or redirects, preserved Upload-Post tracking handles, and improved WaveSpeed submission error detection.
- Updated Edge TTS to address 403 failures, constrained PyAV for Whisper compatibility, and fixed Windows cache cleanup and UTF-8 BOM handling in locale files.
- Improved the agent helper's configuration handling and preflight checks, including workflows with prepared narration and visuals that do not need an LLM.

## Usage and Upgrade Notes

- The official PyPI package is **`mpt`**, maintained by this project, and requires **Python 3.11 or later**. Install or upgrade with `python -m pip install --upgrade mpt`; a virtual environment is recommended. The separately named `MoneyPrinterTurbo` package is a third-party distribution.
- The pip installation stores configuration and generated files in your user directory by default; set `MPT_HOME` to choose another location. Source installations continue to use the project directory. When switching installation methods, migrate your configuration and required resources to the chosen runtime directory.
- Image motion defaults to **slow zoom** for existing users. For a still image without transition effects, select **Static** for image motion and **None** for transitions. The setting is available in the WebUI, CLI, and API.
- Kimi Code requires browser sign-in and uses your plan's usage limits. The other new LLM providers require their corresponding API credentials and configuration.
- The local SDXL server requires separate dependencies, model downloads, and a running GPU service; it is not started automatically with MPT. Its default setup unloads Ollama models before image generation, so avoid concurrent LLM and image work on the same GPU. See the [setup instructions](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/local-image-server/README.md).
- API request tracing is **off by default**. Prompts and business parameters may appear in logs when enabled; turn tracing off after troubleshooting and check for private content before sharing logs. See the [diagnostic logging guide](https://github.com/harry0703/MoneyPrinterTurbo/blob/v1.3.9/docs/diagnostic-logging.md).

## What's Changed

- Official pip package with WebUI and CLI launchers — [c3634de](https://github.com/harry0703/MoneyPrinterTurbo/commit/c3634de097ea4951935d22e94580ff0ad2665503).
- Static and slow zoom image options — [90f7701](https://github.com/harry0703/MoneyPrinterTurbo/commit/90f77011d7e3b46d382e6d0180dabfb6725ffee4).
- Kimi Code subscription provider — @ysntony in [#1599](https://github.com/harry0703/MoneyPrinterTurbo/pull/1599), followed by [sign-in flow improvements](https://github.com/harry0703/MoneyPrinterTurbo/commit/4204d0f).
- Local SDXL image server — @imkashif in [#1668](https://github.com/harry0703/MoneyPrinterTurbo/pull/1668).
- Opper LLM provider — @Felixkw12 in [#1589](https://github.com/harry0703/MoneyPrinterTurbo/pull/1589).
- onomeo LLM provider — @Owaid0723666 in [#1600](https://github.com/harry0703/MoneyPrinterTurbo/pull/1600).
- Muse Spark provider and Brazilian Portuguese translations — @humbertocortezia in [#1657](https://github.com/harry0703/MoneyPrinterTurbo/pull/1657).
- EmpirioLabs AI provider — @Adam-Dalloul in [#1694](https://github.com/harry0703/MoneyPrinterTurbo/pull/1694).
- Runtime diagnostic logging and request tracing — [6940611](https://github.com/harry0703/MoneyPrinterTurbo/commit/6940611cc2791d1ca13b3af7ee893ccd9544b874).
- Chinese and Japanese fallback social titles — [e25326e](https://github.com/harry0703/MoneyPrinterTurbo/commit/e25326e).
- Retained completed videos after batch failures — @hdu-r3drum in [#1597](https://github.com/harry0703/MoneyPrinterTurbo/pull/1597).
- Edge TTS 403 fix — @HEOJUNFO in [#1652](https://github.com/harry0703/MoneyPrinterTurbo/pull/1652).
- Whisper/PyAV compatibility — @cristianchiriac in [#1582](https://github.com/harry0703/MoneyPrinterTurbo/pull/1582).
- Windows cache cleanup — @shan3520 in [#1654](https://github.com/harry0703/MoneyPrinterTurbo/pull/1654).
- UTF-8 BOM locale support — @ketpatil77 in [#1658](https://github.com/harry0703/MoneyPrinterTurbo/pull/1658).
- Task protection for unparseable states — @RerankerGuo in [#1663](https://github.com/harry0703/MoneyPrinterTurbo/pull/1663).
- Clearer model connection errors — @fengqiaoqu in [#1645](https://github.com/harry0703/MoneyPrinterTurbo/pull/1645).
- Local material preflight checks — @drakeo338 in [#1585](https://github.com/harry0703/MoneyPrinterTurbo/pull/1585); preserved Redis task IDs — @ac1982 in [#1588](https://github.com/harry0703/MoneyPrinterTurbo/pull/1588).
- Grayscale rendering, artifact HEAD requests, and prepared-media agent workflows — @rudycelekli in [#1620](https://github.com/harry0703/MoneyPrinterTurbo/pull/1620), [#1636](https://github.com/harry0703/MoneyPrinterTurbo/pull/1636), and [#1608](https://github.com/harry0703/MoneyPrinterTurbo/pull/1608), alongside many subtitle, speech, material, API, and task reliability fixes.

Thanks to all contributors and everyone who reported issues and helped test these changes!

**Full Changelog**: [v1.3.8...v1.3.9](https://github.com/harry0703/MoneyPrinterTurbo/compare/v1.3.8...v1.3.9)
