# Diagnostic logging

In the WebUI, open **Settings → Interface Settings → Diagnostic Logging**, choose a
log level, enable **Log API request parameters** as needed, and click
**Apply logging settings**. The setting applies immediately to the current service,
including the remaining logs of running tasks; all browser sessions of the same
service share it. Turning on "Hide logs" only hides logs in the interface; it does
not stop terminal output.

The API, CLI and WebUI use the same configuration:

```toml
[diagnostics]
log_level = "INFO"
log_requests = false
```

`log_level` accepts TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR and CRITICAL. When
omitted, the previous top-level `log_level` is used, defaulting to DEBUG. Changes
made by hand require a restart; "Apply" in the WebUI only updates the current process
immediately and cannot remotely change other API or CLI processes.

Request tracing is a separate switch and can also be enabled at the ERROR level. It
is off by default and records:

- Request ID, HTTP method, redacted address, and length-limited request parameters.
- Response status (when available), elapsed time, or the exception type on failure.
- LLM SDK call arguments. These are the SDK inputs, not the raw JSON the SDK
  ultimately sends; automatic SDK retries are not shown one by one. For streaming
  requests, the measured time only covers obtaining the response or stream object,
  not the subsequent download.

It covers material, video, speech, publishing and login requests that the project
explicitly makes with `requests`, plus OpenAI-compatible, Azure OpenAI, Qwen, Gemini
and LiteLLM script calls. It does not capture the browser's own network traffic,
inbound API bodies, CLI subprocesses, the Edge TTS WebSocket or other SDK-internal
transport. Normal application logs and each third-party SDK's own log levels are
separate switches.

Request headers, response bodies and uploaded file contents are never logged.
Secrets, tokens, cookies, passwords and signature fields are redacted; URL query
parameter values are hidden uniformly; audio, video, base64 and similar content is
omitted. Raw unstructured request bodies are not recorded. Prompts and other
business parameters may appear in logs, so turn tracing off after troubleshooting,
and still check logs for private content before sharing them. Diagnostics never send
logs externally and never make extra requests, retries or replays.

CLI logs still go to stderr, while stdout remains machine-readable JSON. WebUI task
logs keep the existing thread-ownership filtering and entry limit; changing levels
does not remove the log handlers of running tasks.
