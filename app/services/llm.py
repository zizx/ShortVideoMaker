import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider
from app.utils import utils
from app.utils.request_logging import trace_sdk_call

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
# The "test model connection" action sends only one tiny request. The OpenAI
# SDK defaults to a 600-second timeout with 2 automatic retries, so when the
# endpoint is unreachable (proxy, firewall or DNS failure) the test button
# hangs for a long time and finally leaves only "Request timed out." (see
# issue #1642). Give the connection test a bounded short timeout and disable
# SDK retries so failures are fast and diagnosable; the regular script
# generation path keeps the SDK defaults so slow reasoning models and local
# Ollama cold starts are not penalized.
LLM_CONNECTION_TEST_TIMEOUT_SECONDS = 30.0
LLM_CONNECTION_TEST_MAX_RETRIES = 0
# These fragments come from OpenAI SDK APITimeoutError / APIConnectionError
# and httpx's low-level connection error messages; any match is treated as
# "endpoint unreachable".
_CONNECTION_FAILURE_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to connect",
)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_UNOPENED_THINK_BLOCK_RE = re.compile(r"^.*</think\s*>", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]+@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()

# The Claude Code CLI uses a coding-agent system prompt by default whose many
# constraints are unrelated to copywriting and would steer script and keyword
# generation off target, so replace it entirely when calling the CLI.
CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are a concise copywriter. Follow the user's instructions and output "
    "format exactly, and output nothing else."
)
CLAUDE_CODE_DEFAULT_TIMEOUT = 300.0
# `--tools ""` disables all built-in tools, and `--safe-mode` disables all
# user-level customization such as CLAUDE.md, skills, hooks, plugins and MCP,
# while keeping auth, model selection and permissions working. Both need a
# recent CLI; older versions exit with "unknown option", which the caller
# turns into a clear message.
CLAUDE_CODE_MIN_CLI_VERSION = "2.1.260"
# These environment variables make the CLI switch to an API key or a
# third-party provider (Bedrock, Vertex, Foundry, Mantle, Gateway, etc.),
# bypassing the subscription login and incurring extra billing. Listing them
# one by one is error-prone and the CLI keeps adding providers, so remove
# whole classes by prefix:
#   ANTHROPIC_*           API key, auth token, base URL, vendor endpoints, profile
#   CLAUDE_CODE_USE_*     provider switches
#   CLAUDE_CODE_SKIP_*_AUTH   switches that skip provider auth
CLAUDE_CODE_CONFLICTING_ENV_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
CLAUDE_CODE_CONFLICTING_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_GATEWAY_TOKEN_FILE_DESCRIPTOR",
)
# These two kinds of variables must not be removed:
#   CLAUDE_CODE_OAUTH_TOKEN is the only subscription auth method in the
#   container (it does not match the prefixes above);
#   *_CONFIG_DIR only points at where credentials are stored; removing it
#   would instead break an already logged-in subscription.
CLAUDE_CODE_PRESERVED_ENV_VARS = (
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "CLAUDE_CONFIG_DIR",
)


def _is_conflicting_claude_code_env(name: str) -> bool:
    """Return whether an environment variable would switch the CLI away from
    subscription login to another auth method."""
    if name in CLAUDE_CODE_PRESERVED_ENV_VARS:
        return False
    if name in CLAUDE_CODE_CONFLICTING_ENV_VARS:
        return True
    if name.startswith(CLAUDE_CODE_CONFLICTING_ENV_PREFIXES):
        return True
    return name.startswith("CLAUDE_CODE_SKIP_") and name.endswith("_AUTH")


def coerce_claude_code_timeout(value, config_key: str = "claude_code_timeout"):
    """
    Parse the configured timeout value into a positive, finite number of
    seconds.

    TOML may write it as `claude_code_timeout = 300` (int/float) or as
    `"300"` (string), so `strip()` cannot be called on the raw value. nan / inf
    would make `subprocess.run(timeout=...)` block forever, so reject them too.
    """
    if value is None:
        return CLAUDE_CODE_DEFAULT_TIMEOUT

    if isinstance(value, bool):
        # bool is a subclass of int, but True seconds is obviously not a
        # timeout the user wants.
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return CLAUDE_CODE_DEFAULT_TIMEOUT
        try:
            seconds = float(text)
        except ValueError:
            raise ValueError(
                f"{config_key} must be a number of seconds, got {value!r}"
            ) from None
    elif isinstance(value, (int, float)):
        seconds = float(value)
    else:
        raise ValueError(f"{config_key} must be a number of seconds, got {value!r}")

    if not math.isfinite(seconds):
        raise ValueError(f"{config_key} must be a finite number, got {value!r}")
    if seconds <= 0:
        raise ValueError(f"{config_key} must be greater than 0, got {value!r}")
    return seconds


def _resolve_provider_field_value(raw_value, default_value):
    """
    Fall back to the Registry default only when the value is "not configured".

    The previous `raw or default_value` also replaced legitimate values such as
    0 and false: `claude_code_timeout = 0` was silently changed to 300, while
    `"0"` raised an error. The default now applies only for None or a blank
    string, keeping config validation consistent across all notations.
    """
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and not raw_value.strip():
        return default_value
    return raw_value


def build_claude_code_env(base_env=None):
    """
    Build a subprocess environment that relies only on subscription login.

    Returns (environment dict, list of removed variable names). Removed are the
    variables that switch the auth method or provider; `CLAUDE_CODE_OAUTH_TOKEN`
    must be kept: there is no keychain in the container, so the CLI can only
    complete subscription auth with it.
    """
    env = dict(os.environ if base_env is None else base_env)
    removed = sorted(name for name in env if _is_conflicting_claude_code_env(name))
    for name in removed:
        env.pop(name, None)
    return env, removed


def _normalize_text_response(content, llm_provider: str) -> str:
    # Different LLM SDKs may return None, an empty string or even a non-string
    # object on errors or when blocked. Validate uniformly here so later direct
    # `.replace()` calls do not raise attribute errors such as `NoneType`.
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # Reasoning models such as MiniMax M3 and DeepSeek R1 may wrap their
    # internal reasoning in `<think>...</think>` in the response. Video scripts
    # and keywords only need the final speakable text; without uniform cleanup
    # at the service layer, the WebUI, subtitles and voiceover would treat the
    # reasoning as content.
    content = _THINK_BLOCK_RE.sub("", content)
    # Chat templates of models such as DeepSeek R1 and QwQ put the opening
    # `<think>` into the prompt, so the body returned by gateways such as
    # vLLM/Ollama is only `reasoning...</think>answer`. Everything before a
    # closing tag that remains after paired cleanup is reasoning content and
    # must not enter the script.
    content = _UNOPENED_THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # The preceding ``strip()`` already trimmed leading and trailing
    # whitespace. Single and double newlines inside the body must be preserved:
    # script generation relies on double newlines to separate paragraphs, and
    # subtitle handling reads user copy line by line.
    return content


def _sanitize_error_message(error: object) -> str:
    """
    Sanitize error messages returned to the WebUI/API so credentials in a
    custom base_url do not leak.

    Some OpenAI-compatible SDKs splice the request URL verbatim into exception
    messages. If a user configures `https://user:pass@example.com/v1` for a
    proxy gateway, returning `str(e)` directly would expose the password to the
    page, API callers or subsequent logs. Only the error text is handled here;
    the actual request address is unchanged so the normal call path is not
    affected.
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI-compatible endpoints may, on error, return a response object with
    # no choices, or with empty choices/message/content. Validate the structure
    # uniformly here to avoid low-level attribute errors such as
    # `NoneType is not subscriptable`.
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """Read a field from both dicts and SDK response objects."""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _sdk_transport_options(request_timeout, max_retries) -> dict:
    """Convert optional timeout/retry constraints into OpenAI SDK client
    constructor arguments.

    Passing None for both returns an empty dict, so the client keeps the SDK
    defaults (600-second timeout, 2 automatic retries) and regular paths such
    as script generation behave as before; only callers that need fast failure,
    such as the connection test, explicitly pass bounded values to tighten the
    request budget.
    """
    options = {}
    if request_timeout is not None:
        options["timeout"] = request_timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    return options


def _explain_connection_test_failure(error_message: str) -> str:
    """Add troubleshooting hints for timeout/connection failures instead of
    showing only the raw SDK message.

    In issue #1642 the user's API itself worked; the failure was a link problem
    between this machine and the endpoint (proxy, firewall or DNS), but
    "Request timed out." did not point at any possible cause. Other errors
    (auth, quota, etc.) are returned unchanged.
    """
    normalized = (error_message or "").lower()
    if not any(marker in normalized for marker in _CONNECTION_FAILURE_MARKERS):
        return error_message
    return (
        f"{error_message} (the request did not complete within "
        f"{LLM_CONNECTION_TEST_TIMEOUT_SECONDS:.0f}s). Please check that the "
        "configured base URL is reachable from this machine, and that no proxy "
        "or firewall is blocking the request."
    )


def _extract_qwen_generation_text(response) -> str:
    """
    Extract text from a DashScope Generation response.

    When called with `messages`, Qwen returns the chat structure
    `output.choices[0].message.content`; only the legacy completion form
    returns `output.text`. Both paths are supported here so a None
    `output.text` does not lead to `.replace()` raising an undiagnosable
    AttributeError.
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(
    prompt: str,
    app_config=None,
    request_timeout: float | None = None,
    max_retries: int | None = None,
) -> str:
    sdk_client = None
    sdk_stream = None
    try:
        # The WebUI lets users prepare the next script while video generation
        # runs. Callers can pass a config snapshot from submission time so a
        # retrying model request does not switch to another Provider, Base URL
        # or model when a background task finishes and applies new config.
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Ollama's default address depends on whether it runs in a container
        # and cannot be stored as a static Registry value; the Registry still
        # owns the model and required-field rules, while environment
        # differences are resolved here.
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: _resolve_provider_field_value(
                runtime_app_config.get(provider.config_key(field.config_suffix)),
                field.default_value,
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            response = trace_sdk_call(
                llm_provider, base_url, dashscope.Generation.call,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # The new google-genai exposes model services through a unified
                # Client. The context manager closes the underlying HTTP
                # connections after the request, avoiding accumulated
                # connection resources during frequent generation.
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = trace_sdk_call(
                        llm_provider, base_url, client.models.generate_content,
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Cloudflare's currently recommended AI Gateway REST API is
            # OpenAI SDK compatible. The Account ID builds the unified endpoint
            # and the Gateway ID is selected via a header; the Workers AI
            # `/ai/run/{model}` specific endpoint is no longer used.
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = trace_sdk_call(
                llm_provider, base_url, litellm.completion,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # The Azure OpenAI SDK uses `azure_endpoint` and `api_version` to
            # build its dedicated request URL and cannot reuse the plain
            # OpenAI-compatible `base_url` initialization below. Finish the
            # request inside the Azure branch and return immediately so the
            # client is not overwritten by the later fallback, which would
            # validate the user's Azure credentials but never use them.
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = sdk_client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter in {"openai_codex", "openai_chatgpt"}:
            if adapter == "openai_chatgpt":
                from app.services import chatgpt_oauth as subscription_provider
            else:
                from app.services import codex_oauth as subscription_provider

            timeout_seconds = coerce_claude_code_timeout(
                extra_values.get("timeout"), provider.config_key("timeout")
            )
            if request_timeout is not None:
                timeout_seconds = min(timeout_seconds, request_timeout)
            return _normalize_text_response(
                subscription_provider.generate_text(prompt, model_name, timeout_seconds),
                llm_provider,
            )

        if adapter == "claude_code":
            # Claude subscriptions (Pro / Max / Team) do not issue API keys;
            # their credentials can only be used by the official Claude Code
            # client. Instead of calling the Anthropic API directly, invoke the
            # locally logged-in claude CLI (`claude -p`) in headless mode, let
            # the CLI handle auth, and consume only the text it returns.
            configured_cli = (extra_values.get("cli_path") or "").strip() or "claude"
            cli_path = shutil.which(configured_cli)
            if not cli_path and os.path.isfile(configured_cli):
                cli_path = configured_cli
            if not cli_path:
                raise ValueError(
                    f"{llm_provider}: claude CLI not found ('{configured_cli}'), "
                    f"install it in the runtime or set "
                    f"{provider.config_key('cli_path')} in the config.toml file."
                )

            try:
                timeout_seconds = coerce_claude_code_timeout(
                    extra_values.get("timeout"), provider.config_key("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            # The prompt goes in via stdin rather than the command line: on
            # Windows the npm-installed claude is claude.cmd, and cmd.exe
            # truncates arguments at the first newline, so a multi-line prompt
            # and the isolation flags after it would be lost.
            command = [
                cli_path,
                "-p",
                "--output-format",
                "json",
                "--system-prompt",
                CLAUDE_CODE_SYSTEM_PROMPT,
                # Disable all built-in tools so only text generation happens.
                "--tools",
                "",
                # Disable user-level customization such as CLAUDE.md, skills,
                # hooks, plugins and MCP; auth and model selection are
                # unaffected (do not use --bare, which disables OAuth).
                "--safe-mode",
            ]
            # Leave the model name empty to use the CLI's own default, so a
            # hard-coded model ID here does not break as subscription models
            # change.
            if model_name:
                command += ["--model", model_name]

            cli_env, removed_env = build_claude_code_env()
            if removed_env:
                # Log only variable names, never values, so secrets do not end
                # up in logs.
                logger.warning(
                    f"{llm_provider}: ignoring conflicting environment variables "
                    f"so the subscription login is used: {', '.join(removed_env)}"
                )

            logger.info(f"invoking claude cli, model: {model_name or 'cli default'}")
            # The CLI reads CLAUDE.md and project settings from the working
            # directory, which would contaminate the copy, so run it in a
            # fixed empty temporary directory.
            with tempfile.TemporaryDirectory() as work_dir:
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        # The CLI always emits UTF-8. Without an explicit encoding,
                        # text=True decodes with the system locale (e.g. cp1252 on
                        # non-English Windows), so every non-ASCII character reaches
                        # the script as mojibake.
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout_seconds,
                        cwd=work_dir,
                        env=cli_env,
                    )
                except subprocess.TimeoutExpired:
                    raise Exception(
                        f"[{llm_provider}] claude cli timed out after "
                        f"{timeout_seconds:.0f}s"
                    )

            # Failures such as not being logged in or exhausted usage also
            # return JSON (`is_error` true, `result` the readable reason), just
            # with a non-zero exit code. So parse stdout first and fall back to
            # the exit code and stderr only when no JSON is available.
            stdout = (completed.stdout or "").strip()
            try:
                payload = json.loads(stdout) if stdout else None
            except json.JSONDecodeError:
                payload = None

            if payload is None:
                detail = (completed.stderr or stdout or "").strip()
                if "unknown option" in detail.lower():
                    raise Exception(
                        f"[{llm_provider}] the installed claude CLI does not support "
                        f"the required isolation flags; upgrade to "
                        f"{CLAUDE_CODE_MIN_CLI_VERSION} or newer: {detail[:300]}"
                    )
                if completed.returncode != 0:
                    raise Exception(
                        f"[{llm_provider}] claude cli exited with code "
                        f"{completed.returncode}: {detail[:500]}"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{detail[:500]}"'
                )

            if payload.get("is_error") or completed.returncode != 0:
                reason = str(payload.get("result") or "").strip() or (
                    f"claude cli exited with code {completed.returncode}"
                )
                # Interactive /login cannot run in the container, so point at
                # a usable auth method.
                if "login" in reason.lower():
                    reason += (
                        " (run `claude setup-token` on the host and pass the token "
                        "to the container as CLAUDE_CODE_OAUTH_TOKEN)"
                    )
                raise Exception(
                    f'[{llm_provider}] returned an error response: "{reason[:500]}"'
                )

            return _normalize_text_response(payload.get("result"), llm_provider)

        if adapter == "kimi_code":
            # The Kimi subscription (Kimi Code benefit of Plus+) does not issue
            # a metered API key; auth is done via OAuth device authorization:
            # the WebUI guides the user through a one-click browser
            # confirmation and stores a refresh token locally; the access token
            # expires after 15 minutes and kimi_code_oauth renews it
            # transparently. Requests go to the official OpenAI-compatible
            # endpoint, a separate quota from the metered API.
            from app.services import kimi_code_oauth

            try:
                timeout_seconds = kimi_code_oauth.coerce_timeout(
                    extra_values.get("timeout")
                )
            except ValueError as timeout_error:
                raise ValueError(f"{llm_provider}: {timeout_error}") from None

            access_token = kimi_code_oauth.get_valid_access_token()
            # The authorized region governs the request: a token is only valid
            # for the region that issued it. If a user switches API Platform
            # after authorizing, sending by UI region would hit a China-issued
            # token against the Global endpoint (guaranteed 401), so prefer the
            # token's issuing region.
            authorized_region = kimi_code_oauth.authorized_region()
            if authorized_region:
                request_base_url = kimi_code_oauth.api_base_url(authorized_region)
                if base_url and authorized_region != kimi_code_oauth.resolve_region(
                    base_url
                ):
                    logger.warning(
                        f"kimi_code: API Platform selection differs from the "
                        f"authorized region; using '{authorized_region}' to match "
                        "the issued token"
                    )
            else:
                request_base_url = base_url or kimi_code_oauth.api_base_url(
                    kimi_code_oauth.resolve_region(base_url)
                )
            # When the model name is empty, default to kimi-for-coding, the
            # dedicated coding model included in the subscription, so a
            # hard-coded general model id does not break as benefits change.
            request_model = model_name or "kimi-for-coding"
            logger.info(f"requesting kimi code chat completion, model: {request_model}")
            # The provider's own lenient timeout must be tightened to the test
            # budget for connection tests.
            effective_timeout = timeout_seconds
            if request_timeout is not None:
                effective_timeout = min(effective_timeout, request_timeout)
            client = sdk_client = OpenAI(
                api_key=access_token,
                base_url=request_base_url,
                timeout=effective_timeout,
                # The server's anti-scraping pipeline requires the same device
                # identity headers as the CLI.
                default_headers=kimi_code_oauth.api_headers(),
                **_sdk_transport_options(None, max_retries),
            )
            response = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=request_model, messages=[{"role": "user", "content": prompt}]
            )
            if response and isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            raise Exception(
                f'[{llm_provider}] returned an invalid response: "{response}"'
            )

        if adapter == "modelscope":
            content = ""
            client = sdk_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                **_sdk_transport_options(request_timeout, max_retries),
            )
            response = sdk_stream = trace_sdk_call(
                llm_provider, base_url, client.chat.completions.create,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client = sdk_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **_sdk_transport_options(request_timeout, max_retries),
        )

        response = trace_sdk_call(
            llm_provider, base_url, client.chat.completions.create,
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"

    finally:
        for resource in (sdk_stream, sdk_client):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as cleanup_error:
                    logger.warning(
                        f"could not close LLM transport: {type(cleanup_error).__name__}"
                    )


def test_connection() -> tuple[bool, str, float]:
    """
    Send one minimal request with the current provider config to verify the
    real generation path works.

    The connection test reuses `_generate_response()`, so it covers API key,
    base URL, model name and provider-specific fields, but it does not enter
    the script-generation retry logic and never sends the user's video subject
    or script. The request uses a bounded short timeout and disables SDK
    automatic retries: when the endpoint is unreachable (proxy, firewall or DNS
    failure) it fails fast with troubleshooting hints instead of hanging for
    minutes and leaving only "Request timed out." (issue #1642). The return
    values are success status, error message and request duration.
    """
    started_at = perf_counter()
    response = _generate_response(
        prompt="Reply with exactly: OK",
        request_timeout=LLM_CONNECTION_TEST_TIMEOUT_SECONDS,
        max_retries=LLM_CONNECTION_TEST_MAX_RETRIES,
    )
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = _explain_connection_test_failure(
            response.removeprefix("Error:").strip()
        )
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # The API layer already validates length with Pydantic; this fallback keeps
    # direct generate_script calls from the WebUI or internal services from
    # sending overlong prompts to the model, avoiding abnormal token costs and
    # failed requests.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # Both the WebUI and the API limit the range; this fallback covers
        # internal calls so bad parameters cannot inflate LLM cost or produce
        # empty results.
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # Build "script generation rules" and "runtime context" separately. This
    # way advanced users who override the default system prompt still keep the
    # video subject, language and paragraph count that every generation needs.
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove emphasis and heading markers without changing inline names
        # or musical notes such as C#, F# and G#.
        response = response.replace("*", "")
        response = re.sub(
            r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*$",
            lambda match: re.sub(r"[ \t]+#+[ \t]*$", "", match.group(1)),
            response,
        )

        # Link labels are narration; discard destinations before stripping
        # bracketed stage directions. Common parenthesized URL components fit.
        response = re.sub(
            r"\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)",
            lambda match: match.group(1),
            response,
        )

        # Remove markdown syntax.  Use non-greedy .*? so each bracket/paren
        # group is removed independently; the greedy form would eat all text
        # between the first opener and the last closer on the same line.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if isinstance(response, str) and response.startswith("Error: "):
                # _generate_response returns provider failures as text. Passing
                # that text through would make the task treat it as narration.
                raise ValueError(response)
            if response:
                candidate = format_response(response).strip()
            else:
                logging.error("gpt returned an empty response")
                candidate = ""

            # Some upstream providers may return quota errors as plain text.
            if candidate and "当日额度已消耗完" in candidate:
                raise ValueError(candidate)

            if candidate:
                final_script = candidate
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if not final_script:
        logger.error("failed to generate video script after retries")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_embedded_json(text: str, opening: str):
    """Decode one complete JSON value without absorbing later prose/examples."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text or ""):
        if character != opening:
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("response contains no complete JSON value")


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    video_script = utils.remove_pause_tags(video_script or "").strip()
    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # In ordered-keyword mode the number of examples must match amount so
        # the model is not misled by a fixed four examples, which would make a
        # long script return too few keywords and hurt material coverage.
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        search_terms = []
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # The public return type of generate_terms is List[str]. If the
                # provider's error text were returned as-is, downstream code
                # that only checks for emptiness would treat a non-empty string
                # as success, and the material download loop would iterate the
                # error text character by character, causing pointless external
                # requests. Return an empty list instead so the task
                # orchestrator stops immediately at the real failure point.
                logger.error(f"failed to generate video terms: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                try:
                    search_terms = _extract_embedded_json(response, "[")
                except ValueError as e:
                    logger.warning(f"failed to generate video terms: {str(e)}")

        # Apply the same contract to direct JSON and prose-wrapped recovery.
        # Otherwise a nonempty array of numbers or objects reaches material search.
        if not isinstance(search_terms, list) or not all(
            isinstance(term, str) for term in search_terms
        ):
            logger.error("response is not a list of strings.")
            search_terms = []

        # The model may ignore the requested count or return blank strings.
        # Enforce the prompt contract before material providers are contacted;
        # repeated topics remain meaningful in chronological mode.
        search_terms = [term.strip() for term in search_terms if term.strip()][:amount]
        if search_terms:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Social publishing metadata
#
# Generate the title, caption and hashtags commonly used when publishing a
# video to short-video platforms, based on the video subject and script. This
# capability only reuses the existing LLM providers; it integrates no external
# publishing service and does not affect the main video generation path.
# =============================================================================

# Different platforms prefer different copy lengths and hashtag counts. Use
# conservative limits here so callers do not have to trim overly long model
# output a second time.
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# Generic fallback tags used when the LLM is unavailable. They deliberately do
# not bind to a country or language, so the API returns a usable structure for
# Chinese, English, Vietnamese and other scenarios.
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # The API layer limits length; this fallback keeps internal calls or future
    # direct WebUI calls from sending overlong content to the model, avoiding
    # abnormal token costs.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    Normalize hashtags returned by the LLM into `#tag` format.

    The LLM may return a string, an array, phrases with spaces, duplicate tags
    or content containing punctuation. Cleaning it up centrally keeps the API
    response structure stable and avoids empty tags, duplicate tags or hashtags
    in unusual formats when publishing to platforms.
    """
    if isinstance(raw, str):
        # CJK text separates tags with ，、； and no spaces; # starts a new tag.
        candidates = re.split(r"[\s,，、;；#]+", raw)
    elif isinstance(raw, (list, tuple)):
        # Each entry in the array is treated as a complete tag, so "du lich"
        # becomes "#dulich" instead of being split into two tags.
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        # Hindi, Thai and others rely on combining marks and cannot simply be
        # stripped like \w. But a combining mark must attach to a retained
        # character; when an emoji is removed its variation selector must go
        # too, otherwise #☕️ would leave a tag of invisible characters and
        # #coffee☕️ could not be deduplicated against #coffee. Digits are kept,
        # so also clean emoji presentation selectors and keycap combiners to
        # make #1️⃣ equal #1; other language combining marks and CJK glyph
        # variants still follow the attachment rule.
        characters = []
        can_attach_mark = False
        for character in item:
            if character in {"\ufe0e", "\ufe0f", "\u20e3"}:
                can_attach_mark = False
            elif character.isalnum() or character == "_":
                characters.append(character)
                can_attach_mark = True
            elif can_attach_mark and unicodedata.category(character).startswith("M"):
                characters.append(character)
            else:
                can_attach_mark = False
        tag = "".join(characters)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # Some models wrap the JSON in explanatory text or a markdown fence.
        # API callers only need a stable structure, so try to extract the first
        # JSON object.
        data = _extract_embedded_json(response, "{")

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # Chinese and Japanese sentences usually have no space at the end;
        # English still requires whitespace so numbers like 2.0 are not split.
        title = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", script, maxsplit=1)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    Generate short-video publishing copy metadata.

    The return structure is fixed as
    `{"title": str, "caption": str, "hashtags": List[str]}`. If the LLM is
    unavailable or returns a malformed format, fall back to generic heuristics
    so API callers always get a structure that can be displayed and edited
    before publishing.
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and response.startswith("Error: "):
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "What is the meaning of life"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
