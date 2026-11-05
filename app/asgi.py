"""Application implementation - ASGI."""

import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers import base
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """Handle API process startup recovery and shutdown logging in one place."""
    logger.info("startup event")

    configured_api_key = config.app.get("api_key", "")
    if configured_api_key in (None, ""):
        logger.warning(
            "API key authentication is disabled; keep the API on a trusted network"
        )
    elif isinstance(configured_api_key, str):
        # Log only the protected scope; never emit the key, its length or a
        # digest, so credentials cannot reach the logging system.
        logger.info("API key authentication is enabled for /api/v1 and /tasks")
    else:
        logger.error(
            "API key authentication is misconfigured: app.api_key must be a string"
        )

    # Cross-platform publishing runs in this process's thread pool and does not
    # resume after a service restart. At startup, active states in Redis that
    # have certainly lost their executing process are settled as failed so
    # tasks cannot stay undeletable forever.
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()

    # Redis queue entries persist across API restarts. No task_done callback
    # exists in the new process to dispatch them, so fill its worker slots now.
    from app.controllers.manager.redis_manager import RedisTaskManager
    from app.controllers.v1 import video as video_controller

    redis_task_manager = None
    if isinstance(video_controller.task_manager, RedisTaskManager):
        redis_task_manager = video_controller.task_manager
        redis_task_manager.resume_queued_tasks()
        redis_task_manager.start_queue_dispatcher()
    try:
        yield
    finally:
        if redis_task_manager is not None:
            redis_task_manager.stop_queue_dispatcher()
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    # Rejected inputs can contain NaN/Infinity, and custom validators attach
    # exception objects to ctx. Neither can be emitted by JSONResponse directly.
    errors = jsonable_encoder(
        e.errors(),
        custom_encoder={
            float: lambda value: value if math.isfinite(value) else str(value),
            Exception: str,
        },
    )
    # Describe the first rejected field instead of always claiming a missing
    # field: out-of-range values and malformed JSON are not "field required".
    message = "invalid request parameters"
    if errors and isinstance(errors[0], dict):
        location = ".".join(str(part) for part in errors[0].get("loc") or ())
        detail = errors[0].get("msg") or message
        message = f"{location}: {detail}" if location else str(detail)
    return JSONResponse(
        status_code=400,
        content=utils.get_response(status=400, data=errors, message=message),
    )


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalize_allowed_origin(raw_origin: str) -> str | None:
    """Fold a configured value into the Origin form browsers actually send.

    RFC 6454 pins the Origin header to ``scheme://host[:port]``: no path, no
    trailing slash, scheme and host already lowercased, and the scheme's
    default port omitted. A value copied from the browser address bar such as
    ``https://frontend.example/``, or one that spells out the default port as
    ``https://frontend.example:443``, is not character-for-character equal to
    the ``https://frontend.example`` browsers send, so the allowlist silently
    fails: the frontend only sees a CORS error and the server only logs one
    blocked line, and neither points at the configuration itself.

    ``None`` means the spelling can never match any Origin (missing scheme,
    scheme other than http/https, empty or whitespace-containing host,
    malformed IPv6 literal, invalid or out-of-range port), so the caller should
    drop it with a warning. Parse failures must collapse to ``None``: this
    function runs at module import time, so letting ``urlsplit()``'s
    ``ValueError`` escape means the whole API cannot start.
    """

    if raw_origin == "*":
        return raw_origin

    try:
        parsed = urlsplit(raw_origin)
        scheme = parsed.scheme.lower()
        # netloc is validated only when hostname / port are read: a malformed
        # IPv6 literal raises from urlsplit(), while a non-numeric or
        # out-of-range 0-65535 port raises when port is read.
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    # Only http/https carry browser Origin semantics, and both have an
    # omittable default port.
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    if any(character.isspace() for character in host):
        return None

    # Lowercase explicitly rather than relying on how urlsplit handles case in
    # different Python versions.
    host = host.lower()
    # Browsers keep square brackets around IPv6 literals when serializing an
    # Origin; default-port spellings such as ``:443``, ``:0443`` and ``:80``
    # must be folded to the portless form to match a real frontend.
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def parse_cors_allowed_origins(raw_origins: str | None) -> list[str]:
    """Parse the browser cross-origin allowlist.

    CORS only constrains cross-origin JavaScript in browsers; it does not
    affect curl, Postman, n8n or server-side SDKs. An empty list is returned
    when nothing is configured, meaning cross-origin access is off by default;
    a user who really deploys a standalone web frontend enables it explicitly
    through ``CORS_ALLOWED_ORIGINS``.

    Every source is first folded into the canonical Origin header form, so
    ``https://a.example/``, ``HTTPS://A.Example`` and the explicit default-port
    ``https://a.example:443`` all produce the same result, while spellings that
    cannot match any Origin are dropped with a warning. A malformed entry only
    drops itself and never aborts parsing here -- this function is called at
    module import time and a parse exception would prevent the API from
    starting.
    """

    if not raw_origins:
        return []

    # Strip whitespace around comma-separated items and ignore empty ones, so
    # the common environment-variable format
    # ``https://a.example, https://b.example,`` does not create origins that
    # can never match.
    origins: list[str] = []
    for candidate in raw_origins.split(","):
        item = candidate.strip()
        if not item:
            continue
        origin = _normalize_allowed_origin(item)
        if origin is None:
            logger.warning(
                f"ignoring configured CORS origin that cannot match a browser "
                f"Origin header: {item!r}"
            )
            continue
        if origin not in origins:
            origins.append(origin)
    return origins


def configure_cors(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Configure CORS from an explicit allowlist; an empty list keeps same-origin."""

    if not allowed_origins:
        logger.info(
            "browser cross-origin API access is disabled; set "
            "CORS_ALLOWED_ORIGINS to enable trusted origins"
        )
        return

    allow_all_origins = "*" in allowed_origins
    configured_api_key = config.app.get("api_key", "")
    if allow_all_origins and configured_api_key in (None, ""):
        # ``*`` is a compatibility mode the user chose explicitly, so startup is
        # not forcibly refused; but with authentication off it lets any web
        # page read and call the API, so an actionable security warning is
        # required.
        logger.warning(
            "CORS allows every browser origin while API key authentication is "
            "disabled; configure app.api_key or restrict CORS_ALLOWED_ORIGINS"
        )

    instance.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Starlette reflects any Origin when ``*`` and credentials are enabled
        # together. Wildcard mode does not need cookie authentication, so
        # credentials are turned off deliberately; explicit origins keep the
        # old behavior so existing standalone web frontends that rely on
        # credentialed requests are unaffected.
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        # When a remote HTTPS frontend calls an API on the local machine or
        # LAN, modern browsers send an extra Private Network Access preflight.
        # Only exact allowlist origins are granted permission; wildcard mode
        # keeps denying it, so no arbitrary site can probe the user's private
        # network services.
        allow_private_network=not allow_all_origins,
    )


def is_browser_origin_allowed(
    request: Request, allowed_origins: list[str]
) -> bool:
    """Return whether a browser origin is same-origin or explicitly allowlisted."""

    origin = request.headers.get("origin")
    if not origin:
        # curl, Postman, n8n and server-side SDKs usually send no Origin. Such
        # requests are kept so a security fix does not accidentally change the
        # contract existing API clients rely on.
        return True
    if "*" in allowed_origins or origin in allowed_origins:
        return True

    # Browsers may send Origin even for same-origin POSTs. Only scheme +
    # authority are compared, ignoring path and query parameters; a
    # reverse-proxy deployment that does not forward the public scheme/host
    # correctly can declare the external source explicitly through
    # CORS_ALLOWED_ORIGINS instead of trusting forwarded headers.
    request_url = urlsplit(str(request.url))
    request_origin = f"{request_url.scheme}://{request_url.netloc}"
    return origin == request_origin


def configure_browser_access(instance: FastAPI, allowed_origins: list[str]) -> None:
    """Set up both server-side Origin protection and browser CORS response policy."""

    @instance.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        """Reject untrusted browser origins, covering simple requests without preflight."""

        if not is_browser_origin_allowed(request, allowed_origins):
            origin = request.headers.get("origin", "")
            logger.warning(
                f"blocked untrusted browser origin: method={request.method}, "
                f"path={request.url.path}, origin={origin}"
            )
            return JSONResponse(
                status_code=403,
                content=utils.get_response(
                    status=403,
                    message="cross-origin browser request is not allowed",
                ),
            )

        return await call_next(request)

    # Registered last, the CORS middleware sits outside the Origin guard:
    # trusted preflights succeed directly, untrusted ones are rejected by CORS,
    # and actual requests that need no preflight still hit the 403 guard above.
    configure_cors(instance, allowed_origins)


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()


@app.middleware("http")
async def protect_generated_task_files(request: Request, call_next):
    """Protect the task-artifact static route so files cannot bypass API auth.

    ``/tasks`` is mounted separately by StaticFiles and cannot reuse APIRouter
    dependencies, so the same verify_token is called from a middleware. The
    auth function lets requests through when api_key is not configured;
    OPTIONS preflight requests are also left to the CORS middleware.
    """

    request_path = request.url.path
    is_task_file = request_path == "/tasks" or request_path.startswith("/tasks/")
    if is_task_file and request.method != "OPTIONS":
        try:
            base.verify_token(request)
        except HttpException as exception:
            return exception_handler(request, exception)

    return await call_next(request)


# Browser same-origin policy by default; cross-origin opens only when the user
# explicitly configures trusted web origins.
cors_allowed_origins = parse_cors_allowed_origins(
    os.getenv("CORS_ALLOWED_ORIGINS", "")
)
configure_browser_access(app, cors_allowed_origins)

task_dir = utils.task_dir()
app.mount("/tasks", StaticFiles(directory=task_dir, html=True), name="")

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")
