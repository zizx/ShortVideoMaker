from dataclasses import dataclass


DEFAULT_LLM_PROVIDER_ID = "moonshot"


@dataclass(frozen=True, slots=True)
class LLMProviderField:
    """Describe extra Provider fields beyond API Key, Base URL, and model name."""

    config_suffix: str
    label_key: str
    required: bool = False
    secret: bool = False
    default_value: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderEndpoint:
    """Describe the companion entry points and API addresses for a Provider region."""

    endpoint_id: str
    default_label: str
    base_url: str
    api_key_url: str
    model_docs_url: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderSpec:
    """
    Central declaration of an LLM Provider.

    Holds stable metadata shared by the WebUI, config loading, and service calls,
    including the default display name and locale key. It does not store translated
    copy or implement API requests. The Registry defines what a Provider is, while
    the service-layer adapter remains responsible for how it is called.
    """

    provider_id: str
    default_label: str
    adapter: str = "openai_compatible"
    api_key_url: str = ""
    default_model: str = ""
    default_base_url: str = ""
    model_docs_url: str = ""
    requires_api_key: bool = True
    requires_model_name: bool = True
    requires_base_url: bool = True
    show_api_key: bool = True
    show_base_url: bool = True
    deprecated_models: tuple[str, ...] = ()
    deprecated_base_urls: tuple[str, ...] = ()
    extra_fields: tuple[LLMProviderField, ...] = ()
    service_endpoints: tuple[LLMProviderEndpoint, ...] = ()
    default_service_endpoint_id: str = ""
    international_service_endpoint_id: str = ""

    @property
    def label_key(self) -> str:
        return f"llm_provider_label.{self.provider_id}"

    @property
    def tips_key(self) -> str:
        return f"llm_provider_tips.{self.provider_id}"

    @property
    def endpoint_selector_label_key(self) -> str:
        return f"llm_provider_endpoint_selector.{self.provider_id}"

    @property
    def endpoint_selector_help_key(self) -> str:
        return f"llm_provider_endpoint_selector_help.{self.provider_id}"

    @property
    def authentication_error_key(self) -> str:
        return f"llm_provider_authentication_error.{self.provider_id}"

    def endpoint_label_key(self, endpoint_id: str) -> str:
        return f"llm_provider_endpoint.{self.provider_id}.{endpoint_id}"

    def config_key(self, suffix: str) -> str:
        return f"{self.provider_id}_{suffix}"

    def resolve_model_name(self, configured_model: str | None) -> str:
        """Resolve empty or deprecated historical defaults to the current default model."""
        model_name = (configured_model or "").strip()
        if not model_name or model_name in self.deprecated_models:
            return self.default_model
        return model_name

    def resolve_base_url(self, configured_base_url: str | None) -> str:
        """Resolve the Base URL and migrate retired historical addresses to the current default."""
        base_url = (configured_base_url or "").strip()
        deprecated_urls = {url.rstrip("/") for url in self.deprecated_base_urls}
        if not base_url or base_url.rstrip("/") in deprecated_urls:
            return self.effective_default_base_url
        return base_url

    def get_service_endpoint(self, endpoint_id: str) -> LLMProviderEndpoint | None:
        """Get a service region by stable ID so logic never relies on changing promo links."""
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.endpoint_id == endpoint_id
            ),
            None,
        )

    @property
    def default_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Return the default service region declared by the Provider."""
        return self.get_service_endpoint(self.default_service_endpoint_id)

    @property
    def international_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Return the international service region declared by the Provider."""
        return self.get_service_endpoint(self.international_service_endpoint_id)

    @property
    def effective_default_base_url(self) -> str:
        """Use the default service region's Base URL first, falling back to the original field."""
        endpoint = self.default_service_endpoint
        return endpoint.base_url if endpoint else self.default_base_url

    def preferred_service_endpoint(
        self, *, prefer_international: bool
    ) -> LLMProviderEndpoint | None:
        """Return the preferred entry for the UI region, safely falling back to the default."""
        if prefer_international and self.international_service_endpoint:
            return self.international_service_endpoint
        return self.default_service_endpoint

    def effective_api_key_url(self, *, prefer_international: bool = False) -> str:
        """Centralize the API Key signup URL so Endpoint Providers need no duplicate links."""
        endpoint = self.preferred_service_endpoint(
            prefer_international=prefer_international
        )
        return endpoint.api_key_url if endpoint else self.api_key_url

    def effective_model_docs_url(self, *, prefer_international: bool = False) -> str:
        """Centralize model-list and docs URLs so Endpoint Providers need no duplicate links."""
        endpoint = self.preferred_service_endpoint(
            prefer_international=prefer_international
        )
        return endpoint.model_docs_url if endpoint and endpoint.model_docs_url else self.model_docs_url

    def find_service_endpoint(
        self, configured_base_url: str | None
    ) -> LLMProviderEndpoint | None:
        """Identify the Provider's standard service region from a saved Base URL."""
        normalized_url = (configured_base_url or "").strip().rstrip("/")
        if not normalized_url:
            return None
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.base_url.rstrip("/") == normalized_url
            ),
            None,
        )

    def select_service_endpoint(
        self,
        configured_base_url: str | None,
        *,
        has_api_key: bool,
        prefer_international: bool,
    ) -> LLMProviderEndpoint | None:
        """
        Select the standard service region the WebUI should display.

        A saved standard URL takes priority; unknown URLs remain custom. Legacy
        configs may have only an API Key and no Base URL. Those users keep the
        Registry default region so an upgrade does not switch services because of
        a different UI language. Only new configs select the international entry
        point based on the UI language.
        """
        configured_url = (configured_base_url or "").strip()
        if configured_url:
            return self.find_service_endpoint(configured_url)

        default_endpoint = self.default_service_endpoint
        if has_api_key or not prefer_international:
            return default_endpoint

        return self.preferred_service_endpoint(
            prefer_international=prefer_international
        )


# Tuple order defines the WebUI dropdown order. A new standard OpenAI-compatible
# Provider usually only needs an entry here plus locale strings; only Providers
# with a different protocol need a matching adapter in app/services/llm.py.
LLM_PROVIDER_REGISTRY = (
    # Recommended Providers
    LLMProviderSpec(
        "moonshot",
        "Kimi / Moonshot AI",
        default_model="kimi-k3",
        service_endpoints=(
            LLMProviderEndpoint(
                endpoint_id="china",
                default_label="China",
                base_url="https://api.moonshot.cn/v1",
                api_key_url=(
                    "https://platform.kimi.com?"
                    "track_id=track-6eec1e56a4494e52adcaebbcbbefce59&"
                    "aff=moneyprinterturbo"
                ),
                model_docs_url=(
                    "https://platform.kimi.com/docs/models?"
                    "track_id=track-6eec1e56a4494e52adcaebbcbbefce59&"
                    "aff=moneyprinterturbo"
                ),
            ),
            LLMProviderEndpoint(
                endpoint_id="global",
                default_label="Global",
                base_url="https://api.moonshot.ai/v1",
                api_key_url=(
                    "https://platform.kimi.ai?"
                    "track_id=track-9e3b711aa2594e378f6fe5b8de718a76&"
                    "aff=moneyprinterturbo"
                ),
                model_docs_url=(
                    "https://platform.kimi.ai/docs/models?"
                    "track_id=track-9e3b711aa2594e378f6fe5b8de718a76&"
                    "aff=moneyprinterturbo"
                ),
            ),
        ),
        default_service_endpoint_id="china",
        international_service_endpoint_id="global",
    ),
    # Kimi subscriptions (Kimi Code benefits for Plus and higher tiers) use OAuth
    # device authorization: after the user confirms in the browser, the service
    # calls the official OpenAI-compatible endpoint directly with a Bearer token,
    # so no pay-as-you-go API Key is needed. China and Global use separate account
    # and authorization systems; endpoints pair each auth page, API address, and
    # subscription link.
    LLMProviderSpec(
        "kimi_code",
        "Kimi Code (Kimi subscription)",
        adapter="kimi_code",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        requires_model_name=False,
        default_model="kimi-for-coding",
        extra_fields=(
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
        service_endpoints=(
            LLMProviderEndpoint(
                endpoint_id="china",
                default_label="China",
                base_url="https://api.kimi.com/coding/v1",
                api_key_url="https://www.kimi.com/code?aff=moneyprinterturbo",
                model_docs_url="https://platform.kimi.com/docs/models?aff=moneyprinterturbo",
            ),
            LLMProviderEndpoint(
                endpoint_id="global",
                default_label="Global",
                base_url="https://api.kimi.ai/coding/v1",
                api_key_url="https://www.kimi.ai/code?aff=moneyprinterturbo",
                model_docs_url="https://platform.kimi.ai/docs/models?aff=moneyprinterturbo",
            ),
        ),
        default_service_endpoint_id="china",
        international_service_endpoint_id="global",
    ),
    # Mainstream model vendors and cloud providers
    LLMProviderSpec(
        "openai",
        "OpenAI",
        api_key_url="https://platform.openai.com/api-keys",
        default_model="gpt-5.5",
        default_base_url="https://api.openai.com/v1",
    ),
    LLMProviderSpec(
        "openai_codex",
        "OpenAI - Codex",
        adapter="openai_codex",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        extra_fields=(
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
    ),
    LLMProviderSpec(
        "openai_chatgpt",
        "OpenAI - ChatGPT (Native OAuth)",
        adapter="openai_chatgpt",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        extra_fields=(
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
    ),
    LLMProviderSpec(
        "anthropic",
        "Anthropic Claude",
        api_key_url="https://platform.claude.com/settings/keys",
        default_model="claude-sonnet-5",
        default_base_url="https://api.anthropic.com/v1/",
    ),
    LLMProviderSpec(
        "gemini",
        "Google Gemini",
        adapter="gemini",
        api_key_url="https://aistudio.google.com/app/apikey",
        default_model="gemini-3.1-pro-preview",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("gemini-pro", "gemini-1.0-pro"),
    ),
    LLMProviderSpec(
        "muse_spark",
        "Muse Spark",
        api_key_url="https://dev.meta.ai/docs/authentication",
        default_model="muse-spark-1.3",
        default_base_url="https://api.meta.ai/v1",
        model_docs_url="https://dev.meta.ai/docs/models",
    ),
    LLMProviderSpec(
        "deepseek",
        "DeepSeek",
        api_key_url="https://platform.deepseek.com/api_keys",
        default_model="deepseek-v4-pro",
        default_base_url="https://api.deepseek.com",
    ),
    LLMProviderSpec(
        "qwen",
        "Alibaba Cloud Qwen",
        adapter="qwen",
        api_key_url="https://dashscope.console.aliyun.com/apiKey",
        default_model="qwen-max",
        requires_base_url=False,
        show_base_url=False,
    ),
    LLMProviderSpec(
        "azure",
        "Microsoft Azure OpenAI",
        adapter="azure",
        api_key_url=(
            "https://portal.azure.com/#view/"
            "Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI"
        ),
        default_model="gpt-35-turbo",
    ),
    LLMProviderSpec(
        "volcengine",
        "ByteDance VolcEngine Ark",
        api_key_url=(
            "https://www.volcengine.com/activity/ai618?utm_campaign=hw&"
            "utm_content=hw&utm_medium=devrel_tool_web&utm_source=OWO&"
            "utm_term=MoneyPrinterTurbo"
        ),
        default_model="doubao-seed-2-1-turbo-260628",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    LLMProviderSpec(
        "grok",
        "xAI Grok",
        api_key_url="https://console.x.ai/",
        default_model="grok-4.3",
        default_base_url="https://api.x.ai/v1",
    ),
    LLMProviderSpec(
        "minimax",
        "MiniMax",
        api_key_url="https://platform.minimax.io/",
        default_model="MiniMax-M3",
        default_base_url="https://api.minimax.io/v1",
    ),
    LLMProviderSpec(
        "mimo",
        "Xiaomi MiMo",
        api_key_url=(
            "https://platform.xiaomimimo.com/docs/zh-CN/quick-start/first-api-call"
        ),
        default_model="mimo-v2.5-pro",
        default_base_url="https://api.xiaomimimo.com/v1",
    ),
    # iFlytek Spark MaaS uses different OpenAI-compatible addresses for pay-as-you-go
    # and Token Plan, and their API Keys are not interchangeable. This defaults to
    # pay-as-you-go; Token Plan users can enter the maas-token-api address in Base URL.
    LLMProviderSpec(
        "iflytek",
        "iFlytek Spark (Astron MaaS)",
        api_key_url="https://maas.xfyun.cn/",
        default_model="spark-x2.5",
        default_base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2",
        model_docs_url="https://maas.xfyun.cn/modelSquare",
    ),
    # Aggregation and unified access platforms
    LLMProviderSpec(
        "jalapeno",
        "Jalapeno Cloud",
        api_key_url="https://www.jalapeno-cloud.ai/MoneyPrinterTurbo",
        default_model="DeepSeek-V4.1-Flash",
        default_base_url="https://api.jalapeno-cloud.ai/v1",
        model_docs_url="https://www.jalapeno-cloud.ai/models",
    ),
    LLMProviderSpec(
        "infistar",
        "Infistar",
        api_key_url="https://infistar.cc/keys",
        default_model="gpt-6-luna",
        default_base_url="https://infistar.cc/v1",
        model_docs_url="https://infistar.cc/pricing",
    ),
    LLMProviderSpec(
        "shengsuanyun",
        "Shengsuan Cloud",
        api_key_url="https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
        default_model="deepseek/deepseek-v4-flash",
        default_base_url="https://router.shengsuanyun.com/api/v1",
    ),
    # APIMart offers both an `/api/v1` business API and a `/v1` OpenAI-compatible
    # API. The LLM service relies on the OpenAI SDK to read choices directly, so it
    # must use the `/v1` entry without the code/data wrapper, not the async API.
    LLMProviderSpec(
        "apimart",
        "APIMart",
        api_key_url="https://go.apimart.ai/gh-moneyprinterturbo",
        default_model="gpt-5.6-terra",
        default_base_url="https://api.apimart.ai/v1",
    ),
    LLMProviderSpec(
        "cloudflare",
        "Cloudflare AI Gateway",
        adapter="cloudflare_ai_gateway",
        api_key_url="https://dash.cloudflare.com/",
        default_model="openai/gpt-4.1-mini",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("@cf/meta/llama-3.1-8b-instruct",),
        extra_fields=(
            LLMProviderField("account_id", "Account ID", required=True),
            LLMProviderField(
                "gateway_id",
                "Gateway ID",
                default_value="default",
            ),
        ),
    ),
    LLMProviderSpec(
        "modelscope",
        "Alibaba ModelScope",
        adapter="modelscope",
        api_key_url=("https://modelscope.cn/docs/model-service/API-Inference/intro"),
        default_model="ZhipuAI/GLM-5.2",
        default_base_url="https://api-inference.modelscope.cn/v1/",
    ),
    LLMProviderSpec(
        "aihubmix",
        "AIHubMix",
        api_key_url="https://aihubmix.com/",
        default_model="gpt-5.4-mini",
        default_base_url="https://aihubmix.com/v1",
    ),
    LLMProviderSpec(
        "aimlapi",
        "AIML API",
        api_key_url="https://aimlapi.com/app/keys",
        default_model="openai/gpt-5-5",
        default_base_url="https://api.aimlapi.com/v1",
    ),
    LLMProviderSpec(
        "evolink",
        "EvoLink",
        api_key_url="https://evolink.ai/dashboard/keys",
        default_model="gpt-5.5",
        default_base_url="https://direct.evolink.ai/v1",
    ),
    LLMProviderSpec(
        "openrouter",
        "OpenRouter",
        api_key_url="https://openrouter.ai/settings/keys",
        default_model="minimax/minimax-m3:free",
        default_base_url="https://openrouter.ai/api/v1",
    ),
    LLMProviderSpec(
        "api_route",
        "API Route",
        api_key_url="https://www.api-route.com",
        default_model="gpt-5.4-mini",
        default_base_url="https://www.api-route.com/v1",
        model_docs_url="https://www.api-route.com/pricing",
    ),
    # Fluxion's OpenAI group uses the /v1 API and reuses the existing Chat
    # Completions adapter. The group determines available models and addresses, so
    # only defaults are provided; users can still override config. Anthropic/Gemini
    # native groups cannot use this entry point because their protocols do not match.
    LLMProviderSpec(
        "fluxionai",
        "Fluxion AI",
        api_key_url=(
            "https://fluxionai.space/register?source=github"
            "&campaign=moneyprinterturbo&promo=MONEYPRINTERTURBO"
        ),
        default_model="gpt-5.5",
        default_base_url="https://fluxionai.space/v1",
        model_docs_url="https://fluxionai.space/model-plaza",
    ),
    LLMProviderSpec(
        "cheaperinference",
        "Cheaper Inference",
        api_key_url="https://cheaperinference.com/signup",
        default_model="gpt-5.4-mini",
        default_base_url="https://api.cheaperinference.com/v1",
        model_docs_url="https://cheaperinference.com/#models",
    ),
    LLMProviderSpec(
        "requesty",
        "Requesty",
        api_key_url="https://app.requesty.ai/api-keys",
        default_model="openai/gpt-5.4-mini",
        default_base_url="https://router.requesty.ai/v1",
        model_docs_url="https://www.requesty.ai/models",
    ),
    LLMProviderSpec(
        "futureinfra",
        "FutureInfra",
        api_key_url="https://futureinfra.ai/console/?screen=ai-router",
        default_model="openai/gpt-4o-mini",
        default_base_url="https://futureinfra.ai/v1/ai",
        model_docs_url="https://futureinfra.ai/ai/",
    ),
    LLMProviderSpec(
        "yapi",
        "Y-API",
        api_key_url="https://y-api.bestvirtualgoods.com/app/keys",
        default_model="deepseek/deepseek-v4-flash",
        default_base_url="https://api.y-api.bestvirtualgoods.com/v1",
        model_docs_url="https://y-api.bestvirtualgoods.com/models",
    ),
    LLMProviderSpec(
        "opper",
        "Opper",
        api_key_url="https://platform.opper.ai",
        default_model="gpt-5.4-mini",
        default_base_url="https://api.opper.ai/v3/compat",
        model_docs_url="https://opper.ai/models",
    ),
    LLMProviderSpec(
        "onomeo",
        "onomeo",
        api_key_url="https://onomeo.com/dashboard",
        default_model="auto",
        default_base_url="https://onomeo.com/v1",
        model_docs_url="https://onomeo.com/models",
    ),
    LLMProviderSpec(
        "empiriolabs",
        "EmpirioLabs AI",
        api_key_url="https://platform.empiriolabs.ai/dashboard/api-keys",
        default_model="qwen3-8-flash",
        default_base_url="https://api.empiriolabs.ai/v1",
        model_docs_url="https://docs.empiriolabs.ai/models-pricing",
    ),
    LLMProviderSpec(
        "runcrate",
        "Runcrate",
        api_key_url="https://www.runcrate.ai/dashboard/api-keys?utm_source=MoneyPrinterTurbo&utm_medium=directory&utm_campaign=gpu_pricing",
        default_model="deepseek-ai/DeepSeek-V4.1-Flash",
        default_base_url="https://api.runcrate.ai/v1",
        model_docs_url="https://www.runcrate.ai/inference/pricing?utm_source=MoneyPrinterTurbo&utm_medium=directory&utm_campaign=gpu_pricing",
    ),
    LLMProviderSpec(
        "cloudgpu",
        "CloudGPU",
        api_key_url="https://cloudgpu.app",
        default_model="deepseek-v4-flash",
        default_base_url="https://cloudgpu.app/v1",
        model_docs_url="https://cloudgpu.app/v1/models",
    ),
    # Local deployments and general-purpose gateways
    LLMProviderSpec(
        "ollama",
        "Ollama",
        requires_api_key=False,
        show_api_key=False,
    ),
    # Claude subscriptions (Pro / Max / Team) do not issue API Keys; credentials
    # are available only to the official Claude Code client. This Provider therefore
    # does not use HTTP but invokes the locally signed-in claude CLI. An empty model
    # name keeps the CLI's current default model.
    LLMProviderSpec(
        "claude_code",
        "Claude Code (Claude subscription)",
        adapter="claude_code",
        requires_api_key=False,
        show_api_key=False,
        requires_base_url=False,
        show_base_url=False,
        requires_model_name=False,
        extra_fields=(
            LLMProviderField("cli_path", "Claude CLI Path"),
            LLMProviderField("timeout", "Timeout (seconds)", default_value="300"),
        ),
    ),
    LLMProviderSpec(
        "oneapi",
        "OneAPI",
        api_key_url="https://github.com/songquanpeng/one-api",
    ),
    LLMProviderSpec(
        "litellm",
        "LiteLLM",
        adapter="litellm",
        default_model="openai/gpt-4o-mini",
        requires_api_key=False,
        requires_base_url=False,
        show_api_key=False,
        show_base_url=False,
    ),
    # Other inference and public services
    LLMProviderSpec(
        "groq",
        "Groq",
        api_key_url="https://console.groq.com/keys",
        default_model="openai/gpt-oss-120b",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    LLMProviderSpec(
        "pollinations",
        "Pollinations AI",
        api_key_url="https://enter.pollinations.ai/",
        default_model="openai-fast",
        default_base_url="https://gen.pollinations.ai/v1",
        deprecated_models=("default",),
        deprecated_base_urls=("https://text.pollinations.ai/openai",),
    ),
)

LLM_PROVIDERS = {provider.provider_id: provider for provider in LLM_PROVIDER_REGISTRY}

if len(LLM_PROVIDERS) != len(LLM_PROVIDER_REGISTRY):
    raise RuntimeError("duplicate LLM provider id in registry")


def get_llm_provider(provider_id: str) -> LLMProviderSpec | None:
    return LLM_PROVIDERS.get((provider_id or "").lower())


def normalize_provider_override(value: str | None, default_value: str | None) -> str:
    """
    Keep only user overrides that differ from Registry defaults.

    The WebUI needs to show defaults in input fields without persisting them to
    config.toml; otherwise old configs would keep overriding new Registry default
    models or addresses after an upgrade.
    """
    normalized_value = (value or "").strip()
    normalized_default = (default_value or "").strip()
    if normalized_value == normalized_default:
        return ""
    return normalized_value
