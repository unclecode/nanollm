# NanoLLM

Minimal, zero-bloat LLM API wrapper. Drop-in replacement for [litellm](https://github.com/BerriAI/litellm).

**~1500 lines** of clean code replacing ~50,000+ lines. Single dependency: `httpx`.

## Install

```bash
pip install nanollm
```

## Quick Start

```python
from nanollm import completion, acompletion

# Synchronous
response = completion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.choices[0].message.content)

# Async
response = await acompletion(
    model="anthropic/claude-3-5-sonnet-20240620",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Streaming
for chunk in completion(model="openai/gpt-4o", messages=[...], stream=True):
    print(chunk["choices"][0]["delta"].get("content", ""), end="")

# Batch (parallel)
from nanollm import batch_completion
responses = batch_completion(
    model="openai/gpt-4o",
    messages=[[{"role": "user", "content": f"Count to {i}"}] for i in range(5)],
)

# Embeddings
from nanollm import aembedding
response = await aembedding(
    model="openai/text-embedding-3-small",
    input=["Hello world", "Goodbye world"],
)
embeddings = [item["embedding"] for item in response.data]
```

## litellm Compatibility

NanoLLM includes a `litellm` compatibility shim. Existing code works unchanged:

```python
from litellm import completion  # ← works, routes to nanollm
from litellm.exceptions import RateLimitError  # ← works

import litellm
litellm.drop_params = True  # ← works, sets nanollm.drop_params
```

## Supported Providers

### OpenAI-Compatible (one code path, ~15 providers)

| Provider | Model Format | Env Var |
|----------|-------------|---------|
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Groq | `groq/llama3-70b-8192` | `GROQ_API_KEY` |
| Together | `together/meta-llama/Llama-3-70b` | `TOGETHER_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| Perplexity | `perplexity/sonar-pro` | `PERPLEXITYAI_API_KEY` |
| Fireworks | `fireworks/llama-v3-70b` | `FIREWORKS_API_KEY` |
| OpenRouter | `openrouter/meta-llama/llama-3-70b` | `OPENROUTER_API_KEY` |
| DeepInfra | `deepinfra/meta-llama/Llama-3-70b` | `DEEPINFRA_API_KEY` |
| xAI | `xai/grok-2` | `XAI_API_KEY` |
| Cerebras | `cerebras/llama3.1-70b` | `CEREBRAS_API_KEY` |

### Custom Protocol

| Provider | Model Format | Env Var |
|----------|-------------|---------|
| Anthropic | `anthropic/claude-3-5-sonnet-20240620` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |

### Local

| Provider | Model Format | Notes |
|----------|-------------|-------|
| Ollama | `ollama/llama3` | No API key needed |
| LM Studio | `lm_studio/model-name` | No API key needed |

### Cloud

| Provider | Model Format | Auth |
|----------|-------------|------|
| Azure OpenAI | `azure/deployment-name` | `AZURE_API_KEY` + `base_url` |
| AWS Bedrock | `bedrock/anthropic.claude-3-sonnet` | AWS credentials |
| Google Vertex AI | `vertex_ai/gemini-pro` | Google Cloud credentials |

## Adding New Providers

Most LLM providers are OpenAI-compatible. To add one, just add an entry to `PROVIDER_REGISTRY` in `nanollm/_config.py`:

```python
"new_provider": ProviderConfig(
    base_url="https://api.newprovider.com/v1",
    api_key_env="NEW_PROVIDER_API_KEY",
    adapter="openai_compat",
    supported_params=frozenset(OPENAI_PARAMS),
),
```

For providers with custom APIs, create a new adapter in `nanollm/adapters/`.

## Architecture

```
nanollm/
  __init__.py          # Public API (completion, acompletion, batch_completion, aembedding)
  _router.py           # "provider/model" → adapter resolution
  _types.py            # Response dataclasses (ModelResponse, EmbeddingResponse)
  _http.py             # httpx sync/async POST + SSE streaming
  _config.py           # Provider registry (base_url, auth, supported params)
  exceptions.py        # Exception hierarchy (RateLimitError, etc.)
  adapters/
    _base.py           # BaseAdapter ABC
    openai_compat.py   # OpenAI-compatible (covers ~15 providers)
    anthropic.py       # Anthropic Messages API
    gemini.py          # Google Gemini
    ollama.py          # Ollama (local, no auth)
    azure_openai.py    # Azure OpenAI
    bedrock.py         # AWS Bedrock (inline SigV4)
    vertex.py          # Google Vertex AI
litellm/               # Compatibility shim
  __init__.py
  exceptions.py
```

## License

MIT
