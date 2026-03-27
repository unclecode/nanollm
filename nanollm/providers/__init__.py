"""Provider registry -- class-based provider system.

Providers self-register via the ``@register("name")`` decorator.  The
registry maps provider names (e.g., ``"openai"``, ``"anthropic"``) to
provider class instances that are created on first lookup.

Usage::

    from nanollm.providers import get_provider, list_providers

    provider = get_provider("openai")
    print(list_providers())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseProvider

_REGISTRY: dict[str, type[BaseProvider]] = {}


def register(name: str):
    """Decorator to register a provider class under a given name.

    Can be stacked to register the same class under multiple names::

        @register("openai")
        @register("groq")
        class OpenAICompatProvider(BaseProvider): ...
    """

    def decorator(cls: type[BaseProvider]) -> type[BaseProvider]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_provider(name: str) -> BaseProvider:
    """Look up a provider by name and return a new instance.

    Triggers lazy import of all provider modules on first call so that
    ``@register`` decorators run before lookup.
    """
    if not _REGISTRY:
        _import_all_providers()

    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider: {name!r}. Available: {available}"
        )
    return cls()


def list_providers() -> list[str]:
    """Return a sorted list of all registered provider names."""
    if not _REGISTRY:
        _import_all_providers()
    return sorted(_REGISTRY.keys())


def _import_all_providers() -> None:
    """Lazy-import all provider modules so they register themselves.

    Each provider module uses ``@register("name")`` at the class level,
    so merely importing it is enough to populate ``_REGISTRY``.

    This function is a no-op after the first call since ``_REGISTRY``
    will already be populated.
    """
    # Import is intentionally inside the function for lazy loading.
    # Provider modules that don't exist yet are fine -- they'll be added
    # as the library grows.  Wrap each in a try/except so missing optional
    # providers don't break the core.
    _provider_modules = (
        "openai",
        "anthropic",
        "google",
        "aws",
        "azure",
        "local",
        "huggingface",
    )
    import importlib

    for mod_name in _provider_modules:
        try:
            importlib.import_module(f".{mod_name}", package=__name__)
        except ImportError:
            pass  # optional provider not installed
