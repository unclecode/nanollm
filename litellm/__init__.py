"""litellm compatibility shim.

Re-exports nanollm's public API under the litellm namespace so that
existing code using `from litellm import completion` works unchanged.
"""

from nanollm import (
    __version__,
    acompletion,
    aembedding,
    batch_completion,
    completion,
)
from nanollm import drop_params, set_verbose
from nanollm.exceptions import (
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    InvalidRequestError,
    NanoLLMException,
    RateLimitError,
    ServiceUnavailableError,
)

# Module-level attributes that crawl4ai sets
import nanollm as _nanollm
import sys

# Make litellm.drop_params and litellm.set_verbose work as setattr
class _LitellmModule(sys.modules[__name__].__class__):
    @property
    def drop_params(self):
        return _nanollm.drop_params

    @drop_params.setter
    def drop_params(self, value):
        _nanollm.drop_params = value

    @property
    def set_verbose(self):
        return _nanollm.set_verbose

    @set_verbose.setter
    def set_verbose(self, value):
        _nanollm.set_verbose = value

sys.modules[__name__].__class__ = _LitellmModule
