# SPDX-License-Identifier: Apache-2.0
"""LMCache integration for SGLang.

Two modes are supported:

**In-process mode**
    The LMCache engine runs inside the SGLang scheduler process.
    Use :class:`~lmcache.integration.sglang.sglang_adapter.LMCacheConnector`
    or :class:`~lmcache.integration.sglang.sglang_adapter.\
LMCacheLayerwiseConnector`.

**Multi-process mode**
    The LMCache engine runs as a standalone ZMQ server. Start the
    server first::

        python -m lmcache.v1.multiprocess.server \\
            --host 0.0.0.0 --port 5555

    then construct
    :class:`~lmcache.integration.sglang.sglang_mp_adapter.LMCacheMPConnector`
    or
    :class:`~lmcache.integration.sglang.sglang_mp_adapter.\
LMCacheMPLayerwiseConnector` with ``server_url="tcp://localhost:5555"``
    (or set ``LMCACHE_SERVER_URL``).
"""

# In-process adapter is always available (no third-party requirements
# beyond LMCache itself). The MP adapter requires ``sglang`` and ``zmq``
# at import time, so guard it behind a try/except so importing the
# package doesn't crash in environments that only need the in-process
# entry points.
try:
    # First Party
    from lmcache.integration.sglang.sglang_adapter import (
        LMCacheConnector,
        LMCacheLayerwiseConnector,
        LoadMetadata,
        StoreMetadata,
    )

    __all__ = [
        "LMCacheConnector",
        "LMCacheLayerwiseConnector",
        "LoadMetadata",
        "StoreMetadata",
    ]
except ImportError:
    __all__ = []

try:
    # First Party
    from lmcache.integration.sglang.sglang_mp_adapter import (
        LMCacheMPConnector,
        LMCacheMPLayerwiseConnector,
    )

    __all__ += [
        "LMCacheMPConnector",
        "LMCacheMPLayerwiseConnector",
    ]
except ImportError:
    pass
