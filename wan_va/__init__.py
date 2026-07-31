# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Top-level package for Wan/LingBot VA code.

Keep imports lazy here.  Data-preparation utilities need to import
`wan_va.dataset.*` before any prepared MOT dataset exists, so importing this
package must not eagerly load training configs.
"""

from importlib import import_module

__all__ = ["configs", "distributed", "modules"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
