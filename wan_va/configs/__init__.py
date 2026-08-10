# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Lazy config registry.

Some configs depend on prepared local datasets.  Importing the registry should
not require every dataset to exist; only the selected config is loaded.
"""

from importlib import import_module
from collections.abc import Mapping, Iterator


_CONFIG_MODULES = {
    "wan22_train": ("va_wan22_train_cfg", "va_wan22_train_cfg"),
}

_EXPORTED_CONFIGS = {attr: module for module, attr in _CONFIG_MODULES.values()}


class _LazyConfigRegistry(Mapping):
    def __init__(self):
        self._cache = {}

    def __getitem__(self, key):
        if key not in _CONFIG_MODULES:
            raise KeyError(key)
        if key not in self._cache:
            module_name, attr = _CONFIG_MODULES[key]
            module = import_module(f"{__name__}.{module_name}")
            self._cache[key] = getattr(module, attr)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(_CONFIG_MODULES)

    def __len__(self) -> int:
        return len(_CONFIG_MODULES)


VA_CONFIGS = _LazyConfigRegistry()
__all__ = ["VA_CONFIGS", *_EXPORTED_CONFIGS]


def __getattr__(name):
    if name in _EXPORTED_CONFIGS:
        module = import_module(f"{__name__}.{_EXPORTED_CONFIGS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
