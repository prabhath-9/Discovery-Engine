from __future__ import annotations

import importlib
import pkgutil

import src


def _discover_packages() -> list[str]:
    names = [src.__name__]
    for module_info in pkgutil.walk_packages(src.__path__, prefix=f"{src.__name__}."):
        if module_info.ispkg:
            names.append(module_info.name)
    return names


def test_every_src_package_imports() -> None:
    packages = _discover_packages()
    assert len(packages) >= 7  # src + gateway, session, ranking, guardrails, compliance, explain, shared
    for name in packages:
        importlib.import_module(name)
