from __future__ import annotations

from pathlib import Path

from .schema import AdapterManifest, SetupManifest, load_adapter, load_setup, make_trial


def load_catalog(root: str | Path) -> tuple[dict[str, SetupManifest], dict[str, AdapterManifest]]:
    root = Path(root)
    setups = {item.name: item for path in sorted((root / "setups").glob("*.yaml")) if (item := load_setup(path))}
    adapters = {
        item.name: item for path in sorted((root / "adapters").glob("*.yaml")) if (item := load_adapter(path))
    }
    return setups, adapters


def build_matrix(setup: SetupManifest, adapters: list[AdapterManifest]):
    return [make_trial(setup, adapter) for adapter in adapters]

