"""Portable YAML configuration and reproducible run directories."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import yaml


def mergeConfig(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = mergeConfig(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def loadConfig(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping")
    parent = config.pop("extends", None)
    if parent:
        config = mergeConfig(loadConfig(path.parent / parent), config)
    return config


def configDigest(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def saveConfig(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
