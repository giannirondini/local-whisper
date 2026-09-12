"""Configuration surface for LocalWhisper.

Precedence, lowest to highest:

1. `Settings` defaults
2. `~/.config/localwhisper/config.toml` (or `$XDG_CONFIG_HOME/localwhisper/config.toml`)
3. environment variables `LOCALWHISPER_<FIELD>` (upper-cased field name)
4. command-line flags

The config file uses the field names as keys:

    whisper_model = "small"
    language = "it"
    ollama_model = "gemma4:e2b-mlx"
    hotkey = "<cmd>+<shift>+g"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from . import __version__

ENV_PREFIX = "LOCALWHISPER_"
CONFIG_ENV_VAR = f"{ENV_PREFIX}CONFIG"


class ConfigError(Exception):
    """Raised for an unreadable config file or an invalid setting value."""


@dataclass(frozen=True)
class Settings:
    """Everything a user can tune. Immutable; validated on construction."""

    whisper_model: str = "base.en"
    """faster-whisper model name. `*.en` models are English-only."""

    language: str | None = None
    """ISO 639-1 code passed to Whisper, or None for auto-detection."""

    ollama_model: str = "gemma4:e2b-mlx"
    """Chosen in docs/OLLAMA_MODEL_DECISION.md; `qwen3.5:4b-mlx` is the small alternative."""

    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 120.0
    """Read timeout in seconds for one refinement request."""

    keep_alive: str = "5m"
    """How long Ollama keeps the model loaded after a request."""

    hotkey: str = "<cmd>+<shift>+g"
    """pynput GlobalHotKeys syntax."""

    auto_paste: bool = False
    """Menu bar app only: press Cmd+V in the frontmost app after copying a result."""

    verbose: bool = False
    """Log engine internals to stderr."""

    def __post_init__(self) -> None:
        if not self.whisper_model:
            raise ConfigError("whisper_model must not be empty")
        if self.whisper_model.endswith(".en") and self.language not in (None, "en"):
            raise ConfigError(
                f"whisper_model {self.whisper_model!r} is English-only but language is "
                f"{self.language!r}; use a multilingual model such as 'base' or 'small'"
            )
        if not self.ollama_model:
            raise ConfigError("ollama_model must not be empty")
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"ollama_url must start with http:// or https://: {self.ollama_url!r}"
            )
        if self.ollama_url.endswith("/"):
            object.__setattr__(self, "ollama_url", self.ollama_url.rstrip("/"))
        if self.ollama_timeout <= 0:
            raise ConfigError(f"ollama_timeout must be positive: {self.ollama_timeout!r}")
        if not self.hotkey:
            raise ConfigError("hotkey must not be empty")


FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Settings))


def default_config_path(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if CONFIG_ENV_VAR in env:
        return Path(env[CONFIG_ENV_VAR]).expanduser()
    base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "localwhisper" / "config.toml"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off", ""):
        return False
    raise ConfigError(f"expected a boolean, got {value!r}")


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"expected a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"expected a number, got {value!r}") from e


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_str(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"expected a string, got {value!r}")
    return value.strip()


_COERCE: dict[str, Callable[[Any], Any]] = {
    "language": _to_optional_str,
    "ollama_timeout": _to_float,
    "auto_paste": _to_bool,
    "verbose": _to_bool,
}


def coerce_setting(name: str, value: Any) -> Any:
    """Convert a raw file/env/UI value to the field's type. Raises `ConfigError`."""
    if name not in FIELD_NAMES:
        raise ConfigError(f"Unknown setting: {name}")
    return _COERCE.get(name, _to_str)(value)


def _from_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"Cannot read config file {path}: {e}") from e
    unknown = sorted(set(raw) - set(FIELD_NAMES))
    if unknown:
        raise ConfigError(f"Unknown setting(s) in {path}: {', '.join(unknown)}")
    return {name: coerce_setting(name, value) for name, value in raw.items()}


def _from_env(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in FIELD_NAMES:
        key = f"{ENV_PREFIX}{name.upper()}"
        if key in env:
            out[name] = coerce_setting(name, env[key])
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localwhisper",
        description="Privacy-first local voice-to-text with Whisper and Ollama.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", metavar="PATH", help="config file (TOML)")
    parser.add_argument("--whisper-model", dest="whisper_model", metavar="NAME")
    parser.add_argument(
        "--language", metavar="CODE", help="Whisper language code; omit for auto-detect"
    )
    parser.add_argument("--ollama-model", dest="ollama_model", metavar="NAME")
    parser.add_argument("--ollama-url", dest="ollama_url", metavar="URL")
    parser.add_argument(
        "--ollama-timeout", dest="ollama_timeout", metavar="SECONDS", help="read timeout"
    )
    parser.add_argument("--hotkey", metavar="COMBO", help="pynput syntax, e.g. '<cmd>+<shift>+g'")
    parser.add_argument(
        "--auto-paste",
        dest="auto_paste",
        action="store_true",
        default=None,
        help="menu bar app: press Cmd+V after copying a result",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=None)
    return parser


def _from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: coerce_setting(name, getattr(args, name))
        for name in FIELD_NAMES
        if getattr(args, name, None) is not None
    }


def resolve_config_path(
    args: argparse.Namespace | None = None, env: Mapping[str, str] | None = None
) -> Path:
    """The config file in effect: `--config` if given, else `default_config_path()`."""
    if args is not None and args.config:
        return Path(args.config).expanduser()
    return default_config_path(env)


def load_settings(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> Settings:
    """Build `Settings` from defaults, config file, environment and CLI flags.

    Raises:
        ConfigError: On an unreadable file, unknown key, bad value or invalid combination.
    """
    env = os.environ if env is None else env
    args = build_parser().parse_args(argv)
    path = resolve_config_path(args, env) if config_path is None else config_path

    merged: dict[str, Any] = {}
    merged.update(_from_file(path))
    merged.update(_from_env(env))
    merged.update(_from_args(args))
    return replace(Settings(), **merged)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    # JSON string escapes are valid TOML basic-string escapes.
    return json.dumps(str(value), ensure_ascii=False)


def save_setting(name: str, value: Any, path: Path | None = None) -> Path:
    """Write one setting to the config file in place, keeping every other line.

    `None` (auto-detect language) removes the key so the default applies. The file
    and its directory are created if missing. Returns the path written.

    Raises:
        ConfigError: Unknown setting name, or the file cannot be written.
    """
    if name not in FIELD_NAMES:
        raise ConfigError(f"Unknown setting: {name}")
    path = default_config_path() if path is None else path
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    is_key = re.compile(rf"^\s*{re.escape(name)}\s*=").match
    new_line = None if value is None else f"{name} = {_toml_value(value)}"

    out: list[str] = []
    replaced = False
    for line in lines:
        if is_key(line):
            if new_line is not None and not replaced:
                out.append(new_line)
            replaced = True  # drop duplicates and, for None, the key itself
        else:
            out.append(line)
    if new_line is not None and not replaced:
        out.append(new_line)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Cannot write config file {path}: {e}") from e
    return path
