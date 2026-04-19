"""TOML configuration loader for the comparison app.

The config file is opt-in via ``--config FILE``; CLI args still work on
their own. When both are provided, CLI args take precedence — callers
merge by passing the config's values as argparse defaults.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib

from app.constants import MAX_DISPLAY_POINTS, MAX_VARIABLE_CACHE_SIZE


@dataclasses.dataclass
class PathsConfig:
    """Input file paths for the comparison session."""

    source: str | None = None
    target: str | None = None


@dataclasses.dataclass
class FilterConfig:
    """Initial filter settings applied when the app starts."""

    expression: str | None = None
    flags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ServerConfig:
    """Runtime options for the Dash web server."""

    port: int = 5006
    debug: bool = False


@dataclasses.dataclass
class CacheConfig:
    """Limits controlling in-memory caching and display sizing."""

    max_variable_entries: int = MAX_VARIABLE_CACHE_SIZE
    max_display_points: int = MAX_DISPLAY_POINTS


@dataclasses.dataclass
class LoggingConfig:
    """Logging configuration for the application."""

    level: str = "INFO"


@dataclasses.dataclass
class AppConfig:
    """Top-level application configuration loaded from TOML.

    Attributes:
        paths: Source and target NetCDF file locations.
        filter: Initial expression and flag filters.
        server: Dash server settings.
        cache: Cache and display limits.
        logging: Logging verbosity settings.

    """

    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)
    filter: FilterConfig = dataclasses.field(default_factory=FilterConfig)
    server: ServerConfig = dataclasses.field(default_factory=ServerConfig)
    cache: CacheConfig = dataclasses.field(default_factory=CacheConfig)
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)


def load_config(path: str | pathlib.Path) -> AppConfig:
    """Parse a TOML file into an :class:`AppConfig`.

    Unknown keys are ignored so a config written for a newer version
    remains loadable. Missing sections fall through to dataclass
    defaults.
    """
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    def _section(name: str, cls: type) -> object:
        """Build one config section while ignoring unknown keys."""
        data = raw.get(name, {}) or {}
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})

    return AppConfig(
        paths=_section("paths", PathsConfig),  # type: ignore[arg-type]
        filter=_section("filter", FilterConfig),  # type: ignore[arg-type]
        server=_section("server", ServerConfig),  # type: ignore[arg-type]
        cache=_section("cache", CacheConfig),  # type: ignore[arg-type]
        logging=_section("logging", LoggingConfig),  # type: ignore[arg-type]
    )
