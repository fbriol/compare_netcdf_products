"""Session state management for the Dash application.

The state is split in two:

- :class:`AppContext` — immutable, process-global (open NetCDF handles,
  paths, CLI-supplied flags). Shared by every browser tab.
- :class:`SessionData` — mutable, per-session (active filter, nav state,
  variable cache). Keyed on a UUID stored in ``flask.session`` so each
  tab / user gets its own state.

:class:`SessionState` is kept as a thin proxy so callback code that
reads ``state.active_mask`` or ``state.variable_cache[...]`` keeps
working. Reads/writes on mutable fields are routed to the current
request's :class:`SessionData`; reads on read-only fields hit the
shared :class:`AppContext`.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pathlib

    import netCDF4
    import numpy

    from .comparison import FilterInfo, GroupComparisonResult
    from .data_loading import DataSource
    from .flags import Flag

from .constants import MAX_VARIABLE_CACHE_SIZE

LOGGER = logging.getLogger(__name__)

# Fields that belong to SessionData (per-request). Anything else on
# SessionState resolves against the shared AppContext.
_SESSION_FIELDS = frozenset(
    {
        "active_mask",
        "active_expression",
        "filter_info",
        "group_results",
        "variable_cache",
        "lock",
        "current_view",
        "current_group_path",
        "current_varpath",
    }
)


@dataclasses.dataclass
class AppContext:
    """Process-global, read-only context shared by every session."""

    src_path: pathlib.Path
    tgt_path: pathlib.Path
    src_ds: netCDF4.Dataset
    tgt_ds: netCDF4.Dataset
    flags: list[Flag] | None = None
    max_cache_size: int = MAX_VARIABLE_CACHE_SIZE


@dataclasses.dataclass
class SessionData:
    """Per-session, mutable state (one per browser tab / session id)."""

    active_mask: numpy.ndarray | None = None
    active_expression: str | None = None
    filter_info: FilterInfo | None = None
    group_results: list[GroupComparisonResult] = dataclasses.field(default_factory=list)
    variable_cache: collections.OrderedDict[str, DataSource] = dataclasses.field(
        default_factory=collections.OrderedDict
    )
    lock: threading.RLock = dataclasses.field(
        default_factory=threading.RLock, repr=False, compare=False
    )
    current_view: str = "group"
    current_group_path: str = "/"
    current_varpath: str | None = None


# ── Session registry ────────────────────────────────────────────────

_sessions: dict[str, SessionData] = {}
_sessions_lock = threading.Lock()
_default_session: SessionData | None = None


def _seed_from_default() -> SessionData:
    """Build a fresh :class:`SessionData`, sharing expensive seed data.

    ``group_results`` is shared by reference: applying a filter
    reassigns the attribute on the session, so the shared list is
    never mutated. ``variable_cache`` starts empty so each session
    pays its own cache cost but doesn't starve others.
    """
    data = SessionData()
    seed = _default_session
    if seed is not None:
        data.group_results = seed.group_results
        data.active_mask = seed.active_mask
        data.active_expression = seed.active_expression
        data.filter_info = seed.filter_info
    return data


def _set_default_session(data: SessionData) -> None:
    """Remember the startup-initialised session so new tabs can reuse it."""
    global _default_session
    _default_session = data


def get_session() -> SessionData:
    """Return the current request's :class:`SessionData`.

    Falls back to the module-level default when called outside a Flask
    request context (startup, tests). New sessions are lazily created
    and seeded from the default session.
    """
    try:
        import flask

        sid = flask.session.get("sid")
        if not sid:
            sid = uuid.uuid4().hex
            flask.session["sid"] = sid
        with _sessions_lock:
            data = _sessions.get(sid)
            if data is None:
                data = _seed_from_default()
                _sessions[sid] = data
            return data
    except RuntimeError:
        # Outside a request context — use the default session.
        global _default_session
        if _default_session is None:
            _default_session = SessionData()
        return _default_session


# ── SessionState proxy ──────────────────────────────────────────────


class SessionState:
    """Proxy exposing :class:`AppContext` + current :class:`SessionData`.

    Reads of per-session fields are routed through :func:`get_session`,
    so every browser tab sees its own filter/cache/navigation state.
    Reads of read-only fields hit the shared :class:`AppContext`.
    """

    __slots__ = ("_ctx", "_initial")

    if TYPE_CHECKING:
        # Declared so mypy sees accurate types through the __getattr__
        # proxy; actual resolution happens at runtime.
        src_path: pathlib.Path
        tgt_path: pathlib.Path
        src_ds: netCDF4.Dataset
        tgt_ds: netCDF4.Dataset
        flags: list[Flag] | None
        max_cache_size: int
        active_mask: numpy.ndarray | None
        active_expression: str | None
        filter_info: FilterInfo | None
        group_results: list[GroupComparisonResult]
        variable_cache: collections.OrderedDict[str, DataSource]
        lock: threading.RLock
        current_view: str
        current_group_path: str
        current_varpath: str | None

    def __init__(
        self,
        src_path: pathlib.Path,
        tgt_path: pathlib.Path,
        src_ds: netCDF4.Dataset,
        tgt_ds: netCDF4.Dataset,
        flags: list[Flag] | None = None,
        active_mask: numpy.ndarray | None = None,
        active_expression: str | None = None,
        filter_info: FilterInfo | None = None,
        group_results: list[GroupComparisonResult] | None = None,
        max_cache_size: int = MAX_VARIABLE_CACHE_SIZE,
    ) -> None:
        ctx = AppContext(
            src_path=src_path,
            tgt_path=tgt_path,
            src_ds=src_ds,
            tgt_ds=tgt_ds,
            flags=flags,
            max_cache_size=max_cache_size,
        )
        initial = SessionData(
            active_mask=active_mask,
            active_expression=active_expression,
            filter_info=filter_info,
            group_results=group_results if group_results is not None else [],
        )
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_initial", initial)
        _set_default_session(initial)

    def __getattr__(self, name: str) -> Any:
        if name in _SESSION_FIELDS:
            return getattr(get_session(), name)
        ctx = object.__getattribute__(self, "_ctx")
        return getattr(ctx, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _SESSION_FIELDS:
            setattr(get_session(), name, value)
            return
        raise AttributeError(
            f"SessionState.{name} is read-only (belongs to AppContext)"
        )

    @property
    def ctx(self) -> AppContext:
        """Expose the underlying context (mainly for tests)."""
        ctx: AppContext = object.__getattribute__(self, "_ctx")
        return ctx

    def clear_variable_cache(self) -> None:
        """Clear the current session's variable data cache."""
        session = get_session()
        with session.lock:
            session.variable_cache.clear()
        LOGGER.info("Variable cache cleared")

    def close(self) -> None:
        """Close the NetCDF datasets and drop every session."""
        ctx = object.__getattribute__(self, "_ctx")
        try:
            ctx.src_ds.close()
        finally:
            ctx.tgt_ds.close()
        with _sessions_lock:
            _sessions.clear()
        global _default_session
        _default_session = None
        LOGGER.info("NetCDF datasets closed")
