"""Tests for app/state.py."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from app.state import SessionState


def _make_state(**overrides: MagicMock) -> SessionState:
    kwargs: dict[str, MagicMock] = {
        "src_path": MagicMock(),
        "tgt_path": MagicMock(),
        "src_ds": MagicMock(),
        "tgt_ds": MagicMock(),
    }
    kwargs.update(overrides)
    return SessionState(**kwargs)


def test_close_closes_both_even_when_src_raises() -> None:
    src = MagicMock()
    src.close.side_effect = RuntimeError("boom")
    tgt = MagicMock()
    state = _make_state(src_ds=src, tgt_ds=tgt)
    with pytest.raises(RuntimeError):
        state.close()
    tgt.close.assert_called_once()


def test_clear_variable_cache_clears_and_is_thread_safe() -> None:
    state = _make_state()
    state.variable_cache["a"] = MagicMock()
    state.variable_cache["b"] = MagicMock()

    def concurrent_insert() -> None:
        with state.lock:
            state.variable_cache["c"] = MagicMock()

    t = threading.Thread(target=concurrent_insert)
    t.start()
    t.join()
    state.clear_variable_cache()
    assert len(state.variable_cache) == 0


def test_lock_is_reentrant() -> None:
    state = _make_state()
    with state.lock, state.lock:
        state.variable_cache["x"] = MagicMock()
    assert "x" in state.variable_cache


def test_appcontext_fields_are_read_only() -> None:
    state = _make_state()
    with pytest.raises(AttributeError):
        state.src_path = "something-else"  # type: ignore[misc,assignment]


def test_sessions_are_isolated_across_flask_sessions() -> None:
    import flask

    from app.state import _sessions

    state = _make_state()
    _sessions.clear()

    app = flask.Flask(__name__)
    app.secret_key = "test"

    with app.test_request_context():
        flask.session["sid"] = "tab-a"
        state.active_expression = "${/x} > 0"

    with app.test_request_context():
        flask.session["sid"] = "tab-b"
        # tab-b is freshly seeded from the default; the expression that
        # tab-a set must not leak across.
        assert state.active_expression != "${/x} > 0"
        state.active_expression = "${/y} < 10"

    with app.test_request_context():
        flask.session["sid"] = "tab-a"
        assert state.active_expression == "${/x} > 0"
