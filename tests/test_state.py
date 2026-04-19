"""Tests for app/state.py."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from app.state import SessionState


def _make_state(**overrides):
    kwargs = {
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
