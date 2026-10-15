"""Tests for the TOML config loader."""

from __future__ import annotations

from app.config import AppConfig, load_config


def test_load_full_config(tmp_path):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        """
[paths]
source = "a.nc"
target = "b.nc"

[filter]
expression = "${/x} > 0"
flags = ["/f1", "/f2"]

[server]
port = 9000
debug = true

[cache]
max_variable_entries = 7
max_display_points = 42

[logging]
level = "DEBUG"
"""
    )
    cfg = load_config(cfg_path)
    assert cfg.paths.source == "a.nc"
    assert cfg.paths.target == "b.nc"
    assert cfg.filter.expression == "${/x} > 0"
    assert cfg.filter.flags == ["/f1", "/f2"]
    assert cfg.server.port == 9000
    assert cfg.server.debug is True
    assert cfg.cache.max_variable_entries == 7
    assert cfg.cache.max_display_points == 42
    assert cfg.logging.level == "DEBUG"


def test_load_empty_config_uses_defaults(tmp_path):
    cfg_path = tmp_path / "empty.toml"
    cfg_path.write_text("")
    cfg = load_config(cfg_path)
    defaults = AppConfig()
    assert cfg.server.port == defaults.server.port
    assert cfg.filter.flags == []
    assert cfg.paths.source is None


def test_unknown_keys_are_ignored(tmp_path):
    cfg_path = tmp_path / "extra.toml"
    cfg_path.write_text(
        """
[server]
port = 5000
mystery_setting = "ignored"
"""
    )
    cfg = load_config(cfg_path)
    assert cfg.server.port == 5000


def test_annotations_section_and_defaults(tmp_path):
    import argparse

    from app.config import apply_config_defaults, load_config

    path = tmp_path / "cfg.toml"
    path.write_text(
        '[paths]\nsource = "a.nc"\n[annotations]\nfile = "ann.json"\nreadonly = true\n'
        "[comparison]\ntolerance_steps = 2.0\n"
    )
    config = load_config(path)
    assert config.annotations.file == "ann.json"
    assert config.annotations.readonly is True
    args = argparse.Namespace(
        source=None,
        annotations=None,
        annotations_readonly=False,
        expr="x",
        tolerance_steps=None,
    )
    apply_config_defaults(args, config)
    assert args.source == "a.nc"
    assert args.annotations == "ann.json"
    assert args.annotations_readonly is True
    assert args.expr == "x"
    assert not hasattr(args, "port")
    assert args.tolerance_steps == 2.0
    explicit_zero = argparse.Namespace(tolerance_steps=0.0)
    apply_config_defaults(explicit_zero, config)
    assert explicit_zero.tolerance_steps == 0.0
