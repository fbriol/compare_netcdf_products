# Architecture

Maintainer-facing overview of the `ncjuxta` NetCDF comparison app. Read
[README.md](README.md) first for user-facing behaviour.

## Bird's-eye view

```
                 ┌──────────── app.py ──────────────┐
                 │ argparse + TOML config merge     │
                 │ open NetCDF handles              │
                 │ build SessionState               │
                 │ register callbacks               │
                 │ dash_app.run()                   │
                 └──┬──────────────────────────┬────┘
                    │                          │
          ┌─────────▼──────────┐    ┌──────────▼────────────┐
          │ app/layout.py      │    │ app/callbacks/        │
          │ static skeleton    │    │ _registration.py      │
          │ (Stores, sidebar)  │    │ + views.py (dynamic)  │
          └─────────┬──────────┘    └──────────┬────────────┘
                    │                          │
                    └───────── reads ──────────┘
                            SessionState  (proxy)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          AppContext (shared)        SessionData (per flask.session sid)
           datasets, paths,           mask, filter, nav, cache,
           flags, cache size          group_results, RLock
                                 │
                  ┌──────────────┼─────────────────┐
                  ▼              ▼                 ▼
            comparison.py   data_loading.py   expression.py
                                 │                 │
                                 ▼                 ▼
                              plots.py          numexpr
```

Dash's callback graph is the runtime. Nearly all business logic lives
behind one of two surfaces:

1. **Server-side callbacks** in `app/callbacks/` — register outputs in
   terms of layout element IDs and mutate `SessionState`.
2. **Client-side callbacks** (inlined as strings in `_registration.py`)
   — handle pure UI work that shouldn't round-trip (tab toggles, marker
   size updates, etc.).

## Module responsibilities

### `app.py`
Process entry point. Parses CLI args, loads an optional TOML config
(CLI overrides), opens both NetCDF datasets, builds a `SessionState`,
registers callbacks, and runs the Dash server. The `try/finally`
around `dash_app.run` guarantees the datasets are closed via
`state.close()` even on startup failure.

### `app/state.py`
State is split in two and surfaced through a thin proxy so existing
callback code keeps reading/writing `state.x`:

- **`AppContext`** — process-global, read-only. Owns the open
  `netCDF4.Dataset` handles (`src_ds`, `tgt_ds`), file paths, CLI
  flags, and the cache-size limit. Shared by every tab.
- **`SessionData`** — per-session, mutable. Owns the current filter
  (`active_mask`, `active_expression`, `filter_info`), navigation
  state (`current_view`, `current_group_path`, `current_varpath`),
  cached `GroupComparisonResult` list (`group_results`), the
  per-session LRU `variable_cache`, and an `RLock` guarding cache
  mutations (Dash's dev server dispatches callbacks in multiple
  threads).
- **`SessionState`** — proxy. Reads of session-scoped fields route
  through `get_session()`; reads of context-scoped fields hit the
  shared `AppContext`. Assignments to context-scoped fields raise
  `AttributeError`.

Session lookup uses `flask.session["sid"]` — a UUID set on the first
request of each browser session. The server's `secret_key` is read
from `NCJUXTA_SECRET_KEY` or generated per process. Outside a request
context (startup, tests), `get_session()` returns a module-level
*default* session so initialisation code (`scan_all_groups`, initial
`--expr`) can mutate state before any browser has connected. New
sessions are lazily seeded from that default — `group_results` is
shared by reference (filter changes *reassign* the attribute rather
than mutating the shared list), while `variable_cache` starts empty
so tabs don't starve each other.

> **Known limitation.** `_sessions` grows unbounded. Acceptable for
> the internal-tool deployment profile; a bounded LRU plus a session
> expiry hook would be the fix for public multi-tenant use.

### `app/comparison.py`
Pure data logic — no Dash, no numpy magic. Dataclasses
`Stats`, `VariableComparisonResult`, `AttributeComparisonResult`,
`GroupComparisonResult` describe the shape of a comparison result.

Key entry points:
- `scan_all_groups(src, tgt, filter_info, mask)` — recursive walk,
  comparing every group.
- `rescan_affected_groups(...)` — re-evaluates only the groups whose
  variables intersect the filter mask. Other groups reuse the previous
  result with an updated filter description.
- `compare_variable(...)` — handles the masked-array arithmetic and
  classifies the variable into one of the six status values. Honours
  each variable's `scale_factor` attribute when deciding
  `almost_equal`.

### `app/expression.py`
Safe expression evaluator built on `numexpr`. Exposes:
- `parse_variable_refs(expr)` — returns every `${/path}` found.
- `rewrite_expression(expr)` — substitutes `${...}` with `v0`, `v1`, …
  placeholders and translates `np.isin(x, [a, b, c])` into a chain of
  `(x==a) | (x==b) | (x==c)` via an `ast.NodeTransformer`.
- `create_expression_mask(dataset, expr)` — loads the required
  variables, evaluates with `numexpr.evaluate`, validates that the
  result is a boolean `ndarray`, and returns `(mask, None)` or
  `(None, error_message)`.
- `find_affected_groups(expr)` — used by `rescan_affected_groups` to
  avoid re-comparing groups whose variables weren't referenced.

`numexpr` is *safe by construction*: no attribute access, no Python
calls, no imports can be expressed in its grammar. There is no Python
`eval()` anywhere in the evaluator.

### `app/data_loading.py`
Bridges NetCDF → plot-friendly numpy. Highlights:
- `DataSource` — wraps a pair of `expected` / `computed` arrays plus
  optional longitude/latitude. It promotes integers to float, fills
  masks with NaN, computes the difference, detects "exclusive NaNs"
  (present in one file but not the other), and builds downsampled
  display arrays.
- `lttb_downsample(x, y, n_out)` — vectorised
  Largest-Triangle-Three-Bucket algorithm. Triggers above
  `MAX_DISPLAY_POINTS` (see `constants.py`).
- `load_variable_data(state, varpath)` — the LRU cache entry point.
  Always wrapped in `state.lock` because Dash may fire two callbacks
  concurrently against the same cache. Cache invalidation keys off the
  active mask.
- `_find_coordinates(...)` — discovers lon/lat by first inspecting the
  variable's `coordinates` attribute, then walking up the group tree.

### `app/plots.py`
A family of Plotly figure factories (`fig_signal_comparison`,
`fig_signal_difference`, `fig_nan_distribution`, `fig_scatter`,
`fig_histogram`, `fig_stats_table`, `fig_geo_map`). `build_all_figures`
composes the list of `(label, figure)` tuples that the dynamic view
renders; it inserts the geo map when coordinates exist and the NaN
distribution when `exclusive_nan` is true.

Outlier traces are always named `Source (outlier)` / `Target (outlier)`
so the clientside marker-size callback can match them and apply the
`max(OUTLIER_MIN_SIZE, markerSize + OUTLIER_OFFSET)` clamp.

### `app/flags.py`
`Flag` reads a CF flag variable, pairs `flag_values` with
`flag_meanings` under `strict=True` (so malformed metadata raises
early), and exposes `compute_filtered_indices` — intersects every
active flag selection into a single index array.

### `app/layout.py`
Static skeleton: sidebar (file info, legend, exports, expression
editor), header, central content container, and a set of `dcc.Store`
components used for server↔client state exchange. Required IDs are
asserted in `tests/test_layout.py` to catch silent drift.

### `app/callbacks/`
- `__init__.py` / `_registration.py` — `register_callbacks(app, state)`.
  Every callback captures `state` in its closure. Contains the
  long-running logic: apply expression, apply flag filter, navigate,
  render group/variable views, export, share URL.
- `views.py` — builds the dynamic "variable view" (the
  plots-and-controls block). This is where the Settings row and
  outlier detection live.

Most callbacks return either a Dash `Patch`, a tuple of outputs, or
`no_update`. Strict mypy is relaxed in this package only (see the
override in `pyproject.toml`).

### `app/config.py`
Thin `tomllib` wrapper. One dataclass per config section
(`paths`, `filter`, `server`, `cache`, `logging`) plus an umbrella
`AppConfig`. Unknown keys are ignored so newer configs remain
backwards-compatible. `app.py::_apply_config_defaults` merges the
config into the argparse namespace (CLI wins).

### `app/report.py`
CSV and JSON serializers for the cached `GroupComparisonResult` list.
CSV is one row per variable; JSON is the full tree via
`dataclasses.asdict`. `_default` handles numpy scalars and sets.

### `app/utils.py`
Tiny helpers (`count_diffs`, `count_diffs_recursive`, group navigation
utilities).

## Request flow examples

### Applying an expression filter

1. User types an expression and clicks *Apply*.
2. `apply_expression` callback (`_registration.py`) calls
   `create_expression_mask`.
3. On success it stores the new mask on `SessionState`, computes the
   affected groups, and calls `rescan_affected_groups` to refresh only
   the impacted `group_results` entries.
4. The cache is cleared (`state.clear_variable_cache`) so downstream
   `DataSource` objects are rebuilt against the new mask.
5. The filter summary + table re-render.

### Opening a variable

1. User clicks a "Different" row.
2. `nav-varpath` Store updates and a callback renders the variable
   view via `app/callbacks/views.py`.
3. `views.py` calls `load_variable_data(state, varpath)` — cache hit
   returns immediately, miss triggers NetCDF reads inside the lock.
4. `build_all_figures(varname, data_source, overlay=...)` produces
   the figure list; outlier and overlay logic is layered on top.
5. The clientside callback takes over for marker-size adjustments
   (no server round-trip).

## Threading & resource-safety

- `SessionData.lock` is an `RLock` (per session) so helpers that take
  it can call other lock-acquiring helpers without deadlock.
- `_sessions_lock` guards the session registry itself (lookup /
  insertion). Kept minimal to avoid contention.
- `app.py` wraps `dash_app.run()` in `try/finally` + `state.close()`.
- `state.close()` closes both datasets even if the first `close()`
  raises, then drops every session — covered by
  `tests/test_state.py`.
- netCDF4 handles are read-only and opened once per process. Dash's
  threaded server serves concurrent reads; writes are avoided.

## Testing

- `tests/conftest.py` — `tmp_nc_pair` fixture (minimal NetCDF pair),
  `open_nc_pair` (opens both read-only and closes at teardown).
- `test_expression.py` — parsing, rewriting, AST-based `np.isin`
  translation, mask generation, RCE attempts that now fail at the
  numexpr parser.
- `test_comparison.py` / `test_data_loading.py` / `test_utils.py` —
  core comparison logic.
- `test_flags.py`, `test_state.py`, `test_plots.py`, `test_layout.py`,
  `test_config.py`, `test_report.py` — new coverage for the
  previously-untested modules plus config & report round-trips.

Run `pre-commit run --all-files && python -m pytest tests/ -q` before
every PR — `ruff` (with autofix + formatting) and `mypy` are pinned in
`.pre-commit-config.yaml` and driven by `pre-commit`, not invoked
directly.

## Conventions

- Type annotations everywhere **except** `app/callbacks/*` (opaque
  Dash return shapes). Mypy enforces `disallow_untyped_defs` elsewhere.
- `from __future__ import annotations` in every module.
- No explanatory comments in code — docstrings explain the *why* when
  non-obvious. See the inline feedback rules for what qualifies.
- Ruff's `E, W, F, I, UP, B, SIM, RUF` rule groups are enabled.
