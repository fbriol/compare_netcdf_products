# cmp — NetCDF Comparison App

Interactive Dash application for comparing two NetCDF files side-by-side.
Built for altimetry / SWOT-style hierarchical datasets, but works with any
NetCDF4 file that shares a common structure between a *reference* and a
*target* version.

The app walks every group, compares variables, highlights differences,
and renders signal, scatter, histogram, NaN-distribution and
geographical plots on demand. Filtering is available through flag
variables and safe numerical expressions.

## Features

- Recursive group-by-group comparison (attributes + variables).
- Status classification per variable: `equal`, `almost_equal` (within
  `scale_factor` tolerance), `different`, `masked`, `missing_in_source`,
  `missing_in_target`.
- Interactive plots (Plotly): signal comparison, difference, scatter,
  histogram, geographical map, NaN distribution, statistics table.
- Expression-based filtering with a safe numexpr backend
  (`${/path/to/var} > 30000`, `np.isin(...)` supported).
- Flag-based filtering using CF `flag_values` / `flag_meanings`.
- Outlier detection with adjustable N×σ threshold and marker size.
- Overlay any 1-D variable on the Signal Comparison plot's secondary axis.
- Export: standalone HTML, CSV report, JSON report.
- Optional TOML config file for reproducible runs.
- Shareable URL state (view, group, variable, zoom, filter).
- Per-tab / per-user session isolation: two browser tabs keep
  independent filter, navigation and cache state.

## Installation

Requires **Python ≥ 3.12**.

```bash
git clone <this-repo> cmp
cd cmp
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project uses `pyproject.toml` as its single source of truth; core
runtime dependencies are `dash`, `plotly`, `netCDF4`, `numpy`, and
`numexpr`. Development tools (`pytest`, `ruff`, `mypy`) are invoked
from your environment — install them alongside if you plan to hack on
the code.

## Usage

### Command line

```bash
python app.py <source.nc> <target.nc> [options]
```

Minimal example:

```bash
python app.py ref.nc tgt.nc
```

Then open <http://localhost:5006>.

### Options

| Flag | Description |
|------|-------------|
| `--config FILE` | Load settings from a TOML file. CLI args override config values. |
| `--flag VAR_PATH [VAR_PATH ...]` | One or more flag variables to expose in the sidebar. The variables must carry `flag_meanings` and `flag_values` attributes. |
| `--expr EXPRESSION` | Initial filter expression. See *Expression syntax* below. |
| `--port N` | Port to serve on (default `5006`). |
| `--debug` | Enable Dash debug mode (auto-reload, verbose errors). |

### Environment variables

| Variable | Description |
|----------|-------------|
| `CMP_SECRET_KEY` | Flask session secret. If unset a fresh random key is generated per process, which invalidates existing session cookies on restart. Set it (e.g. `export CMP_SECRET_KEY=$(python -c 'import secrets;print(secrets.token_hex(32))')`) to keep tabs authenticated across restarts. |

Example with flags and an expression filter:

```bash
python app.py ref.nc tgt.nc \
    --flag /data_01/surface_classification_flag \
           /data_20/surface_classification_flag \
           /data_01/ku/wvf_main_class \
           /data_20/ku/wvf_main_class \
    --expr '${/data_20/distance_to_coast} > 30000'
```

### Expression syntax

Expressions reference NetCDF variables with `${/path/to/variable}` and
are evaluated by [`numexpr`](https://numexpr.readthedocs.io/) — safe by
construction (no attribute access, no Python-level calls, no imports).

Supported:
- Arithmetic: `+ - * / ** %`
- Comparisons: `== != < <= > >=`
- Bitwise logic on boolean masks: `& | ~`
- `numexpr` built-ins: `where`, `sin`, `cos`, `exp`, `log`, `sqrt`,
  `abs`, `sum`, …
- `np.isin(x, [a, b, c])` — rewritten to `((x==a) | (x==b) | (x==c))`.

The result **must** be a boolean array with the same shape as the
variables being filtered.

Examples:

```text
${/data_20/distance_to_coast} > 30000
(${/data_20/distance_to_coast} > 30000) & (${/data_20/flag} == 0)
np.isin(${/data_01/surface_classification_flag}, [0, 1, 2])
```

### Config file

Pass `--config my.toml` to preload settings. CLI args always win over
config values.

```toml
[paths]
source = "ref.nc"
target = "tgt.nc"

[filter]
expression = "${/data_20/distance_to_coast} > 30000"
flags = [
    "/data_01/surface_classification_flag",
    "/data_20/surface_classification_flag",
]

[server]
port = 5006
debug = false

[cache]
max_variable_entries = 5
max_display_points = 20000

[logging]
level = "INFO"
```

### Using the interface

- **Sidebar**: file info, legend, export buttons (HTML / CSV / JSON),
  and the expression editor. Toggle the sidebar with the arrow at its
  right edge.
- **Main table view**: browse groups, click a group to drill in, click
  a variable marked *Different* or *Almost equal* to open its plots.
- **Plot view**:
  - Tick the plots you want to display from the *Visible* row.
  - **⚙ Settings** reveals marker-size, outlier-threshold and overlay
    controls.
  - Use the outlier badge to gauge how many points exceed the N×σ
    threshold — they appear as yellow diamonds.
- **Share**: writes the current app state (view, group, variable,
  zoom, filter) into the URL so you can bookmark or send it.

### Exports

- **Export HTML** — a self-contained HTML file of the currently-selected
  variable's plots (plotly.js loaded from CDN).
- **Export CSV report** — one row per variable across every group with
  shape/dtype/status and diff statistics.
- **Export JSON report** — the full comparison tree serialised via
  `dataclasses.asdict`.

## Running the tests

```bash
python -m pytest tests/ -q
python -m ruff check app/ tests/
python -m mypy app/
```

## Project layout

```
app.py              # entry point (argparse, session wiring)
app/                # application package
├── callbacks/      # Dash callbacks (registration + dynamic views)
├── comparison.py   # group/variable comparison logic
├── config.py       # TOML config loader
├── constants.py    # colour palette, cache sizes
├── data_loading.py # lazy variable loading, LTTB downsampling, cache
├── expression.py   # numexpr-based safe filter evaluator
├── flags.py        # CF flag variables
├── layout.py       # static Dash layout
├── plots.py        # Plotly figure factories
├── report.py       # CSV / JSON report serializers
├── state.py        # SessionState (datasets, cache, nav state)
└── utils.py        # small helpers
assets/cmp.css      # stylesheet auto-loaded by Dash
tests/              # pytest suite
```

For a deeper dive into how the pieces fit together, see
[ARCHITECTURE.md](ARCHITECTURE.md).
