# NetCDF Comparison App

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
- Annotations: describe the analysed differences in a JSON file (edited
  by hand or in the built-in web editor), see them next to the plots,
  and generate an interactive HTML or PDF report for the client.
- Optional TOML config file for reproducible runs.
- Shareable URL state (view, group, variable, zoom, filter).
- Per-tab / per-user session isolation: two browser tabs keep
  independent filter, navigation and cache state.

## Installation

Requires **Python ≥ 3.12**.

```bash
git clone <this-repo> ncjuxta
cd ncjuxta
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
| `--tolerance-steps N` | Differences up to N storage steps (`N × scale_factor`) are classified *almost equal* (default `1.5`). See *Tolerance*. |
| `--annotations FILE` | Annotation JSON file to display and edit (created if missing). See *Annotations & reports*. |
| `--annotations-readonly` | Show the annotations without the editing controls. |
| `--debug` | Enable Dash debug mode (auto-reload, verbose errors). |

### Environment variables

| Variable | Description |
|----------|-------------|
| `NCJUXTA_SECRET_KEY` | Flask session secret. If unset a fresh random key is generated per process, which invalidates existing session cookies on restart. Set it (e.g. `export NCJUXTA_SECRET_KEY=$(python -c 'import secrets;print(secrets.token_hex(32))')`) to keep tabs authenticated across restarts. |

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

[comparison]
tolerance_steps = 1.5

[server]
port = 5006
debug = false

[cache]
max_variable_entries = 5
max_display_points = 20000

[logging]
level = "INFO"

[annotations]
file = "annotations.json"
readonly = false
```

### Tolerance

Packed variables (integer + `scale_factor`) can only differ by whole
storage steps. Two processing chains that round the last stored unit
differently produce one-step differences, which are not significant;
moreover, once decoded to floating point, a one-step difference is
sometimes slightly *above* `scale_factor` (about 40 % of the time for
`1e-6`). A variable is therefore classified *almost equal* when every
difference is at most `tolerance_steps × scale_factor`, 1.5 steps by
default: one step plus margin for the decoding error. The same threshold
counts the measurements *out of tolerance* in the CSV export and the
annotated report. Variables without `scale_factor` are compared exactly.

### Using the interface

- **Sidebar**: file info, legend, export buttons (HTML / CSV / JSON),
  and the expression editor. Toggle the sidebar with the arrow at its
  right edge.
- **Main table view**: browse groups, click a group to drill in, click
  a variable marked *Different* or *Almost equal* to open its plots.
- **Plot view**:
  - Tick the plots you want to display from the *Visible* row. The view
    opens on *Signal Difference*: overlaid source/target curves usually
    hide differences that are tiny compared with the signal.
  - The **⛶** button next to a plot title shows it full screen (handy for
    the geographical map); press **Esc** to exit. The annotated report has
    the same button.
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

## Annotations & reports

When a computed product is validated against a reference, every
difference must be explained: some need an action on the input data,
some are expected (specification or method changes), some revealed a
defect that was fixed. ncjuxta keeps these explanations in an
**annotation file** (JSON) and turns them into a report the client can
explore on their own.

### The annotation file

A complete example, modelled on a real validation report, is in
[examples/annotations.example.json](examples/annotations.example.json).
The main blocks are:

| Key | Content |
|-----|---------|
| `title`, `subtitle`, `date`, `authors` | Report header. |
| `language` | `en` or `fr`: language of the report's fixed texts and number format. |
| `source_label`, `target_label` | Names replacing *Source*/*Target* in the report and its plots (e.g. `PDAP`, `c3po`). |
| `datasets` | `{label, value}` rows of the *compared data* table. |
| `categories` | `{id, label, color, description}`: families of remarks (e.g. *action required*, *expected difference*). |
| `sections` | The report outline: `{id, title, body (Markdown), annotations: [ids], auto, subsections}`. `auto` inserts generated content: `summary_counts`, `agreement_table` (every differing variable, its measurements out of tolerance and the annotations covering it) or `unannotated` (differences still to analyse). |
| `annotations` | The remarks: `{id, title, category, status, variables, text, effect, magnitude, cause, refs, figures}`. `variables` accepts exact paths or `fnmatch` globs (`/data_20/ka/*`). `status` is `open`, `confirmed`, `resolved` or `wontfix`. |
| `figures` | Per annotation: `{variable, plots, expression, zoom, caption}`. `plots` keeps a subset of the plot names (`Signal Comparison`, `Signal Difference`, `Scatter Plot`, `Histogram`, `Statistics`, `Geographical Position`, `Exclusive Undefined Positions`); `expression` filters the data with the usual syntax; `zoom` is an `[xmin, xmax]` index range (the statistics are restricted to it too). |

Annotations not placed in any section appear under *Other annotations*.

Helpers:

```bash
python -m app.annotations template ref.nc tgt.nc -o annotations.json  # one empty annotation per difference
python -m app.annotations check annotations.json ref.nc tgt.nc       # validate + list unannotated differences
python -m app.annotations schema > annotations.schema.json           # JSON Schema for editor autocompletion
```

### In the app

```bash
python app.py ref.nc tgt.nc --annotations annotations.json
```

- The global summary and the group tables show the annotations covering
  each variable; a banner counts how many differing variables are annotated.
- The variable view has an **Annotations** panel. *Show this view*
  reproduces a figure of an annotation (filter, zoom, visible plots).
  *Annotate this variable* opens a form prefilled with the current
  variable, filter, zoom and visible plots; *Add current view as figure*
  appends what you are looking at to an existing annotation.
- The **📝 Annotations** button opens the editor: outline (sections,
  ordering, annotations), document metadata and categories. Markdown
  fields have a live preview; **Ctrl+S** saves.
- Every save validates the document, writes the file atomically and keeps
  the previous version as `annotations.json.bak`. If the file was edited
  on disk (or saved from another tab) in the meantime, the app refuses to
  overwrite it silently: use *Reload annotations from disk* or *Overwrite
  anyway*.
- *Export annotated report (HTML)* in the sidebar downloads the report
  for the current filter.

### Generating the report

```bash
ncjuxta-report ref.nc tgt.nc --annotations annotations.json -o report.html
ncjuxta-report ref.nc tgt.nc --annotations annotations.json -o report.pdf
```

The HTML report is a single self-contained file (plotly.js embedded,
works offline; `--plotlyjs cdn` makes it smaller): table of contents,
category/status filters, search, sortable agreement table, and
interactive figures drawn as they scroll into view. It prints cleanly
from a browser.

PDF output needs the optional extra, which prints the same page with
headless Chromium:

```bash
pip install -e ".[pdf]"
playwright install chromium
```

`--strict` makes the command exit with status 2 when a differing
variable has no annotation (useful in CI); `--expr`, `--flag` and
`--config` work as for the app.

## Running the tests

```bash
pip install -e ".[dev]"
pre-commit install            # first time only
pre-commit run --all-files    # ruff (+autofix), ruff-format, mypy
python -m pytest tests/ -q
```

`pre-commit` manages pinned versions of `ruff` and `mypy` in
`.pre-commit-config.yaml`, so you don't need to install them separately.

## Project layout

```
app.py              # entry point (argparse, session wiring)
app/                # application package
├── annotated_report.py # annotated HTML / PDF report builder
├── annotations.py  # annotation file model, validation, on-disk store
├── callbacks/      # Dash callbacks (registration + dynamic views,
│                   #   annotation panel and editor)
├── comparison.py   # group/variable comparison logic
├── config.py       # TOML config loader
├── constants.py    # colour palette, cache sizes
├── data_loading.py # lazy variable loading, LTTB downsampling, cache
├── expression.py   # numexpr-based safe filter evaluator
├── flags.py        # CF flag variables
├── layout.py       # static Dash layout
├── plots.py        # Plotly figure factories
├── report.py       # CSV / JSON report serializers
├── report_cli.py   # ncjuxta-report command
├── state.py        # SessionState (datasets, cache, nav state)
└── utils.py        # small helpers
assets/             # stylesheet + Ctrl+S helper, auto-loaded by Dash
examples/           # example annotation file
tests/              # pytest suite
```

For a deeper dive into how the pieces fit together, see
[ARCHITECTURE.md](ARCHITECTURE.md).
