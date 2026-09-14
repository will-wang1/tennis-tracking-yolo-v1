"""Render a `MatchStats` into a self-contained HTML report.

This is a pure templating step - `report_template.html` is the same
scoreboard/bounce-map/speed-chart/court-coverage/stroke-mix/serve-trend/
rally-log page built by hand as this project's "baseline report" artifact,
lifted into a template with two placeholders (`__DATA_JSON__`,
`__CLIP_LABEL_JSON__`) so any clip's `MatchStats` can produce its own copy
without touching a browser. Everything the page renders - the SVG bounce
map, the canvas speed chart, the movement bars - runs client-side in the
report's own `<script>`, off the embedded JSON; this module's only job is
filling that JSON in and writing the file. The output has no external
script dependency besides a Google Fonts stylesheet, so it opens fine
straight from disk (file://) with a live internet connection for the fonts,
and still renders (system-font fallback) without one.
"""

import html
import json
from pathlib import Path

from src.analysis.match_stats import MatchStats

_TEMPLATE_PATH = Path(__file__).resolve().parent / "report_template.html"


def clip_label_from_path(input_path) -> str:
    """Turn an input filename into a human title: 'zverev_rally.mp4' ->
    'Zverev Rally'. Just a display default - callers with a better name
    (e.g. from the command that invoked them) should pass it directly to
    `render_report_html`/`write_report_html` instead."""
    return Path(input_path).stem.replace("_", " ").replace("-", " ").title()


def render_report_html(stats: MatchStats, clip_label: str) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        template.replace("__DATA_JSON__", json.dumps(stats.to_dict()))
        # __CLIP_LABEL_JSON__ feeds a JS string literal (script context, no
        # HTML-escaping needed there); __CLIP_LABEL__ sits inside <title>
        # (markup context), so it gets HTML-escaped instead.
        .replace("__CLIP_LABEL_JSON__", json.dumps(clip_label))
        .replace("__CLIP_LABEL__", html.escape(clip_label))
    )


def write_report_html(stats: MatchStats, clip_label: str, path) -> None:
    Path(path).write_text(render_report_html(stats, clip_label), encoding="utf-8")
