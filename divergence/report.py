"""Builds results/report.html: one self-contained page that works offline.

It holds the verdict, the in-sample vs. out-of-sample table, equity curves,
the placebo test, and data coverage. Every number is read from the files the
`backtest` command wrote, so the page always matches what was printed.

The page styling and chart code here are shared with the website (site.py),
so the two always look and behave the same.
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Written down before the out-of-sample test was run, so success cannot be
# redefined after seeing the answer. All three must hold.
SUCCESS_CRITERIA = [
    ("sharpe_positive", "Out-of-sample Sharpe ratio above 0 after costs"),
    ("beats_price_only", "Beats the same rule without news, out-of-sample"),
    ("beats_placebo", "Placebo p-value below 0.05 out-of-sample (real news beats shuffled news)"),
]

PERIOD_NAMES = {"in_sample": "2015–2021 (development)", "out_of_sample": "2022–today (test)"}

STAT_COLUMNS = [
    ("annual_return", "Annual return"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "Max drawdown"),
    ("hit_rate", "Hit rate"),
    ("trades", "Trades"),
    ("exposure", "Time in market"),
    ("t_stat", "t-stat"),
]


def series_order(results: dict) -> list[str]:
    """Fixed order, so each strategy keeps its color everywhere."""
    main = f"news divergence ({results['signal_parameters']['sentiment_model']})"
    names = [row["strategy"] for row in results["table"]]
    others = sorted({n for n in names if n.startswith("news divergence") and n != main})
    return [main, *others, "price only (no news)", "buy and hold (equal weight)"]


def verdict(results: dict) -> tuple[bool | None, list[tuple[str, bool]]]:
    if "out_of_sample" not in results["periods"]:
        return None, []
    main = series_order(results)[0]
    oos = {r["strategy"]: r for r in results["table"] if r["period"] == "out_of_sample"}
    checks = {
        "sharpe_positive": oos[main]["sharpe"] > 0,
        "beats_price_only": oos[main]["sharpe"] > oos["price only (no news)"]["sharpe"],
        "beats_placebo": results["placebo"]["out_of_sample"]["p_value"] < 0.05,
    }
    outcome = [(text, bool(checks[key])) for key, text in SUCCESS_CRITERIA]
    return all(ok for _, ok in outcome), outcome


def verdict_html(results: dict | None) -> str:
    passed, checks = verdict(results) if results else (None, [])
    if passed is None:
        return ("<p class='verdict pending'><strong>Not tested yet.</strong> The 2022–today test "
                "period is still locked; only development-period results exist.</p>")
    items = "".join(f"<li>{'✔' if ok else '✘'} {html.escape(text)}</li>" for text, ok in checks)
    word = "held up" if passed else "did not hold up"
    return (f"<p class='verdict {'pass' if passed else 'fail'}'><strong>The news signal {word} "
            f"out-of-sample.</strong> Success was defined before the test was run, and all three "
            f"criteria had to hold:</p><ul class='checks'>{items}</ul>")


def cell(key: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if key in ("annual_return", "max_drawdown"):
        return f"{value:+.1%}"
    if key in ("hit_rate", "exposure"):
        return f"{value:.0%}"
    if key == "trades":
        return f"{int(value):,}"
    return f"{value:.2f}"


def results_table_html(results: dict, columns=STAT_COLUMNS) -> str:
    rows = {(r["strategy"], r["period"]): r for r in results["table"]}
    periods = list(results["periods"])
    head = "".join(f"<th colspan='{len(periods)}'>{label}</th>" for _, label in columns)
    sub = "".join(f"<th class='sub'>{'Dev' if p == 'in_sample' else 'Test'}</th>"
                  for _ in columns for p in periods)
    body = []
    for k, name in enumerate(series_order(results)):
        cells = [f"<td>{cell(key, rows.get((name, p), {}).get(key))}</td>" for key, _ in columns for p in periods]
        body.append(f"<tr><th scope='row'><span class='key s{k + 1}'></span>{html.escape(name)}</th>"
                    + "".join(cells) + "</tr>")
    return (f"<div class='table-wrap'><table><thead><tr><th rowspan='2'>Strategy</th>{head}</tr>"
            f"<tr>{sub}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def markdown_table(results: dict) -> str:
    """The same table for the README."""
    rows = {(r["strategy"], r["period"]): r for r in results["table"]}
    lines = ["| Strategy | Period | Annual return | Sharpe | Max drawdown | Hit rate | Trades |",
             "|---|---|---|---|---|---|---|"]
    for name in series_order(results):
        for p in results["periods"]:
            r = rows.get((name, p))
            if not r:
                continue
            lines.append(f"| {name} | {PERIOD_NAMES[p]} | " + " | ".join(
                cell(key, r.get(key)) for key in
                ("annual_return", "sharpe", "max_drawdown", "hit_rate", "trades")) + " |")
    return "\n".join(lines)


# --- charts (inline SVG, drawn here so the page needs no internet) ---------------

W, H = 720, 280
LEFT, RIGHT, TOP, BOTTOM = 64, 16, 14, 30


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = high - low
    step = 10 ** np.floor(np.log10(span / count))
    for multiple in (1, 2, 2.5, 5, 10):
        if span / (step * multiple) <= count:
            step *= multiple
            break
    start = np.ceil(low / step) * step
    return [round(v, 10) for v in np.arange(start, high + step / 2, step)]


def money(value: float, large: bool) -> str:
    return f"${value:,.0f}" if large else f"${value:g}"


def value_chart(values: pd.DataFrame, names: list[str], chart_id: str, label: str) -> str:
    """Line chart of dollar values over time, with a crosshair tooltip.

    values: one column per series, indexed by date; the first row is the start.
    """
    large = float(values.iloc[0].max()) >= 1000
    low, high = float(values.min().min()), float(values.max().max())
    pad = (high - low) * 0.06 or abs(high) * 0.01 or 0.05
    low, high = low - pad, high + pad
    x_of = lambda i: LEFT + (W - LEFT - RIGHT) * i / max(len(values) - 1, 1)
    y_of = lambda v: TOP + (H - TOP - BOTTOM) * (1 - (v - low) / (high - low))

    parts = []
    for tick in nice_ticks(low, high):
        y = y_of(tick)
        parts.append(f"<line class='grid' x1='{LEFT}' x2='{W - RIGHT}' y1='{y:.1f}' y2='{y:.1f}'/>"
                     f"<text class='tick' x='{LEFT - 6}' y='{y + 4:.1f}' text-anchor='end'>{money(tick, large)}</text>")
    positions = pd.Series(range(len(values)), index=values.index)
    long_span = (values.index[-1] - values.index[0]).days > 400
    groups = positions.groupby(values.index.year if long_span else values.index.to_period("M")).first()
    step = max(1, len(groups) // 7)
    for key, i in list(groups.items())[::step]:
        text = str(key) if long_span else values.index[i].strftime("%b %Y")
        parts.append(f"<text class='tick' x='{x_of(i):.1f}' y='{H - 8}' text-anchor='middle'>{text}</text>")
    start = float(values.iloc[0].iloc[0])
    parts.append(f"<line class='axis' x1='{LEFT}' x2='{W - RIGHT}' y1='{y_of(start):.1f}' y2='{y_of(start):.1f}'/>")
    for k, name in enumerate(names):
        points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values[name]))
        parts.append(f"<polyline class='line s{k + 1}' points='{points}'/>")
    parts.append(f"<line class='crosshair' x1='0' x2='0' y1='{TOP}' y2='{H - BOTTOM}'/>"
                 f"<rect class='hit' x='{LEFT}' y='{TOP}' width='{W - LEFT - RIGHT}' height='{H - TOP - BOTTOM}'/>")

    data = {"dates": [d.strftime("%Y-%m-%d") for d in values.index], "names": names,
            "values": [[round(float(v), 4) for v in values[n]] for n in names],
            "x0": LEFT, "x1": W - RIGHT, "large": large}
    return (f"<figure class='chart' id='{chart_id}'><svg viewBox='0 0 {W} {H}' role='img' "
            f"aria-label='{html.escape(label)}'>{''.join(parts)}</svg>"
            f"<script type='application/json'>{json.dumps(data)}</script></figure>")


def growth_of_one_dollar(daily_returns: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    growth = (1 + daily_returns[names].fillna(0)).cumprod()
    start = pd.DataFrame([[1.0] * len(names)], columns=names, index=[growth.index[0] - pd.Timedelta(days=1)])
    return pd.concat([start, growth])


def placebo_chart(sharpes: list[float], real: float, chart_id: str) -> str:
    """Histogram of shuffled-sentiment Sharpe ratios, with the real strategy marked."""
    values = np.array([s for s in sharpes if not np.isnan(s)])
    if len(values) == 0 or np.isnan(real):
        return "<p class='note'>Not enough trades to compare.</p>"
    low, high = min(values.min(), real), max(values.max(), real)
    pad = (high - low) * 0.08 or 0.1
    edges = np.linspace(low - pad, high + pad, 21)
    counts, _ = np.histogram(values, edges)
    h = 200
    x_of = lambda v: LEFT + (W - LEFT - RIGHT) * (v - edges[0]) / (edges[-1] - edges[0])
    y_of = lambda c: TOP + (h - TOP - BOTTOM) * (1 - c / max(counts.max(), 1))
    parts = [f"<line class='axis' x1='{LEFT}' x2='{W - RIGHT}' y1='{h - BOTTOM}' y2='{h - BOTTOM}'/>"]
    slot = (W - LEFT - RIGHT) / len(counts)
    bar = min(24, slot - 2)
    for c, left_edge, right_edge in zip(counts, edges[:-1], edges[1:]):
        if c == 0:
            continue
        x = (x_of(left_edge) + x_of(right_edge)) / 2 - bar / 2
        top, base, r = y_of(c), h - BOTTOM, min(4, (h - BOTTOM - y_of(c)) / 2)
        tip = html.escape(f"{c} shuffled runs with Sharpe {left_edge:.2f} to {right_edge:.2f}")
        parts.append(f"<path class='bar' tabindex='0' data-tip='{tip}' d='M{x:.1f},{base} V{top + r:.1f} "
                     f"Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} H{x + bar - r:.1f} "
                     f"Q{x + bar:.1f},{top:.1f} {x + bar:.1f},{top + r:.1f} V{base} Z'/>")
    for tick in nice_ticks(edges[0], edges[-1], 6):
        parts.append(f"<text class='tick' x='{x_of(tick):.1f}' y='{h - 10}' text-anchor='middle'>{tick:g}</text>")
    xr = x_of(real)
    parts.append(f"<line class='marker' x1='{xr:.1f}' x2='{xr:.1f}' y1='{TOP}' y2='{h - BOTTOM}'/>"
                 f"<text class='label' x='{xr + 6:.1f}' y='{TOP + 12}'>real news: {real:.2f}</text>")
    return (f"<figure class='chart' id='{chart_id}'><svg viewBox='0 0 {W} {h}' role='img' "
            f"aria-label='Sharpe ratios of shuffled-sentiment runs compared with the real strategy'>"
            f"{''.join(parts)}</svg></figure>")


def equity_table(values: pd.DataFrame, names: list[str]) -> str:
    year_end = values[names].groupby(values.index.year).last()
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = "".join(f"<tr><th scope='row'>{y}</th>" + "".join(f"<td>${v:.2f}</td>" for v in row) + "</tr>"
                   for y, row in zip(year_end.index, year_end.to_numpy()))
    return (f"<details><summary>Show as a table (value of $1 at each year end)</summary>"
            f"<div class='table-wrap'><table><thead><tr><th>Year</th>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div></details>")


def legend(names: list[str]) -> str:
    return "<ul class='legend'>" + "".join(
        f"<li><span class='key s{k + 1}'></span>{html.escape(n)}</li>" for k, n in enumerate(names)) + "</ul>"


# --- the page -----------------------------------------------------------------

def build(results_dir: Path, headline_counts: pd.DataFrame) -> Path:
    results = json.loads((results_dir / "backtest.json").read_text())
    curves = pd.read_csv(results_dir / "daily_returns.csv", index_col="date", parse_dates=True)
    names = [n for n in series_order(results) if n in curves.columns]

    sections = []
    for period, (start, end) in results["periods"].items():
        growth = growth_of_one_dollar(curves.loc[start:end], names)
        sections.append(f"<h3>{PERIOD_NAMES[period]}</h3>"
                        f"<p class='note'>Each strategy starts with $1 on {start}.</p>"
                        + value_chart(growth, names, f"eq-{period}", "Growth of one dollar for each strategy")
                        + equity_table(growth, names))

    placebo_parts = []
    for period, p in results["placebo"].items():
        real = next(r["sharpe"] for r in results["table"]
                    if r["strategy"] == names[0] and r["period"] == period)
        placebo_parts.append(
            f"<h3>{PERIOD_NAMES[period]}</h3><p class='note'>p-value {p['p_value']:.3f}: the share of "
            f"{p['runs']} shuffled runs that did at least as well as the real one.</p>"
            + placebo_chart(p["sharpes"], real, f"pl-{period}"))

    corr_rows = "".join(
        f"<tr><th scope='row'>{PERIOD_NAMES[period]}</th><td>{html.escape(model)}</td>"
        f"<td>{c['correlation']:+.4f}</td><td>{c['n']:,}</td></tr>"
        for period, by_model in results["correlation"].items() for model, c in by_model.items())

    counts_html = ""
    if not headline_counts.empty:
        totals = headline_counts.sum()
        counts_html = ("<div class='table-wrap'><table><thead><tr><th>Year</th>"
                       + "".join(f"<th>{y}</th>" for y in totals.index)
                       + "</tr></thead><tbody><tr><th scope='row'>Headlines</th>"
                       + "".join(f"<td>{int(v):,}</td>" for v in totals.to_numpy()) + "</tr></tbody></table></div>")

    runs = results.get("out_of_sample_runs", [])
    runs_html = "".join(f"<li>{html.escape(r['ran_at'])}: {html.escape(r['reason'] or 'first run')}</li>"
                        for r in runs) or "<li>none yet</li>"
    params = results["signal_parameters"]

    page = REPORT_TEMPLATE.format(
        css=BASE_CSS, js=CHART_JS,
        verdict=verdict_html(results),
        table=results_table_html(results),
        legend=legend(names),
        equity="".join(sections),
        placebo="".join(placebo_parts),
        corr_rows=corr_rows,
        counts=counts_html,
        runs=runs_html,
        params=html.escape(", ".join(f"{k} = {v}" for k, v in params.items())),
        cost=results["backtest_parameters"]["cost_bps_per_side"],
    )
    out = results_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    (results_dir / "results_table.md").write_text(markdown_table(results) + "\n", encoding="utf-8")
    return out


BASE_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --buy: #2a78d6; --short: #e34948; --idle: #898781;
  --good: #006300; --bad: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --buy: #3987e5; --short: #e66767; --good: #0ca30c; --bad: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --buy: #3987e5; --short: #e66767; --good: #0ca30c; --bad: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
a { color: var(--s1); }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; }
h2 { font-size: 20px; margin: 44px 0 8px; }
h3 { font-size: 15px; margin: 24px 0 4px; color: var(--ink-2); }
.lede, .note { color: var(--ink-2); }
.lede { font-size: 17px; max-width: 680px; }
.note { font-size: 13px; margin: 0 0 8px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 16px 20px; }
.verdict { margin: 0 0 8px; }
.verdict.pass strong { color: var(--good); }
.verdict.fail strong { color: var(--bad); }
.checks { margin: 0; padding-left: 20px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th[scope=row], thead th:first-child, td.text, th.text { text-align: left; }
td.text { white-space: normal; }
thead th { color: var(--ink-2); font-weight: 600; }
th.sub { font-weight: 400; color: var(--muted); }
.key { display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin: 0 8px 3px 0; }
.key.s1, .line.s1 { background: var(--s1); stroke: var(--s1); }
.key.s2, .line.s2 { background: var(--s2); stroke: var(--s2); }
.key.s3, .line.s3 { background: var(--s3); stroke: var(--s3); }
.key.s4, .line.s4 { background: var(--s4); stroke: var(--s4); }
.legend { list-style: none; display: flex; flex-wrap: wrap; gap: 6px 20px; padding: 0; margin: 8px 0 0;
  font-size: 13px; color: var(--ink-2); }
.chart { margin: 0; position: relative; }
.chart svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.label { fill: var(--ink-2); font-size: 12px; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.bar { fill: var(--s1); }
.bar:hover, .bar:focus { opacity: .75; outline: none; }
.marker { stroke: var(--ink); stroke-width: 2; }
.crosshair { stroke: var(--muted); stroke-width: 1; visibility: hidden; }
.hit { fill: transparent; }
.tooltip { position: fixed; pointer-events: none; background: var(--surface); border: 1px solid var(--ring);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12);
  display: none; z-index: 10; min-width: 160px; max-width: 320px; }
.tooltip .date { color: var(--muted); margin-bottom: 4px; }
.tooltip .row { display: flex; align-items: center; gap: 8px; }
.tooltip .row strong { font-variant-numeric: tabular-nums; min-width: 48px; }
.tooltip .row span:last-child { color: var(--ink-2); }
details { margin: 8px 0 0; font-size: 13px; }
summary { cursor: pointer; color: var(--ink-2); }
"""

CHART_JS = """
(() => {
  const tip = document.querySelector('.tooltip');
  const place = (e) => { tip.style.left = Math.max(8, Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8)) + 'px';
                         tip.style.top = (e.clientY + 14) + 'px'; };
  const dollars = (v, large) => '$' + v.toLocaleString(undefined,
      { minimumFractionDigits: large ? 0 : 2, maximumFractionDigits: large ? 0 : 2 });
  document.querySelectorAll('figure.chart').forEach((fig) => {
    const dataTag = fig.querySelector('script[type="application/json"]');
    const svg = fig.querySelector('svg');
    if (dataTag) {
      const d = JSON.parse(dataTag.textContent);
      const hair = svg.querySelector('.crosshair'), hit = svg.querySelector('.hit');
      const n = d.dates.length;
      const show = (e) => {
        const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        const x = pt.matrixTransform(svg.getScreenCTM().inverse()).x;
        const i = Math.max(0, Math.min(n - 1, Math.round((x - d.x0) / (d.x1 - d.x0) * (n - 1))));
        const xi = d.x0 + (d.x1 - d.x0) * i / Math.max(n - 1, 1);
        hair.setAttribute('x1', xi); hair.setAttribute('x2', xi); hair.style.visibility = 'visible';
        tip.replaceChildren();
        const date = document.createElement('div'); date.className = 'date'; date.textContent = d.dates[i];
        tip.appendChild(date);
        d.names.forEach((name, k) => {
          const row = document.createElement('div'); row.className = 'row';
          const key = document.createElement('span'); key.className = 'key s' + (k + 1);
          const val = document.createElement('strong'); val.textContent = dollars(d.values[k][i], d.large);
          const lab = document.createElement('span'); lab.textContent = name;
          row.append(key, val, lab); tip.appendChild(row);
        });
        tip.style.display = 'block'; place(e);
      };
      hit.addEventListener('pointermove', show);
      hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; hair.style.visibility = 'hidden'; });
    }
    svg.querySelectorAll('[data-tip]').forEach((mark) => {
      const on = (e) => { tip.textContent = mark.dataset.tip; tip.style.display = 'block';
        const r = mark.getBoundingClientRect(); place(e.clientX ? e : { clientX: r.right, clientY: r.top }); };
      mark.addEventListener('pointermove', on); mark.addEventListener('focus', on);
      ['pointerleave', 'blur'].forEach((ev) => mark.addEventListener(ev, () => { tip.style.display = 'none'; }));
    });
  });
})();
"""

REPORT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Divergence Backtest Report</title>
<style>{css}</style></head>
<body><main>
<h1>News sentiment vs. price divergence</h1>
<p class="lede">Does a stock catch up when its news turns positive (or negative) but its price goes the other way?
Tested on 30 large US stocks, 2015 to today, with FinBERT-scored headlines.</p>

<h2>Verdict</h2>
<div class="card">{verdict}</div>

<h2>Results: development vs. test period</h2>
<p class="note">Dev = 2015–2021, used while building. Test = 2022–today, run once with settings locked.
All returns after {cost} basis points of cost per trade, each way. Sharpe assumes a 0% risk-free rate.</p>
{table}

<h2>Growth of $1</h2>
{legend}
{equity}

<h2>Placebo test: real news vs. shuffled news</h2>
<p class="note">The same rule was run on sentiment scores dealt out to random days. If the real strategy is not
far to the right of these bars, the timing of the news is not what drove its returns.</p>
{placebo}

<h2>Does today's tone predict tomorrow's return?</h2>
<div class="table-wrap"><table><thead><tr><th>Period</th><th>Scorer</th><th>Correlation</th><th>Observations</th></tr>
</thead><tbody>{corr_rows}</tbody></table></div>
<p class="note">Pooled across stocks. Stocks move together on the same day, so observations are not fully independent.</p>

<h2>Data</h2>
<p class="note">Headlines per year (Benzinga via Alpaca), counted by the trading day each one is assigned to.
A headline published at or after the 4:00 pm close counts toward the next trading day.</p>
{counts}

<h2>Settings and audit trail</h2>
<p class="note">Signal settings: {params}</p>
<p class="note">Every run on the test period is recorded:</p>
<ul class="note">{runs}</ul>

<h2>Limitations</h2>
<ul class="note">
<li>Survivorship bias: only companies that still exist today were tested, which flatters results.</li>
<li>Headline wording is the latest version; stories edited after publication could look more informed than they were.</li>
<li>Trades are assumed to fill at the opening price; real fills can be worse.</li>
<li>Research project, not financial advice.</li>
</ul>
</main>
<div class="tooltip" role="status"></div>
<script>{js}</script>
</body></html>
"""
