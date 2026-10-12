"""Builds the public website: site/index.html.

One page that shows, with live numbers, how the bot reads the news and decides
what to trade: today's signals, the headlines it just scored, the paper-trading
account's track record, and whether the idea held up historically.

It is plain HTML with no server behind it, so GitHub Pages can host it for
free. A scheduled GitHub job rebuilds it on weekdays (.github/workflows/site.yml).
Visitors need no account and no keys.
"""

import html
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .report import BASE_CSS, CHART_JS, results_table_html, value_chart, verdict, verdict_html
from .scan import ScanResult
from .trading_calendar import MARKET_TZ


@dataclass
class PaperSnapshot:
    equity: pd.DataFrame       # date, equity
    positions: list[dict]      # as returned by Alpaca
    orders: list[dict]         # most recent first


def collect_paper(broker) -> PaperSnapshot:
    return PaperSnapshot(broker.equity_history(), broker.open_positions(), broker.recent_orders(15))


# --- pieces of the page ----------------------------------------------------------

def latest_rows(result: ScanResult) -> pd.DataFrame:
    latest = result.panel[result.panel["date"] == result.last_session].copy()
    holds = result.positions.set_index("ticker")["position"] if not result.positions.empty else pd.Series(dtype=str)
    latest["holds"] = latest["ticker"].map(holds).fillna("–")
    developing = result.developing.set_index("ticker") if not result.developing.empty else pd.DataFrame()
    latest["since_close"] = latest["ticker"].map(
        lambda t: (f"{int(developing.loc[t, 'headlines'])} · {developing.loc[t, 'provisional']}"
                   if t in developing.index else "–"))
    return latest


def closeness(tone_z: float, return_z: float) -> float:
    """How far a stock has moved into either trading zone. At or above the threshold means a trade."""
    return max(min(tone_z, -return_z), min(-tone_z, return_z))


def todays_example(latest: pd.DataFrame, threshold: float) -> str:
    """One sentence that walks through today's most interesting stock."""
    scored = latest.dropna(subset=["sentiment_z", "return_z"])
    if scored.empty:
        return "No stock had both news and enough price history at the last close."
    scored = scored.assign(closeness=[closeness(t, r) for t, r in zip(scored["sentiment_z"], scored["return_z"])])
    row = scored.sort_values("closeness", ascending=False).iloc[0]
    numbers = f"news tone z {row['sentiment_z']:+.1f}, price z {row['return_z']:+.1f}"
    if row["flag"] == 1:
        return f"<strong>{row['ticker']}</strong> is in the buy zone ({numbers}), so the bot buys it at the next open."
    if row["flag"] == -1:
        return f"<strong>{row['ticker']}</strong> is in the short zone ({numbers}), so the bot shorts it at the next open."
    return (f"Closest call: <strong>{row['ticker']}</strong> ({numbers}). Both numbers have to pass "
            f"±{threshold:g} in opposite directions, so no trade.")


def zone_chart(latest: pd.DataFrame, threshold: float) -> str:
    """Every stock as a dot: news tone (up) against recent price move (across)."""
    w, h, left, right, top, bottom = 720, 440, 56, 16, 16, 48
    plotted = latest.dropna(subset=["sentiment_z", "return_z"])
    biggest = float(np.nanmax(np.abs(plotted[["sentiment_z", "return_z"]].to_numpy()))) if len(plotted) else 0
    lim = float(min(6.0, max(3.5, np.ceil(biggest + 0.5))))
    x_of = lambda v: left + (w - left - right) * (np.clip(v, -lim, lim) + lim) / (2 * lim)
    y_of = lambda v: top + (h - top - bottom) * (1 - (np.clip(v, -lim, lim) + lim) / (2 * lim))

    parts = [
        f"<rect class='zone buy' x='{x_of(-lim):.1f}' y='{y_of(lim):.1f}' width='{x_of(-threshold) - x_of(-lim):.1f}' "
        f"height='{y_of(threshold) - y_of(lim):.1f}'/>",
        f"<rect class='zone short' x='{x_of(threshold):.1f}' y='{y_of(-threshold):.1f}' "
        f"width='{x_of(lim) - x_of(threshold):.1f}' height='{y_of(-lim) - y_of(-threshold):.1f}'/>",
    ]
    for tick in range(-int(lim), int(lim) + 1):
        parts.append(f"<line class='grid' x1='{x_of(tick):.1f}' x2='{x_of(tick):.1f}' y1='{top}' y2='{h - bottom}'/>"
                     f"<line class='grid' x1='{left}' x2='{w - right}' y1='{y_of(tick):.1f}' y2='{y_of(tick):.1f}'/>"
                     f"<text class='tick' x='{x_of(tick):.1f}' y='{h - bottom + 16}' text-anchor='middle'>{tick:+d}</text>"
                     f"<text class='tick' x='{left - 8}' y='{y_of(tick) + 4:.1f}' text-anchor='end'>{tick:+d}</text>")
    parts.append(f"<line class='axis' x1='{x_of(0):.1f}' x2='{x_of(0):.1f}' y1='{top}' y2='{h - bottom}'/>"
                 f"<line class='axis' x1='{left}' x2='{w - right}' y1='{y_of(0):.1f}' y2='{y_of(0):.1f}'/>")
    parts.append(f"<text class='zone-label' x='{x_of(-lim) + 8:.1f}' y='{y_of(lim) + 18:.1f}'>Good news, price lagging → BUY</text>"
                 f"<text class='zone-label' x='{x_of(lim) - 8:.1f}' y='{y_of(-lim) - 10:.1f}' text-anchor='end'>"
                 f"Bad news, price rising → SHORT</text>")
    parts.append(f"<text class='label' x='{(left + w - right) / 2:.1f}' y='{h - 8}' text-anchor='middle'>"
                 f"Price: last 5 days compared with normal (z-score)</text>"
                 f"<text class='label' transform='translate(14 {(top + h - bottom) / 2:.1f}) rotate(-90)' "
                 f"text-anchor='middle'>News tone compared with normal (z-score)</text>")

    for row in plotted.sort_values("flag", key=abs).itertuples(index=False):
        kind = {1: "buy", -1: "short"}.get(int(row.flag), "idle")
        x, y = x_of(row.return_z), y_of(row.sentiment_z)
        tip = html.escape(f"{row.ticker}: news tone z {row.sentiment_z:+.2f} ({row.n_headlines} headlines), "
                          f"5-day return {row.return_recent:+.1%} (z {row.return_z:+.2f})"
                          + {"buy": " — BUY signal", "short": " — SHORT signal"}.get(kind, ""))
        parts.append(f"<circle class='dot {kind}' cx='{x:.1f}' cy='{y:.1f}' r='5'/>"
                     f"<circle class='hit' cx='{x:.1f}' cy='{y:.1f}' r='12' tabindex='0' data-tip='{tip}'/>")
        if kind != "idle":
            parts.append(f"<text class='dot-label' x='{x + 9:.1f}' y='{y + 4:.1f}'>{row.ticker}</text>")
    missing = len(latest) - len(plotted)
    note = (f"<p class='note'>{missing} stock{'s' if missing != 1 else ''} had no headlines that day, "
            f"so there is no tone to plot.</p>") if missing else ""
    return (f"<figure class='chart'><svg viewBox='0 0 {w} {h}' role='img' aria-label='Each stock plotted by "
            f"news tone against recent price move, with buy and short zones'>{''.join(parts)}</svg></figure>{note}")


def signals_table(latest: pd.DataFrame, threshold: float) -> str:
    rows = latest.assign(
        signal=latest["flag"].map({1: "BUY", -1: "SHORT"}).fillna("–"),
        order=[-(abs(f) * 10 + (h != "–") * 5 + (closeness(t, r) if pd.notna(t) and pd.notna(r) else -9))
               for f, h, t, r in zip(latest["flag"], latest["holds"], latest["sentiment_z"], latest["return_z"])],
    ).sort_values("order")
    fmt = lambda v, spec: "–" if pd.isna(v) else format(v, spec)
    body = "".join(
        f"<tr><th scope='row'>{r.ticker}</th><td>{r.n_headlines}</td><td>{fmt(r.sentiment, '+.2f')}</td>"
        f"<td>{fmt(r.sentiment_z, '+.2f')}</td><td>{fmt(r.return_recent, '+.1%')}</td><td>{fmt(r.return_z, '+.2f')}</td>"
        f"<td class='sig {r.signal.lower()}'>{r.signal}</td><td>{r.holds}</td><td>{html.escape(r.since_close)}</td></tr>"
        for r in rows.itertuples(index=False))
    return ("<div class='table-wrap'><table><thead><tr><th>Stock</th><th>Headlines</th><th>Tone</th><th>Tone z</th>"
            "<th>5-day return</th><th>Price z</th><th>Signal</th><th>Bot holds next open</th>"
            f"<th>News since close</th></tr></thead><tbody>{body}</tbody></table></div>")


def tone_bar(score: float) -> str:
    """A small centered bar: right and blue for positive tone, left and red for negative."""
    half = 40 * min(abs(score), 1)
    x = 40 if score >= 0 else 40 - half
    kind = "buy" if score >= 0 else "short"
    return (f"<svg class='tonebar' viewBox='0 0 80 10' aria-hidden='true'><line class='axis' x1='40' x2='40' y1='0' y2='10'/>"
            f"<rect class='{kind}' x='{x:.1f}' y='2' width='{max(half, 1):.1f}' height='6' rx='2'/></svg>")


def headlines_feed(recent: pd.DataFrame, limit: int = 12) -> str:
    if recent.empty:
        return "<p class='note'>No recent headlines.</p>"
    grouped = (recent.groupby("id")
               .agg(published=("published_at_utc", "first"), headline=("headline", "first"),
                    url=("url", "first"), score=("score", "first"), tickers=("ticker", lambda t: ", ".join(sorted(t))))
               .sort_values("published", ascending=False).head(limit))
    rows = []
    for r in grouped.itertuples():
        when = r.published.tz_convert(MARKET_TZ).strftime("%a %I:%M %p").replace(" 0", " ")
        text = html.escape(r.headline)
        link = f"<a href='{html.escape(r.url)}' rel='noopener' target='_blank'>{text}</a>" if isinstance(r.url, str) else text
        rows.append(f"<tr><td class='text muted'>{when}</td><td class='text'>{html.escape(r.tickers)}</td>"
                    f"<td>{r.score:+.2f}</td><td>{tone_bar(r.score)}</td><td class='text'>{link}</td></tr>")
    return ("<div class='table-wrap'><table><thead><tr><th class='text'>Published (ET)</th><th class='text'>Stock</th>"
            "<th>Tone</th><th></th><th class='text'>Headline</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def paper_section(paper: PaperSnapshot | None, tickers: list[str]) -> str:
    if paper is None:
        return "<p class='note'>Paper trading is not connected on this copy of the site.</p>"
    parts = []
    equity = paper.equity
    if len(equity) >= 2:
        series = equity.set_index("date")[["equity"]].rename(columns={"equity": "Paper account"})
        change = equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1
        parts.append(f"<p class='note'>Started {equity['date'].iloc[0]:%b %d, %Y} with "
                     f"${equity['equity'].iloc[0]:,.0f} of fake money. Now ${equity['equity'].iloc[-1]:,.0f} "
                     f"({change:+.2%}).</p>" + value_chart(series, ["Paper account"], "paper", "Paper account value"))
    else:
        parts.append("<p class='note'>The track record starts with the bot's first trading day.</p>")

    mine = [p for p in paper.positions if p["symbol"] in tickers]
    if mine:
        body = "".join(
            f"<tr><th scope='row'>{p['symbol']}</th><td>{p['side'].upper()}</td><td>{abs(int(float(p['qty'])))}</td>"
            f"<td>${abs(float(p['market_value'])):,.0f}</td><td>{float(p['unrealized_plpc']):+.2%}</td></tr>"
            for p in mine)
        parts.append("<h3>Open positions</h3><div class='table-wrap'><table><thead><tr><th>Stock</th><th>Side</th>"
                     f"<th>Shares</th><th>Value</th><th>Unrealized</th></tr></thead><tbody>{body}</tbody></table></div>")
    else:
        parts.append("<h3>Open positions</h3><p class='note'>None right now.</p>")

    orders = [o for o in paper.orders if o["symbol"] in tickers]
    if orders:
        body = "".join(
            f"<tr><td class='text muted'>{pd.Timestamp(o['submitted_at']).tz_convert(MARKET_TZ):%b %d %I:%M %p}</td>"
            f"<th scope='row'>{o['symbol']}</th><td>{o['side'].upper()}</td><td>{o['qty']}</td>"
            f"<td>{html.escape(o['status'])}</td><td>{'$' + format(float(o['filled_avg_price']), ',.2f') if o.get('filled_avg_price') else '–'}</td></tr>"
            for o in orders)
        parts.append("<h3>Recent orders</h3><div class='table-wrap'><table><thead><tr><th class='text'>Sent (ET)</th>"
                     f"<th>Stock</th><th>Side</th><th>Shares</th><th>Status</th><th>Fill price</th></tr></thead>"
                     f"<tbody>{body}</tbody></table></div>")
    return "".join(parts)


def tile(label: str, value: str, note: str = "") -> str:
    return (f"<div class='tile'><div class='tile-label'>{label}</div><div class='tile-value'>{value}</div>"
            f"<div class='tile-note'>{note}</div></div>")


# --- the page -----------------------------------------------------------------

def build(out_dir: Path, result: ScanResult, recent: pd.DataFrame, config: Config, results: dict | None,
          paper: PaperSnapshot | None, now: pd.Timestamp, report_file: Path | None = None) -> Path:
    threshold = config.signal.threshold
    latest = latest_rows(result)
    flagged = int((latest["flag"] != 0).sum())
    longs = int((result.positions["position"] == "LONG").sum()) if not result.positions.empty else 0
    shorts = len(result.positions) - longs
    passed = verdict(results)[0] if results else None

    paper_value = "not connected"
    paper_note = ""
    if paper is not None and len(paper.equity):
        paper_value = f"${paper.equity['equity'].iloc[-1]:,.0f}"
        paper_note = f"{paper.equity['equity'].iloc[-1] / paper.equity['equity'].iloc[0] - 1:+.2%} since start"
    tiles = "".join([
        tile("Diverging at the last close", str(flagged), f"of {len(latest)} stocks"),
        tile("Bot positions from next open", str(longs + shorts), f"{longs} long · {shorts} short"),
        tile("Paper account (fake money)", paper_value, paper_note),
        tile("Historical test", {True: "Held up", False: "Did not hold up", None: "Pending"}[passed],
             "2022–today, out-of-sample"),
    ])

    n_today = int(latest["n_headlines"].sum())
    out_dir.mkdir(parents=True, exist_ok=True)
    backtest_html = verdict_html(results)
    if results:
        backtest_html += results_table_html(results, [("annual_return", "Annual return"), ("sharpe", "Sharpe"),
                                                      ("max_drawdown", "Max drawdown")])
    if report_file is not None:
        shutil.copyfile(report_file, out_dir / "report.html")
        backtest_html += "<p><a href='report.html'>Full report: equity curves, placebo test, data coverage →</a></p>"

    page = SITE_TEMPLATE.format(
        css=BASE_CSS + SITE_CSS, js=CHART_JS,
        updated=now.tz_convert(MARKET_TZ).strftime("%a %b %d, %Y, %I:%M %p ET").replace(" 0", " "),
        last_session=f"{result.last_session:%a %b %d}", next_session=f"{result.next_session:%a %b %d}",
        tiles=tiles, n_today=f"{n_today:,}", threshold=f"{threshold:g}",
        window=config.signal.zscore_window, holding=config.signal.holding_days,
        example=todays_example(latest, threshold),
        zone_chart=zone_chart(latest, threshold),
        signals=signals_table(latest, threshold),
        headlines=headlines_feed(recent),
        paper=paper_section(paper, config.tickers),
        backtest=backtest_html,
        repo=html.escape(config.repo_url or "#"),
    )
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


SITE_CSS = """
header.top { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap;
  font-size: 13px; color: var(--muted); margin-bottom: 20px; }
header.top strong { color: var(--ink); font-size: 14px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 24px 0 0; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 14px 16px; }
.tile-label { font-size: 13px; color: var(--ink-2); }
.tile-value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile-note { font-size: 12px; color: var(--muted); min-height: 1em; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; padding: 0;
  list-style: none; counter-reset: step; margin: 12px 0 16px; }
.steps li { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 14px 16px;
  font-size: 14px; color: var(--ink-2); counter-increment: step; }
.steps li::before { content: counter(step); display: inline-grid; place-items: center; width: 22px; height: 22px;
  border-radius: 50%; background: var(--s1); color: #fff; font-size: 12px; font-weight: 600; margin-right: 8px; }
.steps strong { color: var(--ink); }
.example { border-left: 3px solid var(--s1); padding: 8px 14px; margin: 0 0 12px; background: var(--surface); border-radius: 0 8px 8px 0; }
.zone.buy { fill: var(--buy); opacity: .08; }
.zone.short { fill: var(--short); opacity: .08; }
.zone-label { fill: var(--ink-2); font-size: 12px; font-weight: 600; }
.dot { stroke: var(--surface); stroke-width: 2; }
.dot.idle { fill: var(--idle); }
.dot.buy, .tonebar .buy { fill: var(--buy); }
.dot.short, .tonebar .short { fill: var(--short); }
.dot-label { fill: var(--ink); font-size: 12px; font-weight: 600; }
circle.hit { fill: transparent; cursor: default; outline: none; }
.sig.buy, .sig.short { font-weight: 600; }
.tonebar { width: 80px; height: 10px; vertical-align: middle; }
.muted { color: var(--muted); }
.safeguards { padding-left: 20px; color: var(--ink-2); font-size: 14px; }
.safeguards li { margin: 4px 0; }
footer { margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--grid); font-size: 13px; color: var(--muted); }
"""

SITE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Divergence Bot</title>
<meta name="description" content="A research bot that reads financial news, scores its tone with AI, and trades when news and price disagree.">
<style>{css}</style></head>
<body><main>
<header class="top"><strong>Divergence Bot</strong><span>Updated {updated} · refreshes every 30 minutes on market days</span></header>

<h1>When the news and the stock price disagree</h1>
<p class="lede">This bot reads every headline about 30 large US companies, scores its tone with FinBERT
(an AI model trained on financial news), and looks for days when the news and the price point in opposite
directions. The research question: does the price catch up with the news?</p>
<div class="tiles">{tiles}</div>

<h2>How the bot decides</h2>
<ol class="steps">
<li><strong>Read the news.</strong> {n_today} headlines about these companies counted toward the last close ({last_session}).
News after 4:00 pm counts toward the next trading day, so the bot never uses information it couldn't have had.</li>
<li><strong>Score the tone.</strong> FinBERT rates each headline from −1 (very negative) to +1 (very positive).
Each company's tone for the day is the average.</li>
<li><strong>Compare with normal.</strong> Today's tone and the last 5 days' price move are each compared with the
previous {window} trading days for that same company (a z-score: +2 means unusually high).</li>
<li><strong>Act on disagreement.</strong> Tone z above +{threshold} while price z is below −{threshold}: buy at the next
open and hold {holding} days. The mirror image: short.</li>
</ol>
<p class="example">{example}</p>
{zone_chart}
<p class="note">Each dot is one company at the last close ({last_session}). Hover a dot for its numbers.
Dots in a shaded corner are trades for the next open ({next_session}).</p>

<h2>Today's signals</h2>
{signals}
<p class="note">"News since close" counts headlines published after the last close and shows where their tone is
heading. That reading is provisional until the next close.</p>

<h2>Latest headlines the bot read</h2>
{headlines}

<h2>The bot's paper-trading account</h2>
<p class="note">Every weekday morning before the open, the bot places its trades in an Alpaca paper account
(simulated money, real prices). This is a forward test: these are days the strategy had never seen.</p>
{paper}

<h2>Did the idea work historically?</h2>
<p class="note">Built and tuned on 2015–2021 only; then the settings were locked and it was run once on
2022–today. Returns are after trading costs.</p>
<div class="card">{backtest}</div>

<h2>Why the test can be trusted</h2>
<ul class="safeguards">
<li>A headline only counts once it was public before the closing bell. Tested on 200,000 random timestamps and every real headline.</li>
<li>Each "normal" is computed from earlier days only. A test rewrites the future and checks no past signal changes.</li>
<li>The 2022–today test period stayed locked until the settings were frozen, and every run on it is logged.</li>
<li>Success was defined in advance: profitable after costs, better than the same rule without news, and better than shuffled news.</li>
<li>Compared against buy-and-hold, a price-only version, and 100 placebo runs with shuffled sentiment.</li>
</ul>

<footer>Built by Rishabh Sinha · <a href="{repo}">Source code on GitHub</a> · MIT license ·
News: Benzinga via Alpaca · Prices: Yahoo Finance · Research project, not financial advice.</footer>
</main>
<div class="tooltip" role="status"></div>
<script>{js}</script>
</body></html>
"""
