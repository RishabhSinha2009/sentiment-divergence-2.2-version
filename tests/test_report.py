"""The HTML report and the pre-registered success criteria.

Results come from the invented market in fake_market.py, not real data.
"""

import re

import pandas as pd
from fake_market import random_market

from divergence import backtest, report
from divergence.backtest import BacktestConfig
from divergence.signals import SignalConfig

SIGNAL = SignalConfig(sentiment_model="test", return_days=5, zscore_window=60,
                      min_history=20, threshold=1.5, holding_days=5)


def results_for(locked: bool) -> dict:
    prices, sentiment = random_market(seed=5)
    config = BacktestConfig(in_sample_end="2020-09-30", cost_bps_per_side=5, placebo_runs=5, locked=locked)
    results = backtest.evaluate(prices, {"test": sentiment}, SIGNAL, config, include_out_of_sample=locked)
    results["out_of_sample_runs"] = []
    return results


def test_verdict_needs_all_three_criteria():
    results = results_for(locked=True)
    rows = {(r["strategy"], r["period"]): r for r in results["table"]}
    rows[("news divergence (test)", "out_of_sample")]["sharpe"] = 1.0
    rows[("price only (no news)", "out_of_sample")]["sharpe"] = 0.5
    results["placebo"]["out_of_sample"]["p_value"] = 0.01
    assert report.verdict(results)[0] is True

    results["placebo"]["out_of_sample"]["p_value"] = 0.20  # no better than shuffled news
    passed, checks = report.verdict(results)
    assert passed is False
    assert [ok for _, ok in checks] == [True, True, False]


def test_unlocked_report_says_not_tested_and_shows_no_test_period(tmp_path):
    backtest.save_results(results_for(locked=False), tmp_path)
    page = report.build(tmp_path, pd.DataFrame()).read_text(encoding="utf-8")
    assert "Not tested yet" in page
    assert "2022–today (test)</h3>" not in page


def test_locked_report_renders_every_section(tmp_path):
    backtest.save_results(results_for(locked=True), tmp_path)
    page = report.build(tmp_path, pd.DataFrame()).read_text(encoding="utf-8")
    for expected in ["Verdict", "Growth of $1", "Placebo test", "news divergence (test)",
                     "price only (no news)", "buy and hold (equal weight)"]:
        assert expected in page
    assert not re.search(r"\bnan\b", page, re.IGNORECASE)  # no unformatted missing values
    assert (tmp_path / "results_table.md").exists()
