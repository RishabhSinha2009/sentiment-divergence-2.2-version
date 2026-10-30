# News Sentiment vs. Price Divergence

**When a company's news turns positive but its stock keeps falling (or the reverse),
does the price catch up with the news?**

This project tests that question on 30 large US stocks from 2015 to today. It gathers
every daily price and 171,972 timestamped news headlines about those stocks, and scores
each headline's tone with FinBERT, an AI model trained on financial text. It flags the
days when news and price disagree, then tests a trading rule built on those flags. The
test is designed to tell a real pattern apart from one that only fits the past.

It also runs on today's news: a daily scan, a live mode that scores each headline
within about a second of publication, and a paper-trading bot (fake money).

**Live site: [sentiment-divergence-rouge.vercel.app](https://sentiment-divergence-rouge.vercel.app/).**
It shows today's signals, a step-by-step view of how the bot decides, the headlines it just read,
and its paper-trading track record. It updates itself every 30 minutes on market days, and you
don't need an account to see it.

> **This is a research project, not financial advice.** It tests whether an idea held
> up in past data. Past results, good or bad, do not predict future ones.

## Results

<!-- RESULTS:START -->
**Verdict: the news signal did not hold up.** Before the test was run, success was
defined as passing all three of these checks on the 2022–today test period:

| Success check (2022–today) | Result |
|---|---|
| Sharpe ratio above 0 after costs | ✔ 0.27 |
| Beats the same rule with the news removed | ✔ 0.27 vs. −0.68 |
| Beats shuffled news (placebo p-value below 0.05) | ✘ p = 0.059 |

| Strategy | Period | Annual return | Sharpe | Max drawdown | Hit rate | Trades |
|---|---|---|---|---|---|---|
| news divergence (FinBERT) | 2015–2021 (development) | −2.9% | −0.09 | −34.8% | 46% | 259 |
| news divergence (FinBERT) | 2022–today (test) | +3.4% | 0.27 | −45.1% | 55% | 225 |
| news divergence (VADER) | 2015–2021 (development) | −2.1% | −0.02 | −41.5% | 52% | 236 |
| news divergence (VADER) | 2022–today (test) | +8.1% | 0.50 | −30.1% | 50% | 223 |
| price only (no news) | 2015–2021 (development) | −7.5% | −0.44 | −57.9% | 52% | 2,730 |
| price only (no news) | 2022–today (test) | −10.4% | −0.68 | −47.4% | 51% | 1,958 |
| buy and hold (equal weight) | 2015–2021 (development) | +16.7% | 1.02 | −28.3% | – | – |
| buy and hold (equal weight) | 2022–today (test) | +11.5% | 0.84 | −20.3% | – | – |

*All returns are after 5 basis points of cost per trade, each way. The test period ran
through 2026-10-09.*

**In plain terms:**
- **2015–2021:** the strategy lost money. Even before trading costs it made about 0% a year
  (Sharpe +0.08), so the signal had no edge to lose.
- **2022–today:** it made +3.4% a year, but trailed simply holding the same stocks (+11.5%) and
  had a 45% worst drawdown. Its lead over shuffled news (p = 0.059) misses the bar set in advance.
  Two scorers were tested, which makes a near-miss even less convincing.
- **Tone vs. next-day return:** essentially no relationship (correlation −0.011 in 2015–2021,
  +0.005 in 2022–today, across about 58,000 stock-days).
- **VADER vs. FinBERT:** the simple word-list scorer did *better* than FinBERT out-of-sample.
  When the cruder tool wins, the differences are most likely noise.
- **A likely explanation (not tested here):** news about the most-watched US companies is
  absorbed into the price within minutes. A daily strategy that trades at the next morning's
  open arrives after the move.

Full report with equity curves, the placebo test and data coverage: [`results/report.html`](results/report.html).
The settings were never tuned, and no filters were added after seeing these results.
<!-- RESULTS:END -->

## How it works

```
 prices (Yahoo)  ─┐
                  ├─▶ timing rule ─▶ daily tone per stock ─▶ z-scores ─▶ divergence flags ─▶ backtest
 headlines        │   (which trading    (FinBERT, VADER)      (vs. each      (news vs. price)    (2015–2021 build,
 (Alpaca/Benzinga)┘    day each one                           stock's own                         2022–today test)
                       counts toward)                         history)
                                                                                    └─▶ scan · live · paper trading
```

1. **Data.** Daily prices from Yahoo Finance; headlines with exact publication times
   from Alpaca's news API (Benzinga's feed). Everything is cached in a local SQLite
   database, so nothing is downloaded twice.
2. **Timing rule.** A headline counts toward the first trading day whose closing bell
   rings *after* it was published. News at 3:59 pm counts today; news at 4:00 pm or
   later, on weekends, or on holidays counts toward the next trading day. Early-close
   days (1:00 pm) use the real close.
3. **Tone.** Each headline is scored from −1 (very negative) to +1 (very positive) by
   FinBERT, and by VADER, a simple word list, as a baseline. A stock's daily tone is the
   average of that day's headlines.
4. **Signal.** Today's tone and the last 5 days' return are each turned into a
   z-score: how unusual they are *for this stock*, compared with its previous 60
   trading days. A **bullish divergence** is tone z ≥ +1.5 while return z ≤ −1.5 (good
   news, lagging price). The rule buys at the next morning's open and holds 5 days.
   Bearish is the mirror image (short).
5. **Backtest.** Trades at the next day's opening price, 0.05% cost per trade each
   way, compared against three alternatives (below).

## How the test is kept honest

The easiest way to fool yourself in a backtest is to let information from the future
leak in, or to keep adjusting until the past looks good. Every part of this project
is built to prevent both.

| Safeguard | What it does |
|---|---|
| **Timing rule, tested** | Tests check 18 edge cases (closing bell, weekends, holidays, early closes, daylight saving) plus 200,000 random timestamps. They also check every real headline in the database. |
| **Past-only z-scores** | "Normal" is always computed from earlier days. A test rewrites all data after a chosen day and confirms no signal on or before that day changes. |
| **Planted-bug checks** | Look-ahead bugs were deliberately inserted (using next week's return, trading on the same day as a signal) to confirm the tests catch them. They caught every one. |
| **Locked test period** | 2015–2021 was used to build the strategy. 2022–today stays locked, and its results are not even computed, until the settings are frozen. Every run on the test period is logged in the database and listed in the report. Re-running it with changed settings is refused unless a written reason is given. |
| **Success defined in advance** | Before the test was run, success was defined as all three of: Sharpe > 0 after costs, beating the same rule with the news removed, and beating shuffled news (p < 0.05). |
| **Three comparisons** | **Price-only** (same rule, no news): does the news add anything? **Placebo** (same rule fed sentiment shuffled across days, 100 times): does real news beat random noise? **Buy-and-hold** of the same 30 stocks. |
| **No tuning, no rescue filters** | The settings chosen at the start were never adjusted, even though the development results were weak. The weak result is reported as weak. |

## Running it yourself

You need Python 3.11+ and a free [Alpaca](https://alpaca.markets) account for the news
data. A paper-trading account is enough; no money or bank link is needed. Each person
uses their own keys.

```bash
git clone <this repo> && cd sentiment-divergence
python -m venv .venv
.venv/Scripts/activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # adds the `divergence` command
cp .env.example .env              # then paste your Alpaca key ID and secret into .env
```

**Build the dataset** (first run: about 1 hour each for `news` and `score`):

```bash
divergence prices      # daily prices, 2015 to the last close
divergence news        # every headline for the 30 stocks
divergence score       # FinBERT and VADER tone for each headline
divergence coverage    # data-quality report: gaps, big moves, headlines per year
divergence sample      # 20 random headlines with their scores, for a sanity check
```

**Test the idea:**

```bash
divergence backtest    # prints the results table; saves results/
divergence report      # builds results/report.html (equity curves, placebo test, ...)
pytest                 # the full test suite
```

**Use it today:**

```bash
divergence scan          # stocks diverging at the last close, plus news since then
divergence live          # live news stream: scores each headline as it arrives
divergence paper-trade   # places the rule's trades in an Alpaca PAPER account
divergence paper-status  # paper account value; builds a forward-test track record
divergence site          # builds the public website in site/
```

### About live mode

`divergence live` connects to Alpaca's real-time news stream. Each new headline about
one of the 30 stocks is scored by FinBERT, which takes a fraction of a second on a
laptop CPU. That stock's divergence status is then recomputed, and an alert prints
when it starts or stops diverging. Final signals are still only set at the close,
exactly as in the backtest. Live mode shows which way things are heading before then.
If the connection drops, it reconnects and back-fills anything it missed.

### About paper trading

`divergence paper-trade` is a *forward test*: the rule trades on days that did not
exist when it was designed. Safety rails are built into the code, not settings:

- Only Alpaca's **paper** (fake-money) address exists in the code; a test checks that
  no real-money address appears anywhere.
- It prints every order and sends nothing until you type `yes`.
- Orders are market-on-open, matching the backtest. Alpaca accepts these only between
  7:00 pm and 9:28 am New York time.
- It only touches the 30 stocks in `config.yaml`.

### About the website

`divergence site` builds `site/index.html`, a single page with no server behind it.
A scheduled GitHub Actions job (`.github/workflows/site.yml`) runs on weekdays:

- **Before the open:** it places the bot's paper trades.
- **Every 30 minutes through the trading day:** it rebuilds the page with fresh news, scores and prices.

The page is published on Vercel. The job works from the last 150 days of data,
which is enough for the 60-day comparisons, so it never needs the 10-year history.

**Run your own copy:**
1. Fork the repo.
2. Create a free [Vercel](https://vercel.com) account and an access token
   (**Account Settings → Tokens**).
3. In your fork's **Settings → Secrets and variables → Actions**, add three secrets:
   `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` (your own free Alpaca paper keys),
   and `VERCEL_TOKEN`.
4. In the **Actions** tab, run the "website" workflow once. It creates the Vercel project
   and publishes the page. After that it runs on its own.

## Data

| Data | Source | Notes |
|---|---|---|
| Daily open/high/low/close/volume | Yahoo Finance via `yfinance` | Split- and dividend-adjusted. If Yahoo re-adjusts history, the stock is re-downloaded in full instead of mixing old and new adjustments. |
| Headlines | Alpaca News API (Benzinga) | 2015 to today, timestamped to the second, tagged by ticker. Uses each story's *original* publication time, not its last-edited time. |

Free sources checked and rejected: the Kaggle "Daily Financial News for 6000+ Stocks"
dataset (ends 2020) and FNSPID (ends 2023). Neither covers the 2022–today test period.

**The 30 stocks** (in `config.yaml`) were chosen by one rule: among the largest S&P 500
companies at the *start* of 2015, keep those still listed under the same ticker
today. Choosing by size in 2015, not today, avoids picking stocks because they later
did well.

## Limitations

- **Survivorship bias.** Only companies that survived to today are included.
  Companies that were large in 2015 and later collapsed, were acquired, or changed
  tickers are missing, so results are optimistic.
- **Headline edits.** Stories keep their original timestamp, but the provider returns
  their latest wording.
- **Scoring and tagging errors.** A spot check of 20 random headlines found FinBERT far
  better than VADER but not perfect: it missed some praise and misread one bullish market
  headline. Benzinga's ticker tags are occasionally wrong (a story about the Amazon
  rainforest was tagged as Amazon the company). Both add noise.
- **Concentrated positions.** On days it holds anything, the rule usually holds a single
  stock (median 1; one to three stocks on 96% of those days). That is why its drawdowns are large.
- **Uneven coverage.** Apple gets hundreds of headlines a month, Coca-Cola about 15.
  Quieter stocks rarely produce signals.
- **Fills at the open are assumed.** Real fills can be worse, especially on volatile mornings.
- **Sharpe ratios assume a 0% risk-free rate,** which flatters every strategy in
  2023–2025, buy-and-hold included.
- **The correlation is pooled across stocks.** Stocks move together, so observations
  are not fully independent.
- **Yahoo Finance is unofficial** and can change without warning.

## Project layout

```
config.yaml              every setting: stocks, dates, signal and backtest parameters
divergence/
  trading_calendar.py    NYSE calendar + the headline timing rule
  prices.py, news.py     data download and caching
  sentiment.py           FinBERT and VADER scoring, daily tone per stock
  signals.py             z-scores, divergence flags, positions
  backtest.py            returns, costs, comparisons, statistics, test-period lock
  report.py              the HTML report and the success criteria
  scan.py, live.py       today's signals, and the real-time news stream
  paper.py               paper trading (fake money only)
  website.py             the public website
  db.py                  SQLite storage
tests/                   automated tests (see "Test the idea" above)
results/                 backtest output and report.html
.github/workflows/       tests on every push; the scheduled website job
```

## License

MIT, © 2026 Rishabh Sinha. See [LICENSE](LICENSE).
