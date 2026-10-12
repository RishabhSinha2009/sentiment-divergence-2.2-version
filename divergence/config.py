"""Reads config.yaml into a typed object."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from .backtest import BacktestConfig
from .paper import PaperConfig
from .signals import SignalConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    tickers: list[str]
    news_aliases: dict[str, list[str]]
    start: date
    end: date | None
    database: Path
    news_page_size: int
    signal: SignalConfig | None = None
    backtest: BacktestConfig | None = None
    paper: PaperConfig | None = None
    repo_url: str | None = None

    def news_symbols_for(self, ticker: str) -> list[str]:
        """Every symbol the news provider might use for this ticker."""
        return [ticker, *self.news_aliases.get(ticker, [])]


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    end = raw["dates"]["end"]
    return Config(
        tickers=list(raw["universe"]["tickers"]),
        news_aliases=dict(raw["universe"].get("news_aliases") or {}),
        start=date.fromisoformat(raw["dates"]["start"]),
        end=date.fromisoformat(end) if end else None,
        # Relative paths are resolved from the repo, not from wherever the
        # command happens to be run, so there is only ever one database.
        database=REPO_ROOT / raw["database"],
        news_page_size=int(raw["news"]["page_size"]),
        signal=SignalConfig(**raw["signal"]),
        backtest=BacktestConfig(**raw["backtest"]),
        paper=PaperConfig(**raw["paper"]),
        repo_url=(raw.get("site") or {}).get("repo_url"),
    )
