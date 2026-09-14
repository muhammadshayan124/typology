"""Synthetic transaction-network generator.

Not the Kaggle IBM-AML or PaySim datasets -- both require an account to download (IBM's
own direct-download link is dead, confirmed: 404 as of this build), so this generates a
comparable dataset locally instead, seeded and fully reproducible. See README's "About
the data" section before treating results here as directly comparable to published
IBM/PaySim benchmark numbers -- they aren't the same data, even though the *approach*
(a multi-agent synthetic transaction world with injected laundering patterns) is the same
one IBM's own AMLworld paper uses.

The laundering patterns injected here are real, documented typologies (FATF -- the
Financial Action Task Force -- publishes these), not invented:
- Fan-out ("smurfing"): one account splits a large sum into many small transfers to many
  destinations in a short window, to stay under per-transaction scrutiny.
- Fan-in: the mirror image -- many small transfers converge on one account.
- Layering chain: money moves quickly through a chain of intermediary accounts
  (A -> B -> C -> D...) to obscure its origin.
- Structuring: a sum is split into multiple transfers each just under a reporting
  threshold, spread across several days.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REPORTING_THRESHOLD = 10_000.0


@dataclass
class GeneratorConfig:
    n_accounts: int = 2000
    n_legitimate_tx: int = 50_000
    n_days: int = 90
    n_fan_out: int = 15
    n_fan_in: int = 15
    n_layering_chains: int = 15
    n_structuring: int = 15
    seed: int = 42


@dataclass
class _TxBuilder:
    rows: list = field(default_factory=list)
    next_id: int = 0

    def add(self, ts: float, src: int, dst: int, amount: float, is_laundering: int, typology: str) -> None:
        self.rows.append(
            {
                "tx_id": self.next_id,
                "timestamp": ts,
                "src": src,
                "dst": dst,
                "amount": round(amount, 2),
                "is_laundering": is_laundering,
                "typology": typology,  # ground-truth metadata only -- never a model feature, see model.py
            }
        )
        self.next_id += 1


def _legitimate_transactions(rng: np.random.Generator, cfg: GeneratorConfig, builder: _TxBuilder) -> None:
    # Log-normal amounts: a small number of large transactions, many small ones --
    # matches the real shape of transaction-amount distributions far better than uniform
    # or normal would.
    srcs = rng.integers(0, cfg.n_accounts, cfg.n_legitimate_tx)
    dsts = rng.integers(0, cfg.n_accounts, cfg.n_legitimate_tx)
    amounts = rng.lognormal(mean=4.5, sigma=1.2, size=cfg.n_legitimate_tx)
    timestamps = rng.uniform(0, cfg.n_days, cfg.n_legitimate_tx)
    for ts, s, d, amt in zip(timestamps, srcs, dsts, amounts):
        if s == d:
            continue
        builder.add(ts, int(s), int(d), float(amt), 0, "legitimate")


def _inject_fan_out(rng: np.random.Generator, cfg: GeneratorConfig, builder: _TxBuilder) -> None:
    for _ in range(cfg.n_fan_out):
        source = int(rng.integers(0, cfg.n_accounts))
        k = int(rng.integers(5, 12))
        destinations = rng.choice(cfg.n_accounts, size=k, replace=False)
        total = rng.uniform(20_000, 80_000)
        start = rng.uniform(0, cfg.n_days - 1)
        shares = rng.dirichlet(np.ones(k)) * total
        for i, dst in enumerate(destinations):
            ts = start + rng.uniform(0, 0.5)  # all within half a day
            builder.add(ts, source, int(dst), float(shares[i]), 1, "fan_out")


def _inject_fan_in(rng: np.random.Generator, cfg: GeneratorConfig, builder: _TxBuilder) -> None:
    for _ in range(cfg.n_fan_in):
        dest = int(rng.integers(0, cfg.n_accounts))
        k = int(rng.integers(5, 12))
        sources = rng.choice(cfg.n_accounts, size=k, replace=False)
        total = rng.uniform(20_000, 80_000)
        start = rng.uniform(0, cfg.n_days - 1)
        shares = rng.dirichlet(np.ones(k)) * total
        for i, src in enumerate(sources):
            ts = start + rng.uniform(0, 0.5)
            builder.add(ts, int(src), dest, float(shares[i]), 1, "fan_in")


def _inject_layering_chain(rng: np.random.Generator, cfg: GeneratorConfig, builder: _TxBuilder) -> None:
    for _ in range(cfg.n_layering_chains):
        chain_len = int(rng.integers(4, 8))
        chain = rng.choice(cfg.n_accounts, size=chain_len, replace=False)
        amount = rng.uniform(15_000, 60_000)
        start = rng.uniform(0, cfg.n_days - 1)
        ts = start
        for i in range(chain_len - 1):
            fee = rng.uniform(0.0, 0.02)  # a small cut skimmed at each hop, realistic
            amount *= (1 - fee)
            ts += rng.uniform(0.01, 0.3)  # fast succession -- hours, not days
            builder.add(ts, int(chain[i]), int(chain[i + 1]), float(amount), 1, "layering")


def _inject_structuring(rng: np.random.Generator, cfg: GeneratorConfig, builder: _TxBuilder) -> None:
    for _ in range(cfg.n_structuring):
        src = int(rng.integers(0, cfg.n_accounts))
        dst = int(rng.integers(0, cfg.n_accounts))
        if src == dst:
            dst = (dst + 1) % cfg.n_accounts
        total = rng.uniform(30_000, 90_000)
        remaining = total
        start = rng.uniform(0, cfg.n_days - 4)
        day = 0.0
        while remaining > 0:
            amt = min(remaining, REPORTING_THRESHOLD * rng.uniform(0.7, 0.97))
            builder.add(start + day, src, dst, float(amt), 1, "structuring")
            remaining -= amt
            day += rng.uniform(0.8, 1.5)  # spread across separate days, on purpose


def generate(cfg: GeneratorConfig | None = None) -> pd.DataFrame:
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    builder = _TxBuilder()

    _legitimate_transactions(rng, cfg, builder)
    _inject_fan_out(rng, cfg, builder)
    _inject_fan_in(rng, cfg, builder)
    _inject_layering_chain(rng, cfg, builder)
    _inject_structuring(rng, cfg, builder)

    df = pd.DataFrame(builder.rows).sort_values("timestamp").reset_index(drop=True)
    return df
