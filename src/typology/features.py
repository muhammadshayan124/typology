"""Graph-aware feature engineering. A plain tabular model (amount, timestamp) misses the
actual structure that makes laundering distinctive -- fan-out/fan-in patterns, unusually
fast money velocity through an account, first-time counterparties. These features encode
that structure directly.

Deliberate scope limitation, stated plainly rather than glossed over: features here are
computed over the *whole* dataset's graph, not a point-in-time snapshot as of each
transaction. A real production system would need genuinely temporal/streaming graph
features (what did this account's graph look like *before* this transaction, not
including it or later ones) to avoid leaking future information. This is a real
simplification made for a portfolio-scoped project, not something to represent as
production-ready. See README.
"""

import networkx as nx
import numpy as np
import pandas as pd

from typology.generator import REPORTING_THRESHOLD


def _build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for row in df.itertuples():
        g.add_edge(row.src, row.dst, amount=row.amount, timestamp=row.timestamp)
    return g


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    g = _build_graph(df)

    out_degree = dict(g.out_degree())
    in_degree = dict(g.in_degree())
    pagerank = nx.pagerank(g, weight="amount")

    sent_total = df.groupby("src")["amount"].sum().to_dict()
    received_total = df.groupby("dst")["amount"].sum().to_dict()
    sent_count = df.groupby("src")["amount"].count().to_dict()
    src_amount_mean = df.groupby("src")["amount"].mean().to_dict()
    src_amount_std = df.groupby("src")["amount"].std().fillna(0).to_dict()

    # Known (src, dst) pairs seen anywhere in the data, for a same-day "seen before up to
    # this point" check -- approximated dataset-wide per the module docstring's caveat.
    pair_first_seen = df.groupby(["src", "dst"])["timestamp"].min().to_dict()

    # Per-account sorted timestamp arrays + binary search, not a full-DataFrame scan per
    # row -- the naive `df[df.src == account & ...]` per transaction is O(n^2) and doesn't
    # finish in reasonable time past a few thousand rows. Caught by actually running this
    # against a realistic-sized dataset, not just eyeballing the logic.
    src_timestamps: dict[int, np.ndarray] = {
        src: np.sort(group["timestamp"].to_numpy()) for src, group in df.groupby("src")
    }

    def tx_velocity(account: int, ts: float, window: float = 1.0) -> int:
        ts_array = src_timestamps.get(account)
        if ts_array is None:
            return 0
        lo = np.searchsorted(ts_array, ts - window, side="left")
        hi = np.searchsorted(ts_array, ts + window, side="right")
        return int(hi - lo)

    rows = []
    for row in df.itertuples():
        amt = row.amount
        src, dst, ts = row.src, row.dst, row.timestamp
        src_mean = src_amount_mean.get(src, amt)
        src_std = src_amount_std.get(src, 0.0)
        rows.append(
            {
                "tx_id": row.tx_id,
                "amount": amt,
                "amount_to_threshold_ratio": amt / REPORTING_THRESHOLD,
                "src_out_degree": out_degree.get(src, 0),
                "src_in_degree": in_degree.get(src, 0),
                "dst_out_degree": out_degree.get(dst, 0),
                "dst_in_degree": in_degree.get(dst, 0),
                "src_pagerank": pagerank.get(src, 0.0),
                "dst_pagerank": pagerank.get(dst, 0.0),
                "src_total_sent": sent_total.get(src, 0.0),
                "dst_total_received": received_total.get(dst, 0.0),
                "src_tx_count": sent_count.get(src, 0),
                "amount_zscore_vs_src_history": (amt - src_mean) / src_std if src_std > 0 else 0.0,
                "src_velocity_1day": tx_velocity(src, ts, window=1.0),
                "is_first_time_pair": int(pair_first_seen.get((src, dst)) == ts),
                "near_threshold": int(0.7 * REPORTING_THRESHOLD <= amt < REPORTING_THRESHOLD),
            }
        )

    features = pd.DataFrame(rows).set_index("tx_id")
    return features
