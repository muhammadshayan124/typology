"""Graph-aware feature engineering. A plain tabular model (amount, timestamp) misses the
actual structure that makes laundering distinctive -- fan-out/fan-in patterns, unusually
fast money velocity through an account, first-time counterparties. These features encode
that structure directly.

Two builders, on purpose:

- `build_features(df)` is **point-in-time** (the default): every feature of a transaction
  is computed only from transactions that happened strictly before it (plus the
  transaction itself where noted). This is how a deployed monitor would see the world.
- `build_features(df, point_in_time=False)` is the original **dataset-wide** version, kept
  only so the leakage can be measured. It computes degrees, totals, PageRank and
  velocity windows over the whole dataset, so a transaction's features include its own
  future. Numbers from it are optimistic; `typology leakage` quantifies by how much.

PageRank is the one feature that cannot be updated per transaction cheaply, so the
point-in-time version uses a daily snapshot: a transaction sees PageRank computed on the
graph as of the start of its day. That is stale by up to a day, never forward-looking.
"""

from bisect import bisect_left
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from typology.generator import REPORTING_THRESHOLD


def _build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for row in df.itertuples():
        g.add_edge(row.src, row.dst, amount=row.amount, timestamp=row.timestamp)
    return g


def build_features(df: pd.DataFrame, point_in_time: bool = True) -> pd.DataFrame:
    """Features indexed by tx_id, plus a `timestamp` column (not a model feature; used for
    chronological splitting)."""
    if point_in_time:
        return _build_point_in_time(df)
    return _build_dataset_wide(df)


def _build_dataset_wide(df: pd.DataFrame) -> pd.DataFrame:
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
                "timestamp": ts,
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


def _build_point_in_time(df: pd.DataFrame) -> pd.DataFrame:
    """Single pass over transactions in time order, keeping running per-account state.

    Row k's features read state built from rows 0..k-1 only; the row is added to the state
    after its features are written. `tests/test_features.py` checks this property directly
    by editing the future and asserting past features do not move.
    """
    ordered = df.sort_values(["timestamp", "tx_id"], kind="stable")

    out_to: dict[int, set[int]] = defaultdict(set)  # distinct counterparties paid
    in_from: dict[int, set[int]] = defaultdict(set)  # distinct counterparties received from
    sent_total: dict[int, float] = defaultdict(float)
    received_total: dict[int, float] = defaultdict(float)
    # Welford running mean / M2 of each sender's amounts.
    n_sent: dict[int, int] = defaultdict(int)
    mean_sent: dict[int, float] = defaultdict(float)
    m2_sent: dict[int, float] = defaultdict(float)
    sent_times: dict[int, list[float]] = defaultdict(list)  # appended in time order -> sorted
    pairs: set[tuple[int, int]] = set()

    graph = nx.DiGraph()
    pagerank: dict[int, float] = {}
    snapshot_day = -1

    rows = []
    for row in ordered.itertuples():
        src, dst, ts, amt = row.src, row.dst, row.timestamp, row.amount

        day = int(ts)
        if day != snapshot_day:
            pagerank = nx.pagerank(graph, weight="amount") if graph.number_of_edges() else {}
            snapshot_day = day

        n = n_sent[src]
        std = (m2_sent[src] / (n - 1)) ** 0.5 if n > 1 else 0.0
        times = sent_times[src]
        in_window = len(times) - bisect_left(times, ts - 1.0)

        rows.append(
            {
                "tx_id": row.tx_id,
                "timestamp": ts,
                "amount": amt,
                "amount_to_threshold_ratio": amt / REPORTING_THRESHOLD,
                "src_out_degree": len(out_to[src]),
                "src_in_degree": len(in_from[src]),
                "dst_out_degree": len(out_to[dst]),
                "dst_in_degree": len(in_from[dst]),
                "src_pagerank": pagerank.get(src, 0.0),
                "dst_pagerank": pagerank.get(dst, 0.0),
                "src_total_sent": sent_total[src],
                "dst_total_received": received_total[dst],
                "src_tx_count": n,
                "amount_zscore_vs_src_history": (amt - mean_sent[src]) / std if std > 0 else 0.0,
                "src_velocity_1day": in_window + 1,  # earlier sends in the window, plus this one
                "is_first_time_pair": int((src, dst) not in pairs),
                "near_threshold": int(0.7 * REPORTING_THRESHOLD <= amt < REPORTING_THRESHOLD),
            }
        )

        # State update, after the row's features are written.
        out_to[src].add(dst)
        in_from[dst].add(src)
        sent_total[src] += amt
        received_total[dst] += amt
        n_sent[src] = n + 1
        delta = amt - mean_sent[src]
        mean_sent[src] += delta / (n + 1)
        m2_sent[src] += delta * (amt - mean_sent[src])
        times.append(ts)
        pairs.add((src, dst))
        prev = graph.get_edge_data(src, dst)
        graph.add_edge(src, dst, amount=amt + (prev["amount"] if prev else 0.0))

    return pd.DataFrame(rows).set_index("tx_id")
