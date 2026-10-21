import pandas as pd

from typology.features import build_features


def _tx(tx_id, ts, src, dst, amount):
    return {"tx_id": tx_id, "timestamp": ts, "src": src, "dst": dst, "amount": amount, "is_laundering": 0, "typology": "legitimate"}


def test_degree_counts_are_correct_on_known_graph():
    # account 0 sends to 1, 2, 3 (out-degree 3); account 1 receives from 0 only (in-degree 1)
    df = pd.DataFrame(
        [
            _tx(0, 0.0, 0, 1, 100.0),
            _tx(1, 0.1, 0, 2, 200.0),
            _tx(2, 0.2, 0, 3, 300.0),
        ]
    )
    features = build_features(df, point_in_time=False)
    assert features.loc[0, "src_out_degree"] == 3
    assert features.loc[0, "dst_in_degree"] == 1  # dst of tx 0 is account 1, which has in-degree 1


def test_velocity_counts_burst_within_window_not_outside_it():
    # account 5 sends 4 transactions within half a day of each other (a burst), plus one
    # far outside the 1-day window -- velocity should count the burst, not the outlier.
    df = pd.DataFrame(
        [
            _tx(0, 10.0, 5, 1, 50.0),
            _tx(1, 10.1, 5, 2, 50.0),
            _tx(2, 10.2, 5, 3, 50.0),
            _tx(3, 10.3, 5, 4, 50.0),
            _tx(4, 50.0, 5, 6, 50.0),  # far away in time -- should not count
        ]
    )
    features = build_features(df, point_in_time=False)
    # tx 0 at ts=10.0, window +-1 day catches ts in [9.0, 11.0] -> tx 0,1,2,3 all qualify
    assert features.loc[0, "src_velocity_1day"] == 4
    # tx 4 at ts=50.0 is isolated -- only itself within its own +-1 day window
    assert features.loc[4, "src_velocity_1day"] == 1


def test_near_threshold_flag_and_ratio_are_consistent():
    from typology.generator import REPORTING_THRESHOLD

    df = pd.DataFrame(
        [
            _tx(0, 0.0, 0, 1, REPORTING_THRESHOLD * 0.9),  # near threshold
            _tx(1, 0.1, 0, 1, REPORTING_THRESHOLD * 0.1),  # not near threshold
        ]
    )
    features = build_features(df, point_in_time=False)
    assert features.loc[0, "amount_to_threshold_ratio"] == 0.9
    assert features.loc[1, "amount_to_threshold_ratio"] == 0.1


def test_is_first_time_pair_flags_correctly():
    df = pd.DataFrame(
        [
            _tx(0, 0.0, 0, 1, 100.0),  # first time 0->1
            _tx(1, 1.0, 0, 1, 100.0),  # repeat of 0->1
            _tx(2, 0.5, 0, 2, 100.0),  # first time 0->2
        ]
    )
    features = build_features(df, point_in_time=False)
    assert features.loc[0, "is_first_time_pair"] == 1
    assert features.loc[1, "is_first_time_pair"] == 0
    assert features.loc[2, "is_first_time_pair"] == 1


# --- point-in-time (the default) -------------------------------------------------------

def test_point_in_time_features_ignore_the_future():
    """Editing or appending later transactions must not change earlier rows' features."""
    base = pd.DataFrame(
        [
            _tx(0, 0.0, 0, 1, 100.0),
            _tx(1, 0.5, 0, 2, 250.0),
            _tx(2, 1.5, 1, 0, 40.0),
            _tx(3, 2.0, 0, 1, 900.0),
        ]
    )
    future = pd.concat(
        [
            base,
            pd.DataFrame([_tx(4, 3.0, 0, 5, 9_500.0), _tx(5, 3.1, 0, 6, 9_400.0), _tx(6, 3.2, 7, 0, 8.0)]),
        ],
        ignore_index=True,
    )
    a = build_features(base)
    b = build_features(future)
    pd.testing.assert_frame_equal(a, b.loc[a.index])


def test_point_in_time_degree_and_first_pair_grow_with_history():
    df = pd.DataFrame(
        [
            _tx(0, 0.0, 0, 1, 100.0),
            _tx(1, 0.1, 0, 2, 100.0),
            _tx(2, 0.2, 0, 3, 100.0),
            _tx(3, 0.3, 0, 1, 100.0),  # repeat counterparty
        ]
    )
    f = build_features(df)
    assert list(f["src_out_degree"]) == [0, 1, 2, 3]
    assert list(f["src_tx_count"]) == [0, 1, 2, 3]
    assert list(f["is_first_time_pair"]) == [1, 1, 1, 0]
    assert list(f["src_total_sent"]) == [0.0, 100.0, 200.0, 300.0]


def test_point_in_time_velocity_counts_only_earlier_sends_in_window():
    df = pd.DataFrame(
        [
            _tx(0, 10.0, 5, 1, 50.0),
            _tx(1, 10.1, 5, 2, 50.0),
            _tx(2, 10.2, 5, 3, 50.0),
            _tx(3, 50.0, 5, 6, 50.0),  # far later: earlier burst is outside its window
        ]
    )
    f = build_features(df)
    assert list(f["src_velocity_1day"]) == [1, 2, 3, 1]


def test_point_in_time_zscore_matches_running_statistics():
    df = pd.DataFrame([_tx(i, i * 0.1, 0, 1, a) for i, a in enumerate([10.0, 20.0, 30.0, 100.0])])
    f = build_features(df)
    # Third row sees history [10, 20]: mean 15, sample std ~7.0711.
    assert abs(f.iloc[2]["amount_zscore_vs_src_history"] - (30 - 15) / (50**0.5)) < 1e-9
    # First two rows have <2 prior sends, so no standardisation is possible.
    assert f.iloc[0]["amount_zscore_vs_src_history"] == 0.0
    assert f.iloc[1]["amount_zscore_vs_src_history"] == 0.0


def test_point_in_time_pagerank_uses_a_snapshot_from_before_the_day():
    df = pd.DataFrame([_tx(0, 0.2, 0, 1, 100.0), _tx(1, 0.8, 1, 0, 100.0), _tx(2, 1.1, 0, 1, 100.0)])
    f = build_features(df)
    assert f.loc[0, "src_pagerank"] == 0.0 and f.loc[1, "src_pagerank"] == 0.0  # day 0: empty graph
    assert f.loc[2, "src_pagerank"] > 0.0  # day 1: snapshot of day 0's two edges
