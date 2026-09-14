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
    features = build_features(df)
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
    features = build_features(df)
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
    features = build_features(df)
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
    features = build_features(df)
    assert features.loc[0, "is_first_time_pair"] == 1
    assert features.loc[1, "is_first_time_pair"] == 0
    assert features.loc[2, "is_first_time_pair"] == 1
