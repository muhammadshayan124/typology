from typology.generator import REPORTING_THRESHOLD, GeneratorConfig, generate


def _small_config(**overrides) -> GeneratorConfig:
    defaults = dict(
        n_accounts=200,
        n_legitimate_tx=2000,
        n_days=30,
        n_fan_out=3,
        n_fan_in=3,
        n_layering_chains=3,
        n_structuring=3,
        seed=1,
    )
    defaults.update(overrides)
    return GeneratorConfig(**defaults)


def test_produces_expected_columns():
    df = generate(_small_config())
    assert set(df.columns) == {"tx_id", "timestamp", "src", "dst", "amount", "is_laundering", "typology"}


def test_reproducible_with_same_seed():
    df1 = generate(_small_config(seed=7))
    df2 = generate(_small_config(seed=7))
    assert df1.equals(df2)


def test_different_seeds_produce_different_data():
    df1 = generate(_small_config(seed=1))
    df2 = generate(_small_config(seed=2))
    assert not df1["amount"].equals(df2["amount"])


def test_all_typologies_are_labeled_laundering():
    df = generate(_small_config())
    laundering = df[df["typology"] != "legitimate"]
    assert len(laundering) > 0
    assert (laundering["is_laundering"] == 1).all()


def test_legitimate_transactions_are_not_labeled_laundering():
    df = generate(_small_config())
    legit = df[df["typology"] == "legitimate"]
    assert (legit["is_laundering"] == 0).all()


def test_fan_out_has_one_source_many_destinations_within_short_window():
    df = generate(_small_config(n_fan_out=1, n_fan_in=0, n_layering_chains=0, n_structuring=0))
    fan_out = df[df["typology"] == "fan_out"]
    assert fan_out["src"].nunique() == 1
    assert fan_out["dst"].nunique() >= 5
    assert fan_out["timestamp"].max() - fan_out["timestamp"].min() <= 0.5


def test_structuring_amounts_stay_under_reporting_threshold():
    df = generate(_small_config(n_fan_out=0, n_fan_in=0, n_layering_chains=0, n_structuring=3))
    structuring = df[df["typology"] == "structuring"]
    assert len(structuring) > 0
    assert (structuring["amount"] < REPORTING_THRESHOLD).all()


def test_layering_chain_moves_through_multiple_distinct_accounts():
    df = generate(_small_config(n_fan_out=0, n_fan_in=0, n_layering_chains=1, n_structuring=0))
    chain = df[df["typology"] == "layering"].sort_values("timestamp")
    accounts_touched = set(chain["src"]) | set(chain["dst"])
    assert len(accounts_touched) >= 4  # chain_len is 4-8 accounts per generator.py
    # each hop's destination should be the next hop's source, forming an actual chain
    srcs = chain["src"].tolist()
    dsts = chain["dst"].tolist()
    assert srcs[1:] == dsts[:-1]
