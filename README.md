# typology

A graph-aware AML (anti-money laundering) transaction-monitoring anomaly detector,
evaluated honestly on the precision/recall trade-off — not accuracy, which is actively
misleading on data this imbalanced (see "Why not just report accuracy" below).

Built around a real, current gap: the EU's new **AMLA** (Anti-Money Laundering Authority,
operational since 2025, headquartered in Frankfurt) has published its own finding that
roughly 80% of EU financial institutions currently lack the data/tooling to meet upcoming
AML reporting requirements, with its single rulebook applying EU-wide from July 2027.

## About the data — read this before the results below

**This is not the Kaggle IBM-AML or PaySim dataset.** Both require a Kaggle account to
download, and IBM's own direct-download link (ibm.box.com/v/AML-Anti-Money-Laundering-Data)
returns a 404 as of this build — checked directly, not assumed. Rather than lose the
project to that access gate, `typology` generates its own synthetic transaction network
locally (seeded, fully reproducible, `src/typology/generator.py`) — the same general
approach IBM's own AMLworld paper uses (a synthetic multi-agent transaction world with
injected laundering patterns), just transparently disclosed as self-built rather than
borrowed under someone else's name.

The injected patterns are real, documented laundering typologies — the term "typology" is
itself standard AML/FATF (Financial Action Task Force) terminology, not this project's
invention:

- **Fan-out ("smurfing")** — one account splits a large sum into many small transfers to
  many destinations in a short window.
- **Fan-in** — the mirror image: many small transfers converge on one account.
- **Layering chain** — money moves quickly through a chain of intermediary accounts to
  obscure its origin.
- **Structuring** — a sum is split into transfers each just under a reporting threshold
  (modeled here as €10,000), spread across several days specifically to stay under it.

## The honest caveat that matters most

**The strong numbers below are partly circular, and that's worth being direct about.**
The graph features (`src_velocity_1day`, `near_threshold`, `amount_zscore_vs_src_history`,
...) were deliberately engineered to detect exactly the typologies the generator injects.
Scoring well here mostly demonstrates that the detector recognizes the *known* patterns it
was built to look for — real money launderers adapt, obfuscate, and don't announce which
typology they're using. This project demonstrates the *engineering approach* (graph
features + imbalance-aware evaluation + auditable explanations) honestly; it is not a
claim that real-world AML detection performs anywhere near this well. See also the
feature-scope limitation in `features.py`'s module docstring (features are computed
dataset-wide, not as a true point-in-time snapshot — a real system needs genuinely
temporal features to avoid leaking future information into a transaction's score).

## Why not just report accuracy

With laundering at under 1% of transactions, a model that predicts "legitimate" for
everything scores **99.13% accuracy while catching zero laundering.** Every result below
reports precision, recall, and PR-AUC instead, and states that naive-baseline number
explicitly so nobody mistakes a high accuracy figure for a good model later.

## Install & run

```bash
python -m venv .venv
./.venv/Scripts/pip install -e ".[dev]"
./.venv/Scripts/typology run
```

Real output from a default run (50,416 transactions, 441 labeled laundering, 0.87%):

```
Naive baseline (always predict 'legitimate') accuracy: 99.13%
PR-AUC: 0.988

Threshold  Precision  Recall
0.10       90.00%     95.45%
0.50       95.31%     92.42%
0.90       97.56%     90.91%
```

Plus a SHAP-based explanation for sample flagged transactions — the real regulatory need
this addresses: an AML alert that can't be explained to a compliance officer isn't
actionable, however accurate the underlying model is.

## Test

```bash
pytest
```

16 tests: structural checks on the generator (each typology actually produces the graph
shape it claims to — a fan-out really has one source and many destinations within a short
window, a layering chain really is a connected sequence of hops, structuring amounts stay
under the threshold), hand-verified graph feature computations on small known graphs, and
model evaluation sanity checks (valid probability ranges, beats the naive baseline by a
wide margin, sensible precision/recall trade-off direction across thresholds).

## Real bug found by actually running this, not just writing it

The first version of the threshold-vs-precision/recall report table was nearly useless:
it downsampled scikit-learn's `precision_recall_curve` output by array *index*, but
XGBoost outputs polarized probabilities (clustered near 0 or 1), so an index-based sample
landed almost every row near threshold 0.00. Fixed to evaluate at fixed threshold values
(0.1, 0.2, ... 0.9) directly — see `model.py`. The `src_velocity_1day` feature also
started as an O(n²) full-DataFrame scan per transaction, which doesn't finish in
reasonable time past a few thousand rows; fixed to per-account sorted-timestamp arrays
with binary search before ever running it at realistic scale.
