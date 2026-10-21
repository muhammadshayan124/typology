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

## What the numbers mean (and what they don't)

Two things inflate results on a project like this, and both are measured here rather than
asserted away.

**1. Leakage.** The first version computed graph features over the *whole* dataset and
evaluated on a shuffled split, so a transaction's features included its own future. The
default is now **point-in-time**: each transaction sees only what happened before it, and
the evaluation is a **chronological split** (train on the earliest 70% of the timeline,
test on the latest 30%). `typology leakage` runs all four combinations on the same data:

| Features | Split | PR-AUC | Precision @0.5 | Recall @0.5 |
|---|---|---|---|---|
| dataset-wide (leaky) | shuffled | 0.988 | 95.31% | 92.42% |
| dataset-wide (leaky) | chronological | 0.964 | 92.00% | 92.00% |
| point-in-time | shuffled | 0.934 | 87.05% | 91.67% |
| **point-in-time** | **chronological** | **0.876** | **85.00%** | **81.60%** |

The original headline (0.988) overstated the deployable figure by about 0.11 PR-AUC, and
recall by about 11 points.

**2. The generator is easier than reality.** Injected laundering amounts are large against
a log-normal legitimate distribution (median legitimate transaction about €89), so
`amount` alone separates a lot of it. `typology run` therefore also trains an
**amount-only baseline**: 0.726 PR-AUC against 0.876 for the full feature set. The honest
claim is that the graph features add about 0.15 PR-AUC over a trivial baseline on this
data, not that 0.876 predicts real-world performance. Real launderers adapt, obfuscate and
do not announce their typology, and the features were engineered to detect exactly the
typologies the generator injects, which remains circular. This demonstrates an engineering
approach, not a claim about real AML detection.

Point-in-time features expose a structural weakness the leaky version hid. Recall at 0.5
by typology on the test period:

| Typology | Test transactions | Recall |
|---|---|---|
| layering | 13 | 100% |
| structuring | 41 | 95% |
| fan-out | 42 | 74% |
| fan-in | 29 | 66% |

Fan-out and fan-in are bursts: the first few transfers in a burst arrive before any
history says "burst", so they are scored without it. Dataset-wide features hid this by
letting early transfers see the rest of the burst. Detecting these earlier would need
features about the *recipient* side (a dormant account suddenly receiving) rather than
more velocity features.

PageRank is the one feature a causal pipeline cannot update per transaction cheaply, so it
uses a daily snapshot (stale by up to a day, never forward-looking).

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

Real output from a default run (50,416 transactions, 441 labeled laundering, 0.87%; trained
on the first 70% of the timeline, tested on the last 30%):

```
Naive baseline (always predict 'legitimate') accuracy: 99.17%
PR-AUC: 0.876  (amount-only baseline: 0.726)

Threshold  Precision  Recall
0.10       77.46%     88.00%
0.50       85.00%     81.60%
0.90       93.81%     72.80%
```

```bash
typology leakage     # the four-way comparison above
```

Plus a SHAP-based explanation for sample flagged transactions — the real regulatory need
this addresses: an AML alert that can't be explained to a compliance officer isn't
actionable, however accurate the underlying model is.

## Test

```bash
pytest
```

24 tests: structural checks on the generator (each typology actually produces the graph
shape it claims to), hand-verified feature computations on small known graphs, and model
sanity checks (valid probabilities, beats the naive and the amount-only baselines, chronological
split really tests only on the future).

The key test for the point-in-time claim edits and appends *later* transactions and asserts
that every earlier row's features are unchanged. It was checked against the old dataset-wide
builder to confirm it fails there, so it is not vacuous.

## Real bug found by actually running this, not just writing it

The first version of the threshold-vs-precision/recall report table was nearly useless:
it downsampled scikit-learn's `precision_recall_curve` output by array *index*, but
XGBoost outputs polarized probabilities (clustered near 0 or 1), so an index-based sample
landed almost every row near threshold 0.00. Fixed to evaluate at fixed threshold values
(0.1, 0.2, ... 0.9) directly — see `model.py`. The `src_velocity_1day` feature also
started as an O(n²) full-DataFrame scan per transaction, which doesn't finish in
reasonable time past a few thousand rows; fixed to per-account sorted-timestamp arrays
with binary search before ever running it at realistic scale.

## License

MIT
