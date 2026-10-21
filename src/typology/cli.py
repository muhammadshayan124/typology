import typer
from rich.console import Console
from rich.table import Table

from typology.explain import explain_flagged
from typology.features import build_features
from typology.generator import GeneratorConfig, generate
from typology.model import train_and_evaluate

app = typer.Typer(help="Graph-aware AML transaction-monitoring anomaly detector.")
console = Console()


@app.callback()
def _main() -> None:
    """Present to keep 'run' as an explicit subcommand -- see manifest's cli.py for why
    this matters with only one command registered (Typer auto-promotes otherwise)."""


@app.command()
def run(
    n_accounts: int = 2000,
    n_legitimate_tx: int = 50_000,
    seed: int = 42,
    explain_n: int = 3,
) -> None:
    """Generate a synthetic transaction dataset, train, and report honest evaluation
    metrics -- precision/recall/PR-AUC, not accuracy. See README before treating the
    numbers as directly comparable to published IBM/PaySim benchmarks: this is a
    self-built synthetic dataset, not those datasets.
    """
    console.print("[bold]Generating synthetic transaction network...[/bold]")
    cfg = GeneratorConfig(n_accounts=n_accounts, n_legitimate_tx=n_legitimate_tx, seed=seed)
    df = generate(cfg)
    n_laundering = int(df["is_laundering"].sum())
    console.print(f"  {len(df):,} transactions, {n_laundering} labeled laundering ({n_laundering / len(df):.2%})")

    console.print("[bold]Building point-in-time graph features...[/bold]")
    features = build_features(df)

    console.print("[bold]Training on the earliest 70% of the timeline, testing on the latest 30%...[/bold]")
    model, result = train_and_evaluate(features, df.set_index("tx_id")["is_laundering"])

    console.print(
        f"\n[yellow]Naive baseline (always predict 'legitimate') accuracy: "
        f"{result.naive_baseline_accuracy:.2%}[/yellow] -- this is why accuracy alone "
        f"would be a misleading headline number here.\n"
    )
    labels = df.set_index("tx_id")["is_laundering"]
    _, amount_only = train_and_evaluate(features, labels, columns=["amount"])
    console.print(
        f"PR-AUC: [bold]{result.pr_auc:.3f}[/bold]  "
        f"(amount-only baseline: {amount_only.pr_auc:.3f}; the gap is what the graph features add)"
    )
    console.print(
        f"At the default 0.5 threshold: precision={result.precision_at_default_threshold:.2%}, "
        f"recall={result.recall_at_default_threshold:.2%} "
        f"(n_test={result.n_test:,}, n_positive_test={result.n_positive_test})"
    )

    table = Table(title="Precision/recall across thresholds -- the real trade-off")
    table.add_column("Threshold")
    table.add_column("Precision")
    table.add_column("Recall")
    for row in result.threshold_table.itertuples():
        table.add_row(f"{row.threshold:.2f}", f"{row.precision:.2%}", f"{row.recall:.2%}")
    console.print(table)

    typology_of = df.set_index("tx_id")["typology"]
    positives = result.test_scores[result.test_scores.index.map(lambda t: typology_of[t] != "legitimate")]
    by_typology = Table(title="Recall at 0.5 by typology (test period)")
    by_typology.add_column("Typology")
    by_typology.add_column("Transactions")
    by_typology.add_column("Recall")
    for name, group in positives.groupby(positives.index.map(typology_of)):
        by_typology.add_row(str(name), str(len(group)), f"{(group >= 0.5).mean():.2%}")
    console.print(by_typology)

    laundering_ids = df[df["is_laundering"] == 1]["tx_id"].tolist()
    sample_ids = [tid for tid in laundering_ids if tid in features.index][:explain_n]
    if sample_ids:
        console.print("\n[bold]Sample explanations for known-laundering transactions:[/bold]")
        for tid in sample_ids:
            explanation = explain_flagged(model, features, tid)
            console.print(f"  tx {tid} (score={explanation['score']:.2f}):")
            for factor in explanation["top_factors"]:
                console.print(f"    {factor['feature']} = {factor['value']:.2f} (SHAP {factor['shap_contribution']:+.3f})")


@app.command()
def leakage(
    n_accounts: int = 2000,
    n_legitimate_tx: int = 50_000,
    seed: int = 42,
) -> None:
    """Train the same model under four evaluation setups and show how much the optimistic
    ones overstate performance: dataset-wide vs point-in-time features, shuffled vs
    chronological split.
    """
    cfg = GeneratorConfig(n_accounts=n_accounts, n_legitimate_tx=n_legitimate_tx, seed=seed)
    df = generate(cfg)
    labels = df.set_index("tx_id")["is_laundering"]

    table = Table(title="The cost of evaluating optimistically (same data, same model)")
    table.add_column("Features")
    table.add_column("Split")
    table.add_column("PR-AUC")
    table.add_column("Precision @0.5")
    table.add_column("Recall @0.5")
    for point_in_time in (False, True):
        features = build_features(df, point_in_time=point_in_time)
        for split in ("random", "temporal"):
            _, r = train_and_evaluate(features, labels, split=split)
            table.add_row(
                "point-in-time" if point_in_time else "dataset-wide (leaky)",
                split,
                f"{r.pr_auc:.3f}",
                f"{r.precision_at_default_threshold:.2%}",
                f"{r.recall_at_default_threshold:.2%}",
            )
    console.print(table)


if __name__ == "__main__":
    app()
