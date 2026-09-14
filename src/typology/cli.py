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

    console.print("[bold]Building graph features...[/bold]")
    features = build_features(df)

    console.print("[bold]Training and evaluating...[/bold]")
    model, result = train_and_evaluate(features, df.set_index("tx_id")["is_laundering"])

    console.print(
        f"\n[yellow]Naive baseline (always predict 'legitimate') accuracy: "
        f"{result.naive_baseline_accuracy:.2%}[/yellow] -- this is why accuracy alone "
        f"would be a misleading headline number here.\n"
    )
    console.print(f"PR-AUC: [bold]{result.pr_auc:.3f}[/bold]")
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

    laundering_ids = df[df["is_laundering"] == 1]["tx_id"].tolist()
    sample_ids = [tid for tid in laundering_ids if tid in features.index][:explain_n]
    if sample_ids:
        console.print("\n[bold]Sample explanations for known-laundering transactions:[/bold]")
        for tid in sample_ids:
            explanation = explain_flagged(model, features, tid)
            console.print(f"  tx {tid} (score={explanation['score']:.2f}):")
            for factor in explanation["top_factors"]:
                console.print(f"    {factor['feature']} = {factor['value']:.2f} (SHAP {factor['shap_contribution']:+.3f})")


if __name__ == "__main__":
    app()
