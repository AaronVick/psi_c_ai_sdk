# File: psi_c_ai_sdk/cli/status.py

import typer
from rich import print
from rich.table import Table
from psi_c_ai_sdk.epistemic.epistemic_status import EpistemicStatus

app = typer.Typer()
status = EpistemicStatus()

@app.command("trust")
def trust_status():
    """View trust and persuasion entropy scores"""
    table = Table(title="Epistemic Trust Status")

    table.add_column("Source ID", style="cyan")
    table.add_column("Trust Score", style="green")
    table.add_column("Entropy", style="yellow")

    trust_scores = status.get_all_trust_scores()
    entropy_scores = status.get_entropy_scores()

    for source_id in trust_scores:
        trust = round(trust_scores[source_id], 4)
        entropy = round(entropy_scores.get(source_id, 0.0), 4)
        table.add_row(source_id, str(trust), str(entropy))

    print(table)

@app.command("memories")
def flagged_memories():
    """View memory labels and AGI influence"""
    from rich.pretty import pprint
    metadata = status.export_metadata()
    print("[bold magenta]Flagged Memory Metadata[/bold magenta]")
    pprint(metadata)
