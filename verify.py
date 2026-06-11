import sys
import os
from search import hybrid_search
from agent import analyze_contract
from rich.console import Console
from rich.table import Table

console = Console()

PRESET_QUERIES = [
    ("limitation of liability cap", "Find uncapped liability clauses"),
    ("GDPR personal data transfer", "Find GDPR compliance gaps"),
    ("Service Level Agreement SLA late fee penalty", "Identify missing SLA penalties"),
    ("warranty disclaimer express implied deliverables", "Check for absence of warranties"),
    ("indemnification customer obligation", "Find broad indemnification scopes"),
    ("automatic renewal cancellation notice 180 days", "Detect auto-renewal risk"),
    ("non-compete restriction geographical scope", "Identify non-compete restrictions"),
    ("governing law jurisdiction dispute resolution", "Check governing law/jurisdiction risks"),
    ("export control compliance restrictions", "Identify missing export controls"),
    ("liability cap vendor limitation", "Double check standard vendor liability limits")
]

def run_verification():
    console.print("[bold yellow]Starting Verification Harness...[/bold yellow]")
    
    # 1. Ingestion check
    bm25_path = "./data/bm25_index.pkl"
    if not os.path.exists(bm25_path):
        console.print("[red]BM25 local index file missing. Run ingest.py first.[/red]")
        sys.exit(1)
        
    console.print("[green][OK] BM25 index found.[/green]")
    
    # 2. Run Preset Queries and check Pydantic constraint mapping
    table = Table(title="Preset Search Queries Verification")
    table.add_column("Query Topic", style="cyan")
    table.add_column("Query String", style="magenta")
    table.add_column("Results Found", style="green")
    table.add_column("Schema Validated", style="blue")
    
    success_count = 0
    for query, description in PRESET_QUERIES:
        results = hybrid_search(query, top_n=2)
        results_found = len(results)
        
        schema_ok = "N/A"
        if results_found > 0:
            try:
                contract_id = results[0]["metadata"]["contract_id"]
                analysis = analyze_contract(contract_id, results)
                schema_ok = "Pass"
                success_count += 1
            except Exception as e:
                schema_ok = f"Fail ({str(e)})"
        else:
            schema_ok = "Fail (No matches)"
            
        table.add_row(description, query, str(results_found), schema_ok)
        
    console.print(table)
    
    if success_count == len(PRESET_QUERIES):
        console.print("[bold green][OK] All verification tests passed successfully![/bold green]")
    else:
        console.print(f"[bold red][FAIL] Verification finished with {len(PRESET_QUERIES) - success_count} failures.[/bold red]")

if __name__ == "__main__":
    run_verification()
