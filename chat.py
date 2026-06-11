import sys
from rich.console import Console
from rich.panel import Panel
from search import hybrid_search
from agent import analyze_contract

console = Console()

def run_cli():
    console.print(Panel("[bold green]Legal Contract Analyzer CLI Chat[/bold green]\nCommands:\n- `/summarize <contract_id>` to analyze a contract.\n- `/file <contract_id> <query>` to search within a file.\n- `/exit` to quit.", expand=False))
    
    while True:
        try:
            user_input = console.input("[bold blue]Query/Command > [/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == "/exit":
            break
            
        elif user_input.startswith("/summarize"):
            parts = user_input.split(" ", 1)
            if len(parts) < 2:
                console.print("[red]Please specify a contract file name (e.g. /summarize contract_001_nda.txt)[/red]")
                continue
            c_id = parts[1].strip()
            console.print(f"Retrieving sections for: {c_id}...")
            fused_results = hybrid_search("contract details liability indemnification compliance GDPR SLA warranty", metadata_filter={"contract_id": c_id}, top_n=8)
            if not fused_results:
                console.print(f"[red]No chunks found for contract: {c_id}[/red]")
                continue
            analysis = analyze_contract(c_id, fused_results)
            console.print(Panel(analysis.model_dump_json(indent=2), title=f"[bold yellow]Analysis Report: {c_id}[/bold yellow]"))
            
        elif user_input.startswith("/file"):
            parts = user_input.split(" ", 2)
            if len(parts) < 3:
                console.print("[red]Usage: /file <contract_id> <your query>[/red]")
                continue
            c_id = parts[1].strip()
            query = parts[2].strip()
            console.print(f"Searching {c_id} for: '{query}'...")
            results = hybrid_search(query, metadata_filter={"contract_id": c_id}, top_n=3)
            for r in results:
                console.print(Panel(r["text"], title=f"Match (Section: {r['metadata']['section_name']})"))
                
        else:
            # Global Search Query
            console.print(f"Executing hybrid search for: '{user_input}'...")
            results = hybrid_search(user_input, top_n=3)
            if not results:
                console.print("[yellow]No matches found.[/yellow]")
            for r in results:
                console.print(Panel(r["text"], title=f"Source: {r['metadata']['contract_id']} (Section: {r['metadata']['section_name']})"))

if __name__ == "__main__":
    run_cli()
