import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CONTRACT_TYPES = ["NDA", "SOW", "Vendor Agreement"]
RISK_POOL = [
    "Uncapped liability clause favoring vendor",
    "Overly broad indemnification obligation on customer",
    "Automatic renewal with 180-day written cancellation notice",
    "Strict geographic non-compete and non-solicit restriction",
    "Unfavorable governing law and jurisdiction (e.g., Delaware/London under tight timelines)",
    "Non-GDPR compliant personal data transfer rules",
    "Missing export control compliance declarations",
    "Missing service level agreement (SLA) late delivery penalties",
    "Complete exclusion of all implied and express warranties on deliverables"
]

def build_generation_prompt(contract_type: str, selected_risks: list) -> str:
    risks_str = "\n".join([f"- {r}" for r in selected_risks])
    return f"""
    Generate a realistic, legally structured {contract_type} document.
    You MUST write a full-length agreement containing formal sections, standard preamble, boilerplate, and signatures.
    
    You MUST explicitly insert the following high-risk clauses/compliance gaps:
    {risks_str}
    
    Format the output in clean, clear text. If there are tables (e.g., pricing or SLAs), format them as Markdown tables.
    Avoid placeholders; use realistic names like Acme Corp, Apex Solutions, John Doe, etc.
    """

def generate_fifty_contracts(output_dir="./contracts"):
    os.makedirs(output_dir, exist_ok=True)
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    
    if not api_key or api_key == "mock_key_or_fill_it" or api_key == "your_nebius_api_key_here":
        print("Skipping LLM calls - generating offline mockup contracts instead.")
        for i in range(50):
            c_type = CONTRACT_TYPES[i % len(CONTRACT_TYPES)]
            filename = os.path.join(output_dir, f"contract_{i+1:03d}_{c_type.lower().replace(' ', '_')}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"MOCK CONTRACT {i+1}\nTYPE: {c_type}\nRisks injected:\n")
                for risk in random.sample(RISK_POOL, 4):
                    f.write(f"- {risk}\n")
                f.write("\nThis is standard contract body text.\n| Service | Price |\n| --- | --- |\n| Hosting | $1000/mo |\n")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    for i in range(50):
        c_type = CONTRACT_TYPES[i % len(CONTRACT_TYPES)]
        selected_risks = random.sample(RISK_POOL, 4)
        prompt = build_generation_prompt(c_type, selected_risks)
        
        # Llama-3.3-70B on Nebius
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        
        contract_text = response.choices[0].message.content
        filename = os.path.join(output_dir, f"contract_{i+1:03d}_{c_type.lower().replace(' ', '_')}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(contract_text)
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_fifty_contracts()
