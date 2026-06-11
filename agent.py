import os
import re
import json
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RiskClause(BaseModel):
    clause_name: str = Field(description="Name of the risky clause")
    raw_text: str = Field(description="The exact text of the clause from the contract")
    risk_explanation: str = Field(description="Why this clause is considered high risk")
    severity: str = Field(description="Low, Medium, or High")

class ComplianceGap(BaseModel):
    gap_name: str = Field(description="Name of the gap")
    details: str = Field(description="Details on what is missing or non-compliant")

class ContractAnalysisResponse(BaseModel):
    contract_id: str
    contract_type: str
    risks_found: List[RiskClause]
    compliance_gaps: List[ComplianceGap]
    missing_terms: List[str]
    overall_risk_score: int

def run_local_fallback(contract_id: str, chunks: List[str]) -> ContractAnalysisResponse:
    risks = []
    gaps = []
    missing = []
    
    full_text = "\n".join(chunks)
    
    # Liability Matcher (supporting uncapped liability, liability is uncapped, and general uncapped liability terms)
    if re.search(r'uncapped\s+liability|liability\s+is\s+uncapped|no\s+limit\s+on\s+liability|liability\s+shall\s+be\s+unlimited|liability\s+.*?\s*uncapped', full_text, re.IGNORECASE):
        risks.append(RiskClause(
            clause_name="Limitation of Liability",
            raw_text="Uncapped liability detected in text chunks.",
            risk_explanation="The contract text contains terminology indicating vendor or client liability is unlimited or uncapped.",
            severity="High"
        ))
        
    # Auto-renewal Matcher
    if re.search(r'automatically\s+renew|auto-renew|renew\s+automatically', full_text, re.IGNORECASE):
        risks.append(RiskClause(
            clause_name="Auto-Renewal Term",
            raw_text="Auto-renewal clauses detected.",
            risk_explanation="Contract automatically renews. High risk if cancellation requires long notice periods.",
            severity="Medium"
        ))
        
    # GDPR Check
    if not re.search(r'GDPR|data\s+protection|privacy\s+regulation', full_text, re.IGNORECASE):
        gaps.append(ComplianceGap(
            gap_name="Data Protection",
            details="No explicit GDPR compliance terms or data protection clauses found in text chunks."
        ))
        
    # SLA Check
    if not re.search(r'SLA|service\s+level|performance\s+standards', full_text, re.IGNORECASE):
        missing.append("Service Level Agreement (SLA) Penalties")

    # Export Check
    if not re.search(r'export\s+control|compliance\s+with\s+export', full_text, re.IGNORECASE):
        missing.append("Export Control Compliance Clause")
        
    return ContractAnalysisResponse(
        contract_id=contract_id,
        contract_type="Unknown/Mixed",
        risks_found=risks,
        compliance_gaps=gaps,
        missing_terms=missing,
        overall_risk_score=max(7 if risks else 3, 1)
    )

def analyze_contract(contract_id: str, chunks: list) -> ContractAnalysisResponse:
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    
    text_content = "\n\n".join([c["text"] for c in chunks])
    
    if not api_key or api_key in ["mock_key_or_fill_it", "your_nebius_api_key_here"]:
        # Trigger offline local fallback directly
        return run_local_fallback(contract_id, [c["text"] for c in chunks])
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    prompt = f"""
    You are an expert legal review agent. Analyze the following retrieved contract clauses from '{contract_id}' and identify:
    1. Risky clauses (liability caps, broad indemnifications, non-competes, unfavorable governing laws).
    2. Compliance gaps (data protection/GDPR failures, export controls).
    3. Missing standard terms (SLA penalties, warranties).
    
    Retrieved Contract Clauses:
    {text_content}
    
    You MUST respond with a single, raw JSON object matching this schema:
    {{
      "contract_id": "{contract_id}",
      "contract_type": "NDA, SOW, or Vendor Agreement",
      "risks_found": [
        {{
          "clause_name": "Clause title",
          "raw_text": "Exact text quote",
          "risk_explanation": "Detailed explanation of risk",
          "severity": "Low, Medium, or High"
        }}
      ],
      "compliance_gaps": [
        {{
          "gap_name": "Compliance standard",
          "details": "What is lacking"
        }}
      ],
      "missing_terms": ["Missing standard term 1", "Missing standard term 2"],
      "overall_risk_score": 1 to 10 integer
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return ContractAnalysisResponse(**data)
    except Exception as e:
        print(f"Analysis error: {e}. Activating Offline Local Fallback.")
        return run_local_fallback(contract_id, [c["text"] for c in chunks])

if __name__ == "__main__":
    dummy_chunks = [{"text": "Section: LIABILITY\nUncapped liability applies to both parties."}]
    result = analyze_contract("dummy.txt", dummy_chunks)
    print(result.model_dump_json(indent=2))
