from agent import run_local_fallback

def test_run_local_fallback():
    text_chunks = [
        "Section: SECTION 4. LIMITATION OF LIABILITY\nVendor's liability is uncapped under this agreement.",
        "Section: SECTION 9. GOVERNING LAW\nThis agreement is governed by the laws of Germany/GDPR compliance is missing."
    ]
    response = run_local_fallback("contract_001.txt", text_chunks)
    assert response.contract_id == "contract_001.txt"
    assert len(response.risks_found) > 0
    # Verify liability risk detection
    liability_risk = [r for r in response.risks_found if "liability" in r.clause_name.lower()]
    assert len(liability_risk) > 0
