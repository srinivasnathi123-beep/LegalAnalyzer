import os
from generate_contracts import build_generation_prompt

def test_build_generation_prompt():
    prompt = build_generation_prompt("NDA", ["uncapped liability", "strict non-compete"])
    assert "NDA" in prompt
    assert "uncapped liability" in prompt
    assert "strict non-compete" in prompt
