from search import reciprocal_rank_fusion

def test_reciprocal_rank_fusion():
    dense_results = [{"id": "chunk_1", "text": "A"}, {"id": "chunk_2", "text": "B"}]
    sparse_results = [{"id": "chunk_2", "text": "B"}, {"id": "chunk_3", "text": "C"}]
    
    fused = reciprocal_rank_fusion(dense_results, sparse_results, top_n=2)
    assert len(fused) == 2
    # chunk_2 should be ranked highest as it appears in both lists
    assert fused[0]["id"] == "chunk_2"
