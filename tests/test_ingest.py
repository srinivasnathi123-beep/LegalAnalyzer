from ingest import chunk_text

def test_chunk_text():
    text = "SECTION 1. LIMITATION OF LIABILITY.\nVendor shall not be liable.\n\nSECTION 2. INDEMNIFICATION.\nCustomer shall indemnify Vendor."
    chunks = chunk_text(text, filename="test_doc.txt")
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["section_name"] == "SECTION 1. LIMITATION OF LIABILITY."
    assert chunks[1]["metadata"]["section_name"] == "SECTION 2. INDEMNIFICATION."
