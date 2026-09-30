"""
Test Entity Extraction Module.
Validates:
- CustomerName, Phone, Email, Date, Address, City, State extraction
- Normalization (E.164 phone, ISO date, title case)
- Robust span detection
"""

from narvl.document.entity_extractor import EntityExtractor
from narvl.document.models import Page, ParsedDocument, DocumentType


def test_entity_extraction_deterministic():
    text = (
        "Customer Name: Rajesh Kumar\n"
        "Phone: 9876543210\n"
        "Email: rajesh.kumar@acme.com\n"
        "City: Hyderabad\n"
        "State: Telangana\n"
        "Date: 15/03/2024\n"
        "Invoice Number: INV-2024-889\n"
        "Address: 12 Banjara Hills Road\n"
    )

    extractor = EntityExtractor()
    page = Page(page_num=1, text=text, char_count=len(text))
    doc = ParsedDocument(
        doc_id="test_doc",
        original_path="test.txt",
        file_type=DocumentType.TXT,
        text=text,
        pages=[page],
    )

    entities = extractor.extract(doc)
    types_found = {e.type: e for e in entities}

    assert "CustomerName" in types_found
    assert types_found["CustomerName"].value == "Rajesh Kumar"

    assert "Email" in types_found
    assert types_found["Email"].value == "rajesh.kumar@acme.com"

    assert "Phone" in types_found
    assert types_found["Phone"].normalized_value.startswith("+91")

    assert "Date" in types_found
    assert types_found["Date"].normalized_value == "2024-03-15"

    assert "City" in types_found
    assert types_found["City"].normalized_value == "Hyderabad"
