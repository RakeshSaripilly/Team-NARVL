"""
Script to generate data/data_edge_cases.csv based on data/data.csv.
Injects comprehensive real-world dirty data patterns, outliers, duplicates,
and adversarial edge cases targeted by the NARVL platform:

1. Adversarial Injections (L0):
   - Delimiter bombs (unquoted commas causing ragged/extra columns)
   - Truncated ragged rows (too few columns)
   - Null byte injections (\\x00)
   - Mojibake / mixed character encodings (e.g. CAF\\xc3\\x89, \\xc2\\xa3)

2. PII Injections (L1):
   - Email addresses inside Description/Notes
   - Phone numbers (international, US, Indian formats)
   - Credit card numbers (standard 16-digit formatted)
   - National IDs (SSN, Aadhaar)

3. Semantic / Approximate Functional Dependency Violations (L2-L3):
   - Country name typos & casing (United Kindom, Untied Kingdom, UK, U.K., FRANCE, Franc, Gerrmany)
   - StockCode -> Description approximate typos (e.g. WHITE HANGING HEART T LIGHT HOLDER, abbreviated, extra whitespace)

4. Outliers & Domain Boundary Violations (L4 / Pandera / GE):
   - Negative quantities (e.g. -80995, -500, -15)
   - Absurdly large quantities (e.g. 80000, 999999)
   - Negative unit prices (e.g. -11062.06, -2.55)
   - Absurd unit prices (e.g. 38970.00, 9999.00)

5. Exact Duplicates & Near-Duplicates:
   - Repeated transaction rows (3-5 identical duplicates)

6. Sparsity & Sentinel Values:
   - Missing CustomerID (empty, "N/A", "NULL", "?", "-999", "None")
   - Missing Description
   - Whitespace-only values

7. Timestamp Inconsistencies:
   - Mixed date formats (ISO, US, European, malformed strings)
"""

import csv
import io
import os

RAW_SOURCE = "data/data.csv"
OUTPUT_FILE = "data/data_edge_cases.csv"
BASE_ROW_LIMIT = 5000  # High-quality realistic base chunk

def generate_edge_cases():
    print(f"Reading base records from {RAW_SOURCE}...")
    
    rows = []
    with open(RAW_SOURCE, "r", encoding="ISO-8859-1") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            if i >= BASE_ROW_LIMIT:
                break
            rows.append(row)

    print(f"Extracted {len(rows)} base rows. Beginning injection of edge cases...")
    
    # Track statistics
    injected_counts = {
        "exact_duplicates": 0,
        "outliers_quantity": 0,
        "outliers_unit_price": 0,
        "typos_country": 0,
        "typos_stockcode_fd": 0,
        "pii_emails": 0,
        "pii_phones": 0,
        "pii_credit_cards": 0,
        "pii_ssn_aadhaar": 0,
        "sentinel_nulls": 0,
        "null_bytes": 0,
        "mojibake_encoding": 0,
        "timestamp_formats": 0,
        "delimiter_bombs_ragged": 0
    }

    # 1. Exact Duplicates: Select 5 rows and repeat each 3 times
    duplicate_candidates = [rows[10], rows[50], rows[120], rows[300], rows[500]]
    duplicate_rows_to_insert = []
    for dup in duplicate_candidates:
        for _ in range(3):
            duplicate_rows_to_insert.append(list(dup))
            injected_counts["exact_duplicates"] += 1

    # 2. Iterate through base rows and inject specific corruptions
    modified_rows = []
    
    for idx, row in enumerate(rows):
        r = list(row)
        
        # Outliers in Quantity
        if idx == 15:
            r[3] = "-80995"  # Extreme negative quantity
            injected_counts["outliers_quantity"] += 1
        elif idx == 42:
            r[3] = "-500"    # Negative quantity
            injected_counts["outliers_quantity"] += 1
        elif idx == 88:
            r[3] = "80995"   # Extreme positive volume spike
            injected_counts["outliers_quantity"] += 1
        elif idx == 150:
            r[3] = "999999"  # Absurd quantity
            injected_counts["outliers_quantity"] += 1

        # Outliers in UnitPrice
        if idx == 25:
            r[5] = "-11062.06"  # Negative unit price
            injected_counts["outliers_unit_price"] += 1
        elif idx == 65:
            r[5] = "-2.55"      # Negative price
            injected_counts["outliers_unit_price"] += 1
        elif idx == 110:
            r[5] = "38970.00"   # Extreme positive price outlier
            injected_counts["outliers_unit_price"] += 1
        elif idx == 210:
            r[5] = "9999.99"    # Impossible price
            injected_counts["outliers_unit_price"] += 1

        # Typo in Country (Categorical standardization)
        if idx % 100 == 12:
            r[7] = "United Kindom"  # Typo
            injected_counts["typos_country"] += 1
        elif idx % 100 == 24:
            r[7] = "Untied Kingdom" # Typo
            injected_counts["typos_country"] += 1
        elif idx % 100 == 36:
            r[7] = "UK"             # Abbreviation
            injected_counts["typos_country"] += 1
        elif idx % 100 == 48:
            r[7] = "U.K."           # Dotted abbreviation
            injected_counts["typos_country"] += 1
        elif idx % 100 == 60:
            r[7] = "FRANCE"         # All caps
            injected_counts["typos_country"] += 1
        elif idx % 100 == 72:
            r[7] = "Franc"          # Typo
            injected_counts["typos_country"] += 1
        elif idx % 100 == 84:
            r[7] = "Gerrmany"       # Typo
            injected_counts["typos_country"] += 1

        # StockCode -> Description FD typo / variation
        # For StockCode 85123A (WHITE HANGING HEART T-LIGHT HOLDER)
        if r[1] == "85123A":
            if idx % 4 == 1:
                r[2] = "WHITE HANGING HEART T LIGHT HOLDER"  # Missing hyphen
                injected_counts["typos_stockcode_fd"] += 1
            elif idx % 4 == 2:
                r[2] = "WHITE HANGING HEART T-LIGHT HLDR"     # Abbreviation
                injected_counts["typos_stockcode_fd"] += 1
            elif idx % 4 == 3:
                r[2] = "WHITE HANGING HEART T-LIGHT HOLDER   " # Trailing whitespace
                injected_counts["typos_stockcode_fd"] += 1
        # For StockCode 22632 (HAND WARMER RED POLKA DOT)
        elif r[1] == "22632":
            if idx % 3 == 1:
                r[2] = "HAND WARMER RED POLKA DOTS"
                injected_counts["typos_stockcode_fd"] += 1
            elif idx % 3 == 2:
                r[2] = "HAND WARMER RED POLKADOT"
                injected_counts["typos_stockcode_fd"] += 1

        # PII Traps in Description
        if idx == 105:
            r[2] = "GIFT WRAP - SEND RECEIPT TO sarah.connor@cyberdyne.org"
            injected_counts["pii_emails"] += 1
        elif idx == 205:
            r[2] = "SUPPORT INQUIRY: contact-admin@enterprise-retail.com"
            injected_counts["pii_emails"] += 1
        elif idx == 315:
            r[2] = "DELIVERY NOTE: CALL +1-555-839-2019 ON ARRIVAL"
            injected_counts["pii_phones"] += 1
        elif idx == 425:
            r[2] = "CUSTOMER PHONE: +44 20 7946 0958 FOR RETURNS"
            injected_counts["pii_phones"] += 1
        elif idx == 535:
            r[2] = "REFUND TO CARD 4532-8912-3456-7890 AUTHORIZED"
            injected_counts["pii_credit_cards"] += 1
        elif idx == 645:
            r[2] = "CHARGEBACK REF: 5424 1829 3948 2019 RECORDED"
            injected_counts["pii_credit_cards"] += 1
        elif idx == 755:
            r[2] = "CUSTOMER TAX ID SSN: 042-88-1920 VERIFIED"
            injected_counts["pii_ssn_aadhaar"] += 1
        elif idx == 865:
            r[2] = "GOV IDENTIFIER AADHAAR: 9021 3345 8892 MATCHED"
            injected_counts["pii_ssn_aadhaar"] += 1

        # Missing & Sentinel Values
        if idx % 80 == 5:
            r[6] = ""           # Empty CustomerID
            injected_counts["sentinel_nulls"] += 1
        elif idx % 80 == 15:
            r[6] = "N/A"        # Sentinel null
            injected_counts["sentinel_nulls"] += 1
        elif idx % 80 == 25:
            r[6] = "NULL"       # Sentinel null
            injected_counts["sentinel_nulls"] += 1
        elif idx % 80 == 35:
            r[6] = "?"          # Sentinel null
            injected_counts["sentinel_nulls"] += 1
        elif idx % 80 == 45:
            r[6] = "-999"       # Numeric sentinel null
            injected_counts["sentinel_nulls"] += 1
        elif idx % 80 == 55:
            r[2] = ""           # Missing description
            injected_counts["sentinel_nulls"] += 1
        elif idx % 80 == 65:
            r[2] = "nan"        # Text nan
            injected_counts["sentinel_nulls"] += 1

        # Null Byte Injections (\x00)
        if idx == 95:
            r[2] = "HAND WARMER\x00 RED POLKA DOT"
            injected_counts["null_bytes"] += 1
        elif idx == 195:
            r[1] = "85123A\x00"
            injected_counts["null_bytes"] += 1

        # Mojibake & Encoding Artifacts
        if idx == 135:
            r[2] = "CAF\xc3\x89 MUG SET"   # Mojibake for CAFÉ
            injected_counts["mojibake_encoding"] += 1
        elif idx == 245:
            r[2] = "\xc2\xa32.55 VINTAGE POSTAGE STAMPS"  # Mojibake for £
            injected_counts["mojibake_encoding"] += 1
        elif idx == 355:
            r[2] = "HANDMADE VINTAGE CHRISTMAS CARD SET"
            injected_counts["mojibake_encoding"] += 1

        # Timestamp Format Inconsistencies
        if idx == 18:
            r[4] = "2010-12-01 08:26:00"  # ISO-8601
            injected_counts["timestamp_formats"] += 1
        elif idx == 38:
            r[4] = "01/12/2010 08:26"     # DD/MM/YYYY European
            injected_counts["timestamp_formats"] += 1
        elif idx == 58:
            r[4] = "2010/12/01"           # Date only
            injected_counts["timestamp_formats"] += 1
        elif idx == 78:
            r[4] = "INVALID_TIMESTAMP"    # Corrupt string
            injected_counts["timestamp_formats"] += 1

        modified_rows.append(r)

    # Interleave duplicate rows
    final_rows = []
    dup_idx = 0
    for i, r in enumerate(modified_rows):
        final_rows.append(r)
        if i % 300 == 50 and dup_idx < len(duplicate_rows_to_insert):
            final_rows.append(duplicate_rows_to_insert[dup_idx])
            dup_idx += 1

    # Format into standard CSV rows (escaping quotes and commas properly)
    output_buffer = io.StringIO()
    writer = csv.writer(output_buffer, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for r in final_rows:
        writer.writerow(r)
    
    csv_content = output_buffer.getvalue()

    # Now inject Delimiter Bombs and Ragged Rows directly as raw unescaped text lines
    # These test L0 StreamingAdversarialShield quarantine logging
    lines = csv_content.splitlines(keepends=True)
    
    # Injected raw ragged lines (delimiter bombs & truncated lines)
    # Line with unquoted commas splitting into 12 columns
    ragged_bomb_1 = "536990,22632,HAND WARMER, RED, POLKA, DOT, SPECIAL EDITION,6,12/1/2010 10:00,1.85,17850,United Kingdom,EXTRA_BOMB_FIELD\n"
    # Line with truncated columns (only 4 columns instead of 8)
    ragged_bomb_2 = "536991,85123A,TRUNCATED_RECORD,4\n"
    # Line with 13 columns delimiter bomb
    ragged_bomb_3 = "536992,22752,DELIMITER_BOMB_EXPLODED,1,2,3,4,5,6,7,8,9,10\n"

    # Insert ragged lines at positions 100, 500, 1000
    lines.insert(100, ragged_bomb_1)
    injected_counts["delimiter_bombs_ragged"] += 1
    lines.insert(500, ragged_bomb_2)
    injected_counts["delimiter_bombs_ragged"] += 1
    lines.insert(1000, ragged_bomb_3)
    injected_counts["delimiter_bombs_ragged"] += 1

    # Write out data/data_edge_cases.csv
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="", errors="surrogateescape") as f:
        f.writelines(lines)

    print(f"\nSuccessfully generated {OUTPUT_FILE}!")
    print(f"Total lines: {len(lines)}")
    print("\nInjected Edge Case Breakdown:")
    for k, v in injected_counts.items():
        print(f"  - {k.replace('_', ' ').title()}: {v}")

if __name__ == "__main__":
    generate_edge_cases()
