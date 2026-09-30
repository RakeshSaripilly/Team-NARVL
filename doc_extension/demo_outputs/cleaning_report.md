# NARVL Document Intelligence Cleaning Report

**Document ID**: `fe975e75-1b7f-4b00-bbad-5b2f26cf3926`  
**Original File**: `C:\Team-NARVL\doc_extension\data\originals\fe975e75-1b7f-4b00-bbad-5b2f26cf3926_messy_customers.txt`  
**Document Type**: `TXT`  
**Status**: ✅ PASSED

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Initial Entities Extracted | 60 |
| Cleaned Entities Retained | 55 |
| Quality Issues Detected | 34 |
| Cleaning Recommendations Formulated | 34 |
| Operations Approved & Committed | 34 |
| Validation Suite Status | Passed (100%) |

---

## 2. Detected Data Quality Issues

| Issue ID | Type | Field | Current Value | Severity | Evidence |
|---|---|---|---|---|---|
| `iss_miss_0c184f` | **missing** | `Email` | `None` | `medium` | Customer record 'Rajesh Kumar' is missing an Email address. |
| `iss_miss_1cae56` | **missing** | `Phone` | `None` | `medium` | Customer record 'Rajesh Kumar' is missing a contact phone number. |
| `iss_miss_3abc2e` | **missing** | `Email` | `None` | `medium` | Customer record 'Amit Patel' is missing an Email address. |
| `iss_miss_4c717a` | **missing** | `Email` | `None` | `medium` | Customer record 'Amit Patel' is missing an Email address. |
| `iss_miss_62314c` | **missing** | `Email` | `None` | `medium` | Customer record 'Rajesh Kumar' is missing an Email address. |
| `iss_miss_5d2b45` | **missing** | `Email` | `None` | `medium` | Customer record 'Priya Sharma' is missing an Email address. |
| `iss_miss_d985b8` | **missing** | `Email` | `None` | `medium` | Customer record 'Sarah Jenkins' is missing an Email address. |
| `iss_miss_61c109` | **missing** | `Email` | `None` | `medium` | Customer record 'Suresh Reddy' is missing an Email address. |
| `iss_dup_1cfb15` | **duplicate** | `CustomerName` | `Rajesh Kumar` | `low` | Exact duplicate CustomerName detected: 'Rajesh Kumar' (Page 1 and Page 1) |
| `iss_dup_b64513` | **duplicate** | `CustomerName` | `Rajesh Kumar` | `low` | Exact duplicate CustomerName detected: 'Rajesh Kumar' (Page 1 and Page 1) |
| `iss_dup_bbc1e0` | **duplicate** | `CustomerName` | `Rajesh Kumar` | `low` | Exact duplicate CustomerName detected: 'Rajesh Kumar' (Page 1 and Page 1) |
| `iss_dup_939301` | **duplicate** | `CustomerName` | `Priya Sharma` | `low` | Exact duplicate CustomerName detected: 'Priya Sharma' (Page 1 and Page 1) |
| `iss_dup_880fff` | **duplicate** | `CustomerName` | `Amit Patel` | `low` | Exact duplicate CustomerName detected: 'Amit Patel' (Page 1 and Page 1) |
| `iss_dup_be1f45` | **duplicate** | `CustomerName` | `Sarah Jenkins` | `low` | Exact duplicate CustomerName detected: 'Sarah Jenkins' (Page 1 and Page 1) |
| `iss_dup_674258` | **duplicate** | `Phone` | `9876543210` | `low` | Exact duplicate Phone detected: '9876543210' (Page 1 and Page 1) |
| `iss_dup_2c2da0` | **duplicate** | `Phone` | `+91 9123456780` | `low` | Exact duplicate Phone detected: '+91 9123456780' (Page 1 and Page 1) |
| `iss_dup_33dfed` | **duplicate** | `Phone` | `9988776655` | `low` | Exact duplicate Phone detected: '9988776655' (Page 1 and Page 1) |
| `iss_ndup_3d8e7f` | **duplicate** | `Phone` | `+1-555-234-5678` | `medium` | Near-duplicate Phone (88.9% similarity): '+1-555-234-5678' vs '555-234-5678' |
| `iss_fmt_47b10d` | **format_inconsistent** | `Phone` | `9876543210` | `low` | Inconsistent phone format '9876543210'; standard E.164 is '+91-98765-43210' |
| `iss_fmt_201f07` | **format_inconsistent** | `Phone` | `+91 9123456780` | `low` | Inconsistent phone format '+91 9123456780'; standard E.164 is '+91-91234-56780' |
| `iss_fmt_79c12c` | **format_inconsistent** | `Phone` | `9988776655` | `low` | Inconsistent phone format '9988776655'; standard E.164 is '+91-99887-76655' |
| `iss_fmt_69c7b4` | **format_inconsistent** | `Phone` | `9988776655` | `low` | Inconsistent phone format '9988776655'; standard E.164 is '+91-99887-76655' |
| `iss_fmt_92e8aa` | **format_inconsistent** | `Phone` | `9876543210` | `low` | Inconsistent phone format '9876543210'; standard E.164 is '+91-98765-43210' |
| `iss_fmt_83788f` | **format_inconsistent** | `Phone` | `+91 9123456780` | `low` | Inconsistent phone format '+91 9123456780'; standard E.164 is '+91-91234-56780' |
| `iss_fmt_ff1ae6` | **format_inconsistent** | `Phone` | `555-234-5678` | `low` | Inconsistent phone format '555-234-5678'; standard E.164 is '+1-555-234-5678' |
| `iss_fmt_b8512b` | **format_inconsistent** | `Phone` | `9440123456` | `low` | Inconsistent phone format '9440123456'; standard E.164 is '+91-94401-23456' |
| `iss_datefmt_079cd9` | **format_inconsistent** | `Date` | `15/03/2023` | `low` | Inconsistent date format '15/03/2023'; ISO 8601 standard is '2023-03-15' |
| `iss_datefmt_5550d8` | **format_inconsistent** | `Date` | `04-20-2022` | `low` | Inconsistent date format '04-20-2022'; ISO 8601 standard is '2022-04-20' |
| `iss_datefmt_e2aca0` | **format_inconsistent** | `Date` | `12/01/2024` | `low` | Inconsistent date format '12/01/2024'; ISO 8601 standard is '2024-01-12' |
| `iss_datefmt_29fa28` | **format_inconsistent** | `Date` | `15/03/2024` | `low` | Inconsistent date format '15/03/2024'; ISO 8601 standard is '2024-03-15' |
| `iss_datefmt_f04c8e` | **format_inconsistent** | `Date` | `20/04/2024` | `low` | Inconsistent date format '20/04/2024'; ISO 8601 standard is '2024-04-20' |
| `iss_datefmt_7dff5c` | **format_inconsistent** | `Date` | `05-11-2023` | `low` | Inconsistent date format '05-11-2023'; ISO 8601 standard is '2023-11-05' |
| `iss_sem_4fb77f` | **semantic** | `City` | `HYD` | `low` | Semantic variant 'HYD' detected for canonical city 'Hyderabad' |
| `iss_sem_a74585` | **semantic** | `City` | `Hyd` | `low` | Semantic variant 'Hyd' detected for canonical city 'Hyderabad' |

---

## 3. Cleaning Recommendations & Risk Governance

| Recommendation ID | Action | From -> To | Confidence | Risk Level | Loss Score | Status |
|---|---|---|---|---|---|---|
| `iss_miss_0c184f` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_1cae56` | `fill` | {'field': 'Phone', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_3abc2e` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_4c717a` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_62314c` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_5d2b45` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_d985b8` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_61c109` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_dup_1cfb15` | `deduplicate` | {'duplicate_record': 'Rajesh Kumar', 'keep_canonical': 'Rajesh Kumar'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_b64513` | `deduplicate` | {'duplicate_record': 'Rajesh Kumar', 'keep_canonical': 'Rajesh Kumar'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_bbc1e0` | `deduplicate` | {'duplicate_record': 'Rajesh Kumar', 'keep_canonical': 'Rajesh Kumar'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_939301` | `deduplicate` | {'duplicate_record': 'Priya Sharma', 'keep_canonical': 'Priya Sharma'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_880fff` | `deduplicate` | {'duplicate_record': 'Amit Patel', 'keep_canonical': 'Amit Patel'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_be1f45` | `deduplicate` | {'duplicate_record': 'Sarah Jenkins', 'keep_canonical': 'Sarah Jenkins'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_674258` | `deduplicate` | {'duplicate_record': '9876543210', 'keep_canonical': '9876543210'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_2c2da0` | `deduplicate` | {'duplicate_record': '+91 9123456780', 'keep_canonical': '+91 9123456780'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_33dfed` | `deduplicate` | {'duplicate_record': '9988776655', 'keep_canonical': '9988776655'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_ndup_3d8e7f` | `deduplicate` | {'duplicate_record': '+1-555-234-5678', 'keep_canonical': '555-234-5678'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_fmt_47b10d` | `standardize` | `9876543210` -> `+91-98765-43210` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_201f07` | `standardize` | `+91 9123456780` -> `+91-91234-56780` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_79c12c` | `standardize` | `9988776655` -> `+91-99887-76655` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_69c7b4` | `standardize` | `9988776655` -> `+91-99887-76655` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_92e8aa` | `standardize` | `9876543210` -> `+91-98765-43210` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_83788f` | `standardize` | `+91 9123456780` -> `+91-91234-56780` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_ff1ae6` | `standardize` | `555-234-5678` -> `+1-555-234-5678` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_b8512b` | `standardize` | `9440123456` -> `+91-94401-23456` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_datefmt_079cd9` | `standardize` | `15/03/2023` -> `2023-03-15` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_5550d8` | `standardize` | `04-20-2022` -> `2022-04-20` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_e2aca0` | `standardize` | `12/01/2024` -> `2024-01-12` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_29fa28` | `standardize` | `15/03/2024` -> `2024-03-15` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_f04c8e` | `standardize` | `20/04/2024` -> `2024-04-20` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_7dff5c` | `standardize` | `05-11-2023` -> `2023-11-05` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_sem_4fb77f` | `standardize` | `HYD` -> `Hyderabad` | 0.98 | `low` | 0.15 | ✅ Approved |
| `iss_sem_a74585` | `standardize` | `Hyd` -> `Hyderabad` | 0.98 | `low` | 0.15 | ✅ Approved |

---

## 4. Automated Validation Checks

- ✅ **required_fields_present** (`ALL`): All extracted entity categories preserved in cleaned output
- ✅ **valid_date_format_iso8601** (`Date`): All 10 dates conform to ISO-8601
- ✅ **valid_phone_format_e164** (`Phone`): All 9 phones conform to valid E.164/standard formats
- ✅ **valid_email_regex_rfc** (`Email`): All 3 email addresses match RFC standard
- ✅ **no_unintended_duplicates** (`ALL`): Zero duplicate entities detected in cleaned output
- ✅ **volumetric_loss_ceiling** (`Dataset`): Volumetric loss is 8.3% (Limit <= 15.0%)
