# NARVL Document Intelligence Cleaning Report

**Document ID**: `c94c4872-e531-45cc-99dc-90d908147e53`  
**Original File**: `C:\Users\Anirudh Potukuchi\Desktop\Team-NARVL\doc_extension\data\originals\c94c4872-e531-45cc-99dc-90d908147e53_messy_customers.txt`  
**Document Type**: `TXT`  
**Status**: ✅ PASSED

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Initial Entities Extracted | 54 |
| Cleaned Entities Retained | 54 |
| Quality Issues Detected | 25 |
| Cleaning Recommendations Formulated | 25 |
| Operations Approved & Committed | 22 |
| Validation Suite Status | Passed (100%) |

---

## 2. Detected Data Quality Issues

| Issue ID | Type | Field | Current Value | Severity | Evidence |
|---|---|---|---|---|---|
| `iss_miss_5951ff` | **missing** | `Email` | `None` | `medium` | Customer record 'Rajesh Kumar' is missing an Email address. |
| `iss_miss_43c07d` | **missing** | `Email` | `None` | `medium` | Customer record 'Priya Sharma' is missing an Email address. |
| `iss_dup_a25588` | **duplicate** | `Phone` | `9876543210` | `low` | Exact duplicate Phone detected: '9876543210' (Page 1 and Page 1) |
| `iss_dup_3475c9` | **duplicate** | `Phone` | `+91 9123456780` | `low` | Exact duplicate Phone detected: '+91 9123456780' (Page 1 and Page 1) |
| `iss_dup_69acd7` | **duplicate** | `Phone` | `9988776655` | `low` | Exact duplicate Phone detected: '9988776655' (Page 1 and Page 1) |
| `iss_ndup_06e202` | **duplicate** | `Phone` | `+1-555-234-5678` | `medium` | Near-duplicate Phone (88.9% similarity): '+1-555-234-5678' vs '555-234-5678' |
| `iss_fmt_98a75c` | **format_inconsistent** | `Phone` | `9876543210` | `low` | Inconsistent phone format '9876543210'; standard E.164 is '+91-98765-43210' |
| `iss_fmt_a8dbb0` | **format_inconsistent** | `Phone` | `+91 9123456780` | `low` | Inconsistent phone format '+91 9123456780'; standard E.164 is '+91-91234-56780' |
| `iss_fmt_8fabad` | **format_inconsistent** | `Phone` | `9988776655` | `low` | Inconsistent phone format '9988776655'; standard E.164 is '+91-99887-76655' |
| `iss_fmt_012588` | **format_inconsistent** | `Phone` | `9988776655` | `low` | Inconsistent phone format '9988776655'; standard E.164 is '+91-99887-76655' |
| `iss_fmt_fe17af` | **format_inconsistent** | `Phone` | `9876543210` | `low` | Inconsistent phone format '9876543210'; standard E.164 is '+91-98765-43210' |
| `iss_fmt_ad219a` | **format_inconsistent** | `Phone` | `+91 9123456780` | `low` | Inconsistent phone format '+91 9123456780'; standard E.164 is '+91-91234-56780' |
| `iss_fmt_62e214` | **format_inconsistent** | `Phone` | `555-234-5678` | `low` | Inconsistent phone format '555-234-5678'; standard E.164 is '+1-555-234-5678' |
| `iss_fmt_9eeeeb` | **format_inconsistent** | `Phone` | `9440123456` | `low` | Inconsistent phone format '9440123456'; standard E.164 is '+91-94401-23456' |
| `iss_datefmt_62d76b` | **format_inconsistent** | `Date` | `15/03/2023` | `low` | Inconsistent date format '15/03/2023'; ISO 8601 standard is '2023-03-15' |
| `iss_datefmt_0f7e91` | **format_inconsistent** | `Date` | `04-20-2022` | `low` | Inconsistent date format '04-20-2022'; ISO 8601 standard is '2022-04-20' |
| `iss_datefmt_31e3bf` | **format_inconsistent** | `Date` | `12/01/2024` | `low` | Inconsistent date format '12/01/2024'; ISO 8601 standard is '2024-01-12' |
| `iss_datefmt_508852` | **format_inconsistent** | `Date` | `15/03/2024` | `low` | Inconsistent date format '15/03/2024'; ISO 8601 standard is '2024-03-15' |
| `iss_datefmt_a94ae6` | **format_inconsistent** | `Date` | `20/04/2024` | `low` | Inconsistent date format '20/04/2024'; ISO 8601 standard is '2024-04-20' |
| `iss_datefmt_5ce18d` | **format_inconsistent** | `Date` | `05-11-2023` | `low` | Inconsistent date format '05-11-2023'; ISO 8601 standard is '2023-11-05' |
| `iss_confl_c31ba0` | **conflict** | `Phone` | `+1-555-234-5678` | `high` | Conflicting Phone values for 'Sarah Jenkins': +1-555-234-5678, +91-98765-43210 |
| `iss_confl_7e7bb5` | **conflict** | `City` | `San Francisco` | `high` | Conflicting City values for 'Sarah Jenkins': San Francisco, Hyderabad |
| `iss_confl_cc6639` | **conflict** | `Phone` | `+91-94401-23456` | `high` | Conflicting Phone values for 'Suresh Reddy': +91-94401-23456, +91-91234-56780 |
| `iss_sem_42432b` | **semantic** | `City` | `HYD` | `low` | Semantic variant 'HYD' detected for canonical city 'Hyderabad' |
| `iss_sem_d1e5fa` | **semantic** | `City` | `Hyd` | `low` | Semantic variant 'Hyd' detected for canonical city 'Hyderabad' |

---

## 3. Cleaning Recommendations & Risk Governance

| Recommendation ID | Action | From -> To | Confidence | Risk Level | Loss Score | Status |
|---|---|---|---|---|---|---|
| `iss_miss_5951ff` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_miss_43c07d` | `fill` | {'field': 'Email', 'action': 'impute_or_flag'} | 0.70 | `medium` | 0.35 | ✅ Approved |
| `iss_dup_a25588` | `deduplicate` | {'duplicate_record': '9876543210', 'keep_canonical': '9876543210'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_3475c9` | `deduplicate` | {'duplicate_record': '+91 9123456780', 'keep_canonical': '+91 9123456780'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_dup_69acd7` | `deduplicate` | {'duplicate_record': '9988776655', 'keep_canonical': '9988776655'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_ndup_06e202` | `deduplicate` | {'duplicate_record': '+1-555-234-5678', 'keep_canonical': '555-234-5678'} | 0.88 | `medium` | 0.50 | ✅ Approved |
| `iss_fmt_98a75c` | `standardize` | `9876543210` -> `+91-98765-43210` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_a8dbb0` | `standardize` | `+91 9123456780` -> `+91-91234-56780` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_8fabad` | `standardize` | `9988776655` -> `+91-99887-76655` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_012588` | `standardize` | `9988776655` -> `+91-99887-76655` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_fe17af` | `standardize` | `9876543210` -> `+91-98765-43210` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_ad219a` | `standardize` | `+91 9123456780` -> `+91-91234-56780` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_62e214` | `standardize` | `555-234-5678` -> `+1-555-234-5678` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_fmt_9eeeeb` | `standardize` | `9440123456` -> `+91-94401-23456` | 0.95 | `low` | 0.20 | ✅ Approved |
| `iss_datefmt_62d76b` | `standardize` | `15/03/2023` -> `2023-03-15` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_0f7e91` | `standardize` | `04-20-2022` -> `2022-04-20` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_31e3bf` | `standardize` | `12/01/2024` -> `2024-01-12` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_508852` | `standardize` | `15/03/2024` -> `2024-03-15` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_a94ae6` | `standardize` | `20/04/2024` -> `2024-04-20` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_datefmt_5ce18d` | `standardize` | `05-11-2023` -> `2023-11-05` | 0.95 | `low` | 0.10 | ✅ Approved |
| `iss_confl_c31ba0` | `merge` | `+1-555-234-5678` -> `+91-98765-43210` | 0.60 | `high` | 0.90 | ⏸️ Review Needed |
| `iss_confl_7e7bb5` | `merge` | `San Francisco` -> `Hyderabad` | 0.60 | `high` | 0.90 | ⏸️ Review Needed |
| `iss_confl_cc6639` | `merge` | `+91-94401-23456` -> `+91-91234-56780` | 0.60 | `high` | 0.90 | ⏸️ Review Needed |
| `iss_sem_42432b` | `standardize` | `HYD` -> `Hyderabad` | 0.98 | `low` | 0.15 | ✅ Approved |
| `iss_sem_d1e5fa` | `standardize` | `Hyd` -> `Hyderabad` | 0.98 | `low` | 0.15 | ✅ Approved |

---

## 4. Automated Validation Checks

- ✅ **required_fields_present** (`ALL`): All extracted entity categories preserved in cleaned output
- ✅ **valid_date_format_iso8601** (`Date`): All 10 dates conform to ISO-8601
- ✅ **valid_phone_format_e164** (`Phone`): All 9 phones conform to valid E.164/standard formats
- ✅ **valid_email_regex_rfc** (`Email`): All 3 email addresses match RFC standard
- ✅ **no_unintended_duplicates** (`ALL`): Zero duplicate entities detected in cleaned output
- ✅ **volumetric_loss_ceiling** (`Dataset`): Volumetric loss is 0.0% (Limit <= 15.0%)
