# Data README - processed corpus

Generated: 2026-08-05T19:13:03.997621+00:00
Row count: 10673

## Cleaning applied

- `contribution_text_raw`: original text, untouched, kept for audit.
- `contribution_text`: HTML tags/entities stripped, whitespace normalized (see hansard_pm_nlp.cleaning.clean_contribution_text).
- `debate_section`: leading/trailing whitespace stripped, merging variants that were previously distinct categories (e.g. ' Covid-19 Update' vs 'Covid-19 Update').
- `is_pmqs`: True where `debate_section == 'Engagements'` (Hansard's own filing for Prime Minister's Questions).
- No rows dropped: the 5 exact-duplicate-text rows found (same PM/date/text) are distinct short interjections ('Sit down.', 'No.') with distinct contribution_ext_id and debate order, not ingestion artifacts. See Phase 2 notes in PROJECT_SUMMARY.md.

PMQs contributions: 6076 / 10673

## debate_section value counts (top 15)

- Engagements: 6076
- Covid-19 Update: 505
- Ukraine: 275
- Middle East: 265
- Sue Gray Report: 176
- Covid-19: 144
- Israel and Gaza: 138
- Priorities for Government: 134
- Prime Minister's Update: 123
- Security Vetting: 115
- G7 and NATO Summits: 110
- Afghanistan: 110
- Covid-19: Road Map: 99
- UK-EU Summit: 99
- G7 Summit: 96
