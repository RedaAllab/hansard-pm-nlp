# Data README

Generated: 2026-08-05T19:04:32.879600+00:00
Corpus cutoff date: 2026-07-15

## pm_contributions.parquet

Row count: 10673

| Column | Type | Description |
|---|---|---|
| pm_name | string | PM speaking, resolved via Members API tenure window |
| pm_party | string | PM's party at time of speech (from Members API `latestParty`) |
| member_id | int64 | Members API member id |
| item_id | int64 | Hansard contribution item id |
| contribution_ext_id | string | Hansard external contribution id (unique, dedup key) |
| contribution_text | string | Full contribution text |
| hansard_section | string | Hansard structural tag |
| debate_section | string | Debate title; 'Engagements' identifies PMQs |
| debate_section_id | int64 | Hansard debate section id |
| debate_section_ext_id | string | Hansard external debate section id |
| sitting_date | datetime64[ns] | Date of the sitting |
| section | string | Chamber section (Commons Chamber) |
| house | string | House (Commons only, by extraction scope) |
| order_in_debate_section | int64 | Contribution order within the debate section |
| debate_section_order | int64 | Debate section order within the sitting |
| ingested_at | datetime64[ns] | Ingestion timestamp (UTC) |

Categorical values, `pm_name`: ['Boris Johnson', 'Keir Starmer', 'Liz Truss', 'Rishi Sunak']
Categorical values, `house`: ['Commons']

## pm_tenures.parquet

Row count: 5

| Column | Type | Description |
|---|---|---|
| pm_name | string | Commonly known PM name |
| pm_party | string | Party at time of tenure |
| member_id | int64 | Members API member id |
| tenure_start | datetime64[ns] | Date appointed PM |
| tenure_end | datetime64[ns] (nullable) | Date tenure ended; null for the sitting PM |

See PHASE0_SCOPING.md for how these tenure windows and the cutoff date were decided.
