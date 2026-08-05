# Phase 5 - LDA topic model

Generated: 2026-08-05T20:55:57.942202+00:00
Final model: K=14, seed=42, c_v coherence=0.554

## Document unit

One document per (PM, sitting date): 296 documents after dropping 2 under 50 words. Raw contributions range from one-word interjections to multi-page statements, too uneven for stable per-document topic distributions.

## Preprocessing

Standard English stopwords + HANSARD_STOPWORDS (parliamentary address vocabulary: hon, right, friend, gentleman, house...) + FILLER_STOPWORDS (politeness formulas found polluting the first LDA run: thank, raising, really, important, grateful, certainly...). Bigrams via gensim Phrases. Dictionary filtered to words in >=3 documents and <=50% of documents.

## K selection log

No clean coherence plateau across K=5..30 (see lda_coherence_sweep.csv). K was picked by hand after inspecting topic word lists, not by maximizing coherence alone:

| Step | K | c_v coherence |
|---|---|---|
| Initial sweep, K=5..30, before filler stopword cleaning | 15 | 0.452 |
| Refinement after adding FILLER_STOPWORDS, K=15 | 15 | 0.503 |
| Refinement after adding FILLER_STOPWORDS, K=10 | 10 | 0.400 |
| Refinement after adding FILLER_STOPWORDS, K=12 | 12 | 0.500 |
| Refinement after adding FILLER_STOPWORDS, K=13 | 13 | 0.448 |
| Refinement after adding FILLER_STOPWORDS, K=14 (final) | 14 | 0.554 |

## Known limitation: Ukraine/security topic duplication

Topics 0 and 1 are both Ukraine/Russia/NATO/security, with substantial vocabulary overlap, at every K tested (10, 12, 13, 14, 15) - this did not resolve by lowering K, and lowering K instead merged genuinely unrelated topics together (e.g. Northern Ireland with Israel/Gaza at K=10). Read as a real structural feature of the corpus: PMs discuss Russia/Ukraine across distinct sub-periods (the 2022 invasion, ongoing military aid, NATO summits) with different vocabulary each time, not as a preprocessing or K artifact to fix.

## Topics (K=14)

- **Topic 0**: ukraine, nato, russia, russian, putin, sanctions, allies, ukrainian, money, countries, ukrainians, defence
- **Topic 1**: ukraine, security, countries, allies, russia, global, peace, defence, europe, china, international, houthis
- **Topic 2**: hs, project, rail, buses, north, plans, transport, route, line, costs, constituency, conservatives
- **Topic 3**: eu, conservatives, relation, businesses, impact, yesterday, war, business, deals, reform, agreement, huge
- **Topic 4**: report, inquiry, community, conservatives, justice, affected, failure, victims, truth, sir_martin, got, relation
- **Topic 5**: virus, businesses, testing, crisis, restrictions, areas, disease, nhs_test, trace, approach, alas, covid
- **Topic 6**: northern_ireland, eu, agreement, brexit, parliament, election, free_trade, trade, friends, voted, october, union
- **Topic 7**: pandemic, schools, vaccine, vaccines, covid, virus, disease, tests, education, testing, businesses, possible
- **Topic 8**: cop, countries, coal, private_sector, commitment, power, net_zero, glasgow, climate_change, summit, commitments, department
- **Topic 9**: pandemic, covid, pay, job, report, nurses, inquiry, putting, social_care, staff, possible, approach
- **Topic 10**: region, israel, iran, hamas, aid, gaza, allies, action, partners, law, attack, escalation
- **Topic 11**: afghanistan, taliban, afghan, military, lives, friends, kabul, forces, nato, mission, efforts, threat
- **Topic 12**: conservatives, budget, reform, minister, plans, communities, labour_government, left, services, delivered, committed, constituency
- **Topic 13**: process, relation, information, peter_mandelson, told, review, asked, appointment, decision, clearance, got, security_vetting
