# Dataset Verification (benchmark v1 candidates)

Verified 2026-09-27 from primary sources (HF dataset cards, HF Hub API, repo loading scripts, linked papers).
Claim → citation: every factual claim carries its source URL inline. Kaggle competition page
(https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) could not be fetched at
verification time, so Kaggle-original numbers below come from the HF mirror's loading script and card,
which cite that competition as homepage.

## A. Jigsaw Toxic Comment Classification

- Sources: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge (homepage per card),
  HF mirror https://huggingface.co/datasets/google/jigsaw_toxicity_pred,
  loading script https://huggingface.co/datasets/google/jigsaw_toxicity_pred/resolve/main/jigsaw_toxicity_pred.py,
  metadata https://huggingface.co/api/datasets/google/jigsaw_toxicity_pred
- 1. License: card metadata `cc0-1.0` (https://huggingface.co/api/datasets/google/jigsaw_toxicity_pred);
  card text: dataset under CC0, underlying comment text under Wikipedia CC-BY-SA-3.0
  (https://huggingface.co/datasets/google/jigsaw_toxicity_pred). Permits benchmark/eval use
  (CC0, attribution-style care only for comment text).
- 2. Provenance: Wikipedia Talk-page comments labeled by human raters for toxic behavior
  (https://huggingface.co/datasets/google/jigsaw_toxicity_pred). Built for the 2017–18 Jigsaw/Kaggle
  competition (homepage link on card). Human labels, not synthetic.
- 3. Schema (HF mirror; `ClassLabel ["false","true"]` per label):
  `comment_text` (string), `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
  (https://huggingface.co/api/datasets/google/jigsaw_toxicity_pred).
  Splits/rows: train 159,571; test 63,978 (same API endpoint).
  Note: card example shows an `id` field, but the loading script's `features` emit only
  `comment_text` + 6 labels (id used only as yield key)
  (https://huggingface.co/datasets/google/jigsaw_toxicity_pred/resolve/main/jigsaw_toxicity_pred.py).
- 4. Labels: 6 binary labels, all 0/1 in the HF mirror: toxic, severe_toxic, obscene, threat,
  insult, identity_hate (card: "value of 0 … or 1 … classifying the comment", each)
  (https://huggingface.co/datasets/google/jigsaw_toxicity_pred).
  Kaggle-original representation differs: `test_labels.csv` uses `-1` for unscored rows; the HF
  script merges `test.csv`+`test_labels.csv` and drops rows with `toxic == -1`
  (https://huggingface.co/datasets/google/jigsaw_toxicity_pred/resolve/main/jigsaw_toxicity_pred.py).
  So Kaggle raw labels are ternary (-1/0/1); HF mirror is binary-only after filtering.
- 5. Splits: train = Kaggle `train.csv` (fully labeled); test = labeled subset of Kaggle test set
  (63,978 of the larger Kaggle test pool). Canonical eval split: **test** (competition scoring set).
- 6. Imbalance: severe skew is inherent to the task — `severe_toxic`, `threat`, `identity_hate` are
  rare classes (single-digit/low-single-digit % in the Kaggle train set, widely reported in
  competition literature; not quantified on the card). Card's bias note: comments containing
  swear/insult words are likely classified toxic "regardless of tone or intent … e.g.
  humorous/self-deprecating", risking bias against vulnerable minority groups
  (https://huggingface.co/datasets/google/jigsaw_toxicity_pred).
- 7. Contamination: HIGH. 2017–18 competition, Wikipedia-sourced, one of the most-used public
  toxicity benchmarks; near-certain presence in LLM pretraining/fine-tuning corpora. Treat as
  leaked-by-default; useful for relative comparisons only.
- 8. Duplicates/quality: card lists no dedup info ("More Information Needed" in most curation
  fields). Known externally (not on card): train/test overlap and duplicate comments reported in
  competition write-ups — verify independently before trusting absolute scores.

## B. BANKING77 (PolyAI)

- Sources: https://huggingface.co/datasets/PolyAI/banking77,
  metadata https://huggingface.co/api/datasets/PolyAI/banking77,
  split/feature spec https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json,
  paper https://arxiv.org/abs/2003.04807, repo https://github.com/PolyAI-LDN/task-specific-datasets
- 1. License: `cc-by-4.0` (metadata + card "Creative Commons Attribution 4.0 International")
  (https://huggingface.co/datasets/PolyAI/banking77). Permits benchmark/eval with attribution.
- 2. Provenance: PolyAI (https://github.com/PolyAI-LDN/task-specific-datasets), Casanueva et al. 2020
  (https://arxiv.org/abs/2003.04807). Card tags: annotations_creators + language_creators =
  `expert-generated` (https://huggingface.co/api/datasets/PolyAI/banking77); card body leaves
  collection/annotation process as "[More Information Needed]".
- 3. Schema: fields `text` (string), `label` (ClassLabel, 77 classes, 0–76)
  (https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json).
  Splits/rows: train 10,003; test 3,080; total 13,083 (card + dataset_infos.json).
  Test is exactly 40 examples × 77 intents (3080/77 = 40).
- 4. Labels: 77 intents; full id→name map is in the card's label table
  (https://huggingface.co/datasets/PolyAI/banking77) and machine-readable in
  `dataset_infos.json` `features.label.names`
  (https://huggingface.co/datasets/PolyAI/banking77/resolve/main/dataset_infos.json).
  Note two data-quality quirks in the names list: `"Refund_not_showing_up"` (capital R, breaks
  snake_case convention) and `"reverted_card_payment?"` (trailing `?`) — quote/keep verbatim.
- 5. Splits: canonical eval split is **test** (3,080). Card's own stats table is train/test only
  (no validation split).
- 6. Imbalance: mild. Train ≈ 130 examples/intent (10,003/77); test exactly balanced at 40/intent.
  Fine-grained confusable pairs (e.g. card/top-up/transfer families) are the difficulty, not skew.
- 7. Contamination: MEDIUM-HIGH. 2020 dataset, extremely popular intent-classification benchmark
  (12k+ downloads/mo per card); likely seen during pretraining. Age + fame argue for treating it as
  possibly leaked; still fine for relative model comparison.
- 8. Duplicates/quality: card notes none ("[More Information Needed]" on biases/limitations).
  No viewer on HF (viewer disabled for this dataset).

## C. NVIDIA HelpSteer2

- Sources: https://huggingface.co/datasets/nvidia/HelpSteer2,
  README https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/README.md,
  metadata https://huggingface.co/api/datasets/nvidia/HelpSteer2,
  paper https://arxiv.org/abs/2406.08673, preference paper https://arxiv.org/abs/2410.01257
- 1. License: `cc-by-4.0` — confirmed in metadata, README header, and body ("open-source
  Helpfulness Dataset (CC-BY-4.0)") (https://huggingface.co/datasets/nvidia/HelpSteer2).
  Permits benchmark/eval with attribution.
- 2. Provenance: NVIDIA + Scale AI. Prompts: >95% ShareGPT user turns (assistant turns stripped),
  ~5% Scale-AI-written (enterprise use cases). Responses: 2/prompt from mixed sources — Nemotron-2
  18.9%, Nemotron-3 40.4%, Nemotron-4 26.9%, Mixtral-8x7B-Instruct 7.9%, Scale AI humans 5.9%.
  Ratings: ~1,000 US-based annotators via Scale AI, 3–5 annotators/sample (mean 3.41), final score =
  mean of 3 most-agreeing annotators rounded to int; ~50% of annotations filtered out in QA; samples
  kept only if helpfulness spread ≤ 2 (all per
  https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/README.md and
  https://arxiv.org/abs/2406.08673).
- 3. Schema: `prompt` (string), `response` (string), `helpfulness`, `correctness`, `coherence`,
  `complexity`, `verbosity` (all int 0–4). Splits/rows: train 20,324; validation 1,038; total
  21,362 = 10,681 prompts × 2 responses (README `Using the Huggingface Datasets` section; viewer
  shows train ~20.3k / validation ~1.04k, consistent). Extra dirs (not default config):
  `preference/` (pairwise preference strength −3…+3 + justifications) and `disagreements/`
  (per-annotator label lists), added 1 Oct 2024
  (https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/README.md).
- 4. Label definitions (card gives one-liners; full 0–4 anchors live in paper Appendix G,
  https://ar5iv.org/html/2406.08673). Card text, quoted verbatim
  (https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/main/README.md):
  1. "Helpfulness: Overall helpfulness of the response to the prompt."
  2. "Correctness: Inclusion of all pertinent facts without errors."
  3. "Coherence: Consistency and clarity of expression."
  4. "Complexity: Intellectual depth required to write response (i.e. whether the response can be written by anyone with basic language competency or requires deep domain expertise)."
  5. "Verbosity: Amount of detail included in the response, relative to what is asked for in the prompt."
  Paper Appendix G anchors (quoted/paraphrased tightly from https://ar5iv.org/html/2406.08673):
  Helpfulness 4 "extremely helpful and completely aligned with the spirit of what the prompt was
  asking for" → 0 "not useful or helpful at all… completely missed the essence".
  Correctness 4 "completely correct and accurate… no necessary details missing and without false,
  misleading, or hallucinated information" → 0 "completely incorrect. All information… wrong, false
  or hallucinated". Coherence 4 "perfectly clear and self-consistent… no contradictory assertions"
  → 0 "completely incomprehensible". Complexity 4 "Expert… specific and technically relevant
  vocabulary" (lawyer/scientist/engineer/doctor register) → 0 "Basic… interpretable by children".
  Verbosity 4 "particularly lengthy, wordy, and/or extensive with extra details given what the prompt
  requested" → 0 "short, to the point, and the most concise it can be".
  Rubric guidance worth quoting: "Even if two responses have exactly the same length, one can be rated
  as verbose and the other as succinct depending on the prompting context" and "A response that gives
  an incorrect answer can still be succinct" (same source). Note: complexity/verbosity are
  descriptive (higher ≠ better); helpfulness/correctness/coherence are quality (higher = better).
  Annotators also flag PII / non-English / coding / assistant-identity tasks as invalid and skip them
  (Appendix G.2, same source).
- 5. Splits: canonical eval split is **validation** (1,038). Train (20,324) is the reward-model
  training set; paper reports RewardBench/AlpacaEval externally, not a held-out HF test split.
- 6. Imbalance/stats (paper Table 2, https://ar5iv.org/html/2406.08673): means —
  helpfulness 2.87, correctness 2.96, coherence 3.64 (ceiling-skewed; "generating coherent responses
  is no longer a challenge"), complexity 1.70, verbosity 2.00. Correctness↔helpfulness r=0.943
  (correctness dominates helpfulness); verbosity↔helpfulness r=0.060 (near-independent by design).
  ~29% of prompts are multi-turn (prompt field packs prior turns with `<extra_id_1>` separators).
- 7. Contamination: MEDIUM. Released June 2024 (https://arxiv.org/abs/2406.08673); prompts derive
  from ShareGPT (public, widely crawled) but responses are model-generated for this dataset and
  labels are original. Prompt text may be in pretraining; (prompt, response, rating) triples are
  almost certainly not.
- 8. Duplicates/quality: consecutive samples share a prompt by design (pairs for preference
  training). ~50% of annotations excluded; retained helpfulness κ (quadratic-weighted Cohen) 0.706,
  correctness 0.715, but coherence 0.428 / complexity 0.427 / verbosity 0.548 — expect noise on the
  non-helpfulness axes (https://ar5iv.org/html/2406.08673). Coding prompts heuristically filtered;
  some residual inconsistency in multi-turn completions acknowledged (Appendix F, same source).

## D. PersonalFinance_v2 (CANDIDATE — caution)

- Sources: https://huggingface.co/datasets/Akhil-Theerthala/PersonalFinance_v2,
  metadata https://huggingface.co/api/datasets/Akhil-Theerthala/PersonalFinance_v2
- 1. License: `apache-2.0` (https://huggingface.co/api/datasets/Akhil-Theerthala/PersonalFinance_v2).
  Permits benchmark/eval use. (Note: Compact sibling is MIT — §E; license differs across versions.)
- 2. Provenance: solo author (Akhil Theerthala); won first prize, Reasoning Datasets Competition
  (Bespoke Labs × HF × Together.AI, Apr–May 2025) (card §1). Queries: real user posts from
  r/personalfinance, restricted to pre-mid-2023 posts "to ensure full compliance and ethical data
  handling" after Reddit's API/ToS change. Everything downstream is synthetic/LLM-generated —
  there are NO human gold labels (card §§3–5).
- 3. Schema (viewer columns): `category`, `query`, `chain_of_thought`, `response`. Note: card prose
  calls the label column `query-category`, but the actual column is `category` — use `category`.
  Splits/rows: **train only**, 7,036 rows (card footer "Number of rows: 7,036"; viewer "train ·
  7.04k rows"; category table sums to 7,036). No validation/test split.
- 4. Labels: `category` ∈ 8 domains with counts (card §2 table):
  Debt Management & Credit 1,989; Retirement Planning 1,169; Investing & Wealth Building 1,156;
  Tax Planning & Optimization 1,019; Budgeting & Cash Flow Management 986;
  Insurance & Risk Management 390; Savings & Emergency Funds 259; Estate Planning & Legacy 68.
  (No per-class annotation rubric — categories are author-defined buckets, LLM-assigned.)
- 5. Splits: no canonical eval split exists (train-only). Any benchmark use requires carving your
  own held-out split from the 7,036 rows.
- 6. Imbalance: heavy. Debt (28%) dominates; Estate Planning has 68 rows (~1%). Avg CoT length
  ~7.4–8.1k chars, response ~1.9–2.5k chars per category (card §2 table).
- 7. Contamination: LOW-MEDIUM for the triples (synthetic reasoning/response text, Apr–Aug 2025,
  niche dataset, ~104 downloads/mo). Query text itself is pre-2023 Reddit — likely in pretraining.
  Dataset too new for most model cutoffs, but queries are public text.
- 8. Duplicates/quality + construction pipeline (card's own words — quote, §§3–5):
  filtering: "we implemented a filtering process using an LLM-as-a-jury approach. Multiple models,
  including Gemini 2.0 Flash, Phi-4 14B, and Gemma-3-27B, evaluated potential posts, filtering out
  noise and retaining only those queries that presented a clear personal finance problem requiring
  analysis and a reasoned response." Pipeline stages: "(1) Input Query Processing, (2) Parallel
  Analysis Streams [Query Analysis + Psychological Intent Extraction], (3) RAG-based Context
  Extraction [~800k-token knowledge base] … (4) Response Generation … (5) Response Validation."
  "However, It is important to note that not every fact/piece of information is dumped into the
  reasoning chain. Instead, the reasoning chain tries to provide reference to the contextual
  information needed to answer the query." Validation: "Several LLMs, including GPT-4o-mini,
  Deepseek-v3-0324 … and Gemini-2.0-Flash, served as distinct juries".
  Card's own limitations (§9, abridged): "Model and Data Biases", "RAG Fidelity … Retrieval might
  be incomplete or inaccurate", "Psychological Intent Subjectivity", "US-centric … limited to
  US-based personal finance advice", "Validation Scope … subject to … subjectivity".
  Bottom line: `chain_of_thought`/`response` are unreviewed LLM outputs — reference-style text,
  not ground truth. Financial-advice safety review is mandatory before any eval use.

## E. PersonalFinance-CoTR-V2-Compact (CANDIDATE — caution)

- Sources: https://huggingface.co/datasets/Akhil-Theerthala/PersonalFinance-CoTR-V2-Compact,
  metadata https://huggingface.co/api/datasets/Akhil-Theerthala/PersonalFinance-CoTR-V2-Compact
- 1. License: `mit` (https://huggingface.co/api/datasets/Akhil-Theerthala/PersonalFinance-CoTR-V2-Compact)
  — DIFFERS from PersonalFinance_v2's Apache-2.0. Both permit benchmark use; record per-artifact
  license if both are used.
- 2. Provenance: same author/pipeline as §D, "condensed version … specifically adapted for training
  smaller language models (e.g., 1B–4B parameters)" (card §1). Same Reddit-query sourcing and
  LLM-jury filtering; same synthetic-generation caveats. No human labels.
- 3. Schema (viewer columns): `category`, `query`, `context_directions`, `psychological_intent`,
  `chain_of_thought`, `response` — i.e. §D's 4 fields PLUS `context_directions` (which RAG context
  was referenced) and `psychological_intent` (sentiment/emotion/intent/certainty analysis) as
  first-class columns. Splits/rows: **train only**; viewer shows 1.19k rows but the card §2 table
  totals **1,150** (297/207/199/155/152/83/42/15) — count discrepancy to resolve before use
  (likely a version skew between card text and parquet).
- 4. Labels: same 8 `category` domains as §D, counts: Debt 297, Retirement 207, Investing 199,
  Tax 155, Budgeting & Cash Flow Mgmt 152, Insurance 83, Savings 42, Estate 15.
- 5. Splits: train-only; no canonical eval split — carve your own.
- 6. Imbalance: same skew as §D (Debt 26%, Estate 1.3%). Stated compression vs v2: CoT tokens −71%,
  response tokens −22.7% (card §1); avg CoT ~2.1–2.3k chars, response ~1.6–1.9k chars (card §2).
- 7. Contamination: same as §D (LOW-MEDIUM; synthetic triples, public Reddit queries).
- 8. Duplicates/quality + diffs vs D: card's stated difference — "the significant reduction in the
  length and exploratory nature of the reasoning phase (chain-of-thought) and the omission of
  justifications where possible in the final response, aiming for conciseness" (card §2). Same §D
  limitations apply (LLM-generated, US-centric, RAG fidelity, intent subjectivity). Extra caution:
  shorter CoTs omit justification — weaker as reference reasoning. Plus the 1,150-vs-1.19k row-count
  mismatch and MIT-vs-Apache license difference noted above.

## Recommendations (benchmark v1)

- A. Jigsaw: USE test split (63,978) for multi-label toxicity classification (fields
  `comment_text` → 6 binary labels); report per-label F1 given rarity of severe_toxic/threat/
  identity_hate; treat scores as relative-only (high contamination); spot-check for train/test
  duplicates before publishing absolute numbers.
- B. BANKING77: USE test split (3,080, 40/intent) for 77-way intent classification (fields
  `text` → `label` + name map from card/dataset_infos.json); keep the two odd label strings
  verbatim; attribute CC-BY-4.0; relative-scores-only (likely leaked).
- C. HelpSteer2: USE validation split (1,038) for reward/rating prediction on
  `prompt`+`response` → `helpfulness` (primary), `correctness`/`coherence` secondary; use
  Appendix G 0–4 anchors as the grading rubric; do NOT treat `complexity`/`verbosity` as
  quality signals; handle multi-turn `<extra_id_1>` packing.
- D. PersonalFinance_v2: CONDITIONAL — use only `query`+`category` for domain-classification
  eval on a self-carved held-out split; do NOT score against `chain_of_thought`/`response` as
  ground truth (unreviewed LLM output); manually review a sample for financial-advice safety first.
- E. PersonalFinance-CoTR-V2-Compact: PREFER OVER D for classification probing (explicit
  `psychological_intent` + `context_directions` columns enable stage-wise checks); resolve the
  1,150-vs-1.19k row mismatch first; same no-LLM-output-as-ground-truth and safety-review
  caveats as D; record MIT license separately.
