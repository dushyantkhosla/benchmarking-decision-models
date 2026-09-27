# Handoff: Decision-Model Benchmark

## Next session focus

Start a new code repository that benchmarks **JEV, Laya, and KEV** as
small/fast decision models for text-based application decisions.

The benchmark should answer a practical engineering question:

> When can a specialized decision model replace a general-purpose LLM
> call for binary adjudication, bounded classification, rubric scoring,
> and application routing?

The intended workload is **not browser automation or general agent
capability**. It is predominantly:

-   read text and answer a yes/no question about it;
-   classify text into one of several explicitly defined categories;
-   score text against an explicit bounded rubric;
-   route a user's request to the appropriate specialist based on both
    their query and contextual case-file data.

Keep the benchmark **zero-shot** initially. Do not fine-tune the
evaluated models.

------------------------------------------------------------------------

## Models to benchmark

Initial target models:

-   **JEV**
-   **Laya**
-   **KEV**
    -   include multiple KEV sizes where practical, especially 0.8B / 4B
        / 9B variants if currently available

Add one or more conventional small LLMs as baselines later, but do not
let that delay the first JEV/Laya/KEV comparison.

Model versions, inference runtimes, quantization, and hardware must be
recorded with every benchmark run. Do not silently compare different
precision/runtime configurations.

Local Apple Silicon performance is important. The benchmark should be
runnable on a Mac, ideally including models served through tools such as
Ollaya where supported.

------------------------------------------------------------------------

## Benchmark tracks

Build separate tracks rather than collapsing everything into one
composite score.

### 1. Binary adjudication

**Dataset:** Jigsaw Toxic Comment Classification

Source:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Useful mirror/reference:
https://huggingface.co/datasets/google/jigsaw_toxicity_pred

Jigsaw gives text with multiple independently labelled criteria such as
toxicity, threat, insult, obscenity, severe toxicity, and identity hate.

Transform each applicable `(text, criterion)` pair into a decision such
as:

``` text
TEXT:
<comment>

QUESTION:
Does this text contain a threat?

OPTIONS:
yes
no
```

Why this track matters:

This resembles the target workload better than generic yes/no QA. The
model is being asked to **apply a defined criterion to supplied text**.

Important:

-   inspect the exact source schema before implementation;
-   balance positive and negative cases, especially for rare labels;
-   do not use raw accuracy alone on highly imbalanced labels.

Primary metrics:

-   F1
-   precision
-   recall
-   accuracy
-   calibration/Brier score where model confidence is available

Suggested initial benchmark size: \~500 source texts,
stratified/balanced deliberately.

------------------------------------------------------------------------

### 2. Bounded multi-class classification

**Dataset:** BANKING77

Source: https://huggingface.co/datasets/PolyAI/banking77

BANKING77 contains roughly 13k online banking queries across **77
fine-grained intents**.

Example shape:

``` text
TEXT:
I am still waiting on my card.

CATEGORIES:
card_arrival
card_delivery_estimate
card_not_working
card_payment_not_recognised
...

DECISION:
<one category>
```

Why this track matters:

The categories are semantically close, making this much more
representative of application routing than trivial sentiment
classification.

Create at least two difficulty levels:

#### BANKING77 --- reduced choice

For each query, present the correct label plus a fixed number of
distractors, e.g. 10 total choices.

Distractor selection must be deterministic and reproducible.

#### BANKING77 --- full choice

Present all 77 labels.

This lets us measure how decision quality and latency change as the
bounded choice space grows.

Primary metrics:

-   accuracy
-   macro-F1
-   confusion matrix
-   latency
-   calibration if available

Suggested initial benchmark size: \~500 queries.

------------------------------------------------------------------------

### 3. Rubric scoring

**Dataset:** NVIDIA HelpSteer2

Source: https://huggingface.co/datasets/nvidia/HelpSteer2

HelpSteer2 contains prompt/response pairs with human ratings on
dimensions including:

-   helpfulness
-   correctness
-   coherence
-   complexity
-   verbosity

Ratings are bounded ordinal scores.

Transform each `(prompt, response, dimension)` into a rubric decision.

Example:

``` text
PROMPT:
<user prompt>

RESPONSE:
<assistant response>

RUBRIC:
Coherence
0 = ...
1 = ...
2 = ...
3 = ...
4 = ...

DECISION:
0 | 1 | 2 | 3 | 4
```

Do **not** invent rubric descriptions. Retrieve the dataset's actual
annotation guidance or another authoritative source and preserve the
intended semantics.

Why this track matters:

This is close to real application tasks where a model reads an artifact
and assigns a bounded score against a rubric.

Primary metrics:

-   exact-match accuracy
-   mean absolute error (MAE)
-   score distribution
-   confusion matrix
-   optionally quadratic weighted kappa if appropriate

MAE is important because ordinal errors have magnitude: predicting 3 for
a human score of 4 is less wrong than predicting 0.

Suggested initial benchmark size: \~500 prompt/response pairs. Multiple
rubric dimensions can yield several decisions per source example.

------------------------------------------------------------------------

### 4. Personal-finance domain routing

**Dataset candidate:** PersonalFinance_v2

Source:
https://huggingface.co/datasets/Akhil-Theerthala/PersonalFinance_v2

The dataset has personal-finance questions categorized into domains
resembling the routing problem for a personal-finance application.

Previously identified domains include:

-   Debt Management & Credit
-   Retirement Planning
-   Investing & Wealth Building
-   Tax Planning & Optimization
-   Budgeting & Cash Flow Management
-   Insurance & Risk Management
-   Savings & Emergency Funds
-   Estate Planning & Legacy

There is also a compact related dataset:

https://huggingface.co/datasets/Akhil-Theerthala/PersonalFinance-CoTR-V2-Compact

**Important caveat:** treat this dataset as a candidate, not pristine
ground truth. Its construction reportedly includes LLM-assisted
filtering/categorization and synthetic stages. Verify the dataset card
and provenance before using it. For the final benchmark subset, manually
inspect/freeze labels or otherwise establish defensible gold labels.

#### PF-Routing Query-Only

Input:

``` text
QUERY:
<personal finance question>

ROUTES:
Debt Management & Credit
Retirement Planning
Investing & Wealth Building
...
```

Output: exactly one route.

This tests ordinary domain routing from query text.

Suggested size: \~500 manually reviewed cases.

Primary metrics:

-   accuracy
-   macro-F1
-   per-route precision/recall
-   confusion matrix

------------------------------------------------------------------------

### 5. Personal-finance contextual routing

This is the benchmark track most representative of the intended
application.

The real decision is:

> **case file + current user query -\> problem domain**

A user's query may be ambiguous without their financial situation.

Example:

``` text
CASE FILE:
Monthly income: €4,800
Monthly expenses: €4,100
Cash savings: €2,300
Credit-card debt: €14,000 at 19% APR
Investments: €31,000
Pension: €62,000
Mortgage: €280,000

QUERY:
I'm getting a €5,000 bonus next month. What should I do with it?

ROUTES:
Debt Management & Credit
Budgeting & Cash Flow Management
Savings & Emergency Funds
Investing & Wealth Building
Retirement Planning
...

DECISION:
Debt Management & Credit
```

The same query paired with a materially different case file may
legitimately route elsewhere.

This means query-only classification is insufficient for the intended
product architecture.

#### Proposed PF-Context difficulty groups

Create a curated benchmark extension with four groups:

**A. Query sufficient**

The query itself clearly determines the route.

Example:

> How should I pay down my €18,000 credit-card balance?

**B. Context helpful**

The query contains clues but the case file substantially improves the
decision.

**C. Context decisive**

The query is deliberately ambiguous and the correct route depends on the
case file.

Example:

> I have an extra €500 every month. Where should it go?

The route may depend on high-interest debt, emergency savings,
retirement position, cash-flow stability, etc.

**D. Distractor context**

The case file contains many irrelevant facts. The model must identify
which facts actually affect the routing decision.

This tests resistance to context noise.

#### Critical methodology rule

Do not automatically generate hundreds of synthetic profiles and call
their labels ground truth.

First define:

1.  the route taxonomy;
2.  routing policy/decision rules;
3.  precedence rules for cases where several routes are plausible;
4.  how multi-problem queries are handled;
5.  whether the benchmark requires exactly one route or allows
    abstention/multiple routes.

Only then generate or curate contextual cases.

A small, high-quality manually validated set is more useful than
thousands of weak synthetic labels.

Suggested first version: 200--500 contextual cases.

------------------------------------------------------------------------

## Optional harder track

### LegalBench

Source: https://huggingface.co/datasets/nguha/legalbench

LegalBench contains many text + criterion/question tasks with bounded
outputs, including binary legal reasoning.

This is useful as an optional **reasoning-heavy adjudication** track:
can the decision model apply a non-trivial criterion to text rather than
merely detect surface-level linguistic features?

Do not make LegalBench a blocker for benchmark v1.

------------------------------------------------------------------------

## Evaluation philosophy

Do **not** produce a single "best decision model" number as the primary
result.

The goal is to understand each model's suitability for different
application decisions.

Report results per:

-   model;
-   model size/version;
-   task/track;
-   dataset;
-   difficulty tier;
-   hardware/runtime;
-   quantization/precision.

At minimum capture:

-   task accuracy / F1 as appropriate;
-   ordinal MAE for rubric scoring;
-   confidence/calibration where supported;
-   latency per decision;
-   throughput / decisions per second;
-   peak memory/RAM where measurable;
-   model size;
-   failures/invalid outputs.

A useful engineering view will eventually be:

> decision quality vs latency vs memory footprint.

------------------------------------------------------------------------

## Confidence and fallback evaluation

A major hypothesis worth testing is:

> A small local decision model can handle high-confidence decisions
> while ambiguous cases fall back to a larger general-purpose LLM.

If the APIs expose meaningful confidence/probabilities, evaluate this
explicitly.

For confidence thresholds, report:

-   percentage of cases handled locally;
-   accuracy of locally handled cases;
-   percentage escalated;
-   error rate at each threshold.

This produces a **coverage vs error** curve.

Example question:

> What percentage of finance-routing decisions can KEV-0.8B handle while
> maintaining \<= 5% error?

Do not assume confidence values from different model families are
calibrated equivalently.

------------------------------------------------------------------------

## Common benchmark representation

Normalize datasets into a model-independent internal schema.

Suggested conceptual schema:

``` json
{
  "id": "banking77:test:123",
  "track": "classification",
  "dataset": "banking77",
  "input": {
    "text": "I am still waiting on my card."
  },
  "decision": {
    "type": "choice",
    "options": [
      "card_arrival",
      "card_not_working",
      "cash_withdrawal"
    ]
  },
  "gold": "card_arrival",
  "metadata": {
    "difficulty": "reduced-choice"
  }
}
```

For contextual finance routing:

``` json
{
  "id": "pf-context:001",
  "track": "routing",
  "dataset": "pf-context",
  "input": {
    "case_file": {
      "monthly_income": 4800,
      "monthly_expenses": 4100,
      "cash_savings": 2300,
      "credit_card_debt": 14000,
      "credit_card_apr": 0.19
    },
    "query": "I'm getting a €5,000 bonus. What should I do with it?"
  },
  "decision": {
    "type": "choice",
    "options": [
      "debt",
      "cashflow",
      "savings",
      "investing",
      "retirement"
    ]
  },
  "gold": "debt",
  "metadata": {
    "difficulty": "context-decisive"
  }
}
```

Treat these as design sketches, not a finalized schema. Keep dataset
adapters separate from model adapters.

------------------------------------------------------------------------

## Architecture direction

Prefer a simple benchmark harness with clean seams:

``` text
datasets/
    jigsaw
    banking77
    helpsteer2
    personal_finance
    pf_context

models/
    jev
    laya
    kev

core/
    schemas
    runner
    metrics

results/
```

Desired pipeline:

``` text
raw dataset
    ↓
dataset adapter
    ↓
normalized decision examples
    ↓
model adapter
    ↓
normalized prediction
    ↓
metrics
    ↓
run artifact
```

Avoid coupling dataset formatting to any one model's API.

Every model adapter should ultimately expose the same conceptual
operation:

``` text
decide(example) -> prediction
```

The normalized prediction should preserve, where available:

-   selected answer;
-   confidence/probabilities;
-   latency;
-   raw model output;
-   parse/validation errors.

------------------------------------------------------------------------

## Reproducibility

Every benchmark run should save enough metadata to reproduce it:

-   git commit;
-   timestamp;
-   dataset name/version/revision;
-   exact selected example IDs;
-   random seed;
-   model name/version;
-   model artifact/checkpoint;
-   runtime;
-   quantization/precision;
-   hardware;
-   benchmark configuration;
-   per-example predictions;
-   aggregate metrics.

Do not rely only on printed terminal summaries.

Prefer immutable JSON/JSONL result artifacts that can later feed reports
or charts.

------------------------------------------------------------------------

## Dataset hygiene

Before implementing a dataset adapter, verify:

1.  license permits the intended benchmark use;
2.  dataset provenance;
3.  exact schema;
4.  official label definitions;
5.  train/test splits;
6.  whether evaluated models may have trained on the dataset;
7.  class imbalance;
8.  duplicate/near-duplicate examples;
9.  whether labels are human, synthetic, LLM-generated, or mixed.

Record these findings in repository documentation.

Contamination is especially relevant for public benchmarks and newer
models. Results should be described as practical benchmark performance,
not necessarily clean measures of unseen generalization.

------------------------------------------------------------------------

## Benchmark v1 scope

Aim for a small useful benchmark before expanding.

Suggested v1:

  Track                        Dataset                Approx. source examples
  ---------------------------- -------------------- -------------------------
  Binary adjudication          Jigsaw                                     500
  Multi-class classification   BANKING77                                  500
  Rubric scoring               HelpSteer2                                 500
  Finance routing              PersonalFinance_v2                500 reviewed
  Contextual finance routing   PF-Context                    200--500 curated

This should yield several thousand individual decisions because some
source examples produce multiple criterion/rubric decisions.

Do not add browser-agent benchmarks to v1.

------------------------------------------------------------------------

## Key questions the benchmark should answer

1.  How do JEV, Laya, and KEV compare on binary text adjudication?
2.  How do they compare on fine-grained bounded classification?
3.  Can they reliably apply an explicit ordinal rubric?
4.  How does performance degrade as the number of available choices
    increases?
5.  Can they route a personal-finance problem using query text alone?
6.  Can they actually use structured case context when the query is
    ambiguous?
7.  Do irrelevant case-file facts degrade routing performance?
8.  How do KEV model sizes trade accuracy for latency and memory?
9.  Are model confidence scores calibrated enough for safe fallback
    routing?
10. At a chosen error threshold, what percentage of decisions can remain
    local instead of escalating to a larger LLM?

------------------------------------------------------------------------

## Decisions already made

-   Focus on **decision-model workloads**, not generic agent capability.
-   Browser tasks are not representative of the intended use case.
-   Benchmark JEV, Laya, and KEV first.
-   Keep benchmark tracks separate.
-   Use public datasets where defensible.
-   Prefer human-labelled gold data.
-   Treat PersonalFinance_v2 labels with caution and review the
    benchmark subset.
-   Build a custom contextual personal-finance routing track because
    existing query-only datasets do not capture the full product
    problem.
-   Keep v1 zero-shot.
-   Measure operational characteristics as well as prediction quality.
-   Preserve per-example results for later analysis.

------------------------------------------------------------------------

## Open decisions

Do not silently choose these without considering them:

-   implementation language and dependency stack;
-   exact model runtimes/APIs;
-   which JEV/Laya/KEV checkpoints are canonical;
-   whether Ollaya should be the common runtime or merely one adapter;
-   final finance route taxonomy;
-   single-route vs multi-route finance cases;
-   abstention support;
-   how PF-Context gold labels will be adjudicated;
-   conventional LLM baseline(s);
-   benchmark sample sizes after inspecting class distributions;
-   confidence normalization/calibration across model families;
-   licensing constraints discovered during dataset verification.

------------------------------------------------------------------------

## First actions in the new repository

1.  Initialize the repository and write a concise README stating the
    benchmark question.
2.  Verify current JEV, Laya, KEV repositories/checkpoints/APIs from
    primary sources.
3.  Verify the five candidate datasets from their primary dataset
    cards/repos, including licenses and schemas.
4.  Decide the minimal normalized `DecisionExample` and `Prediction`
    schemas.
5.  Implement **one vertical slice first**:
    -   BANKING77 adapter;
    -   one model adapter;
    -   deterministic sample selection;
    -   runner;
    -   accuracy + latency;
    -   persisted per-example results.
6.  Run it end-to-end before abstracting further.
7.  Add the remaining model adapters.
8.  Add Jigsaw and HelpSteer2.
9.  Add PersonalFinance_v2 only after inspecting its provenance and
    labels.
10. Design PF-Context taxonomy/routing policy before generating
    contextual examples.
11. Add reporting only after raw result artifacts are stable.

Prefer a tracer-bullet implementation over building a large benchmark
framework upfront.

------------------------------------------------------------------------

## Suggested skills

The next Codex session should inspect the installed skills and use
equivalents where available.

Recommended Matt Pocock flow:

-   **`grill-with-docs`** --- if important repository-level design
    questions remain once primary sources have been inspected.
-   **`to-spec`** --- convert this handoff plus verified research into
    the repository's buildable benchmark specification.
-   **`to-tickets`** --- if the implementation is large enough to span
    multiple independent slices.
-   **`implement`** --- implement each vertical slice.
-   **`tdd`** --- especially useful for dataset normalization, metrics,
    deterministic sampling, and model-adapter contracts.
-   **`code-review`** --- review each completed implementation against
    both standards and the benchmark spec.

Do not treat this handoff as a substitute for the spec. Its purpose is
to carry the decisions and research direction into the new repository.

Matt Pocock handoff skill:
https://github.com/mattpocock/skills/blob/main/skills/productivity/handoff/SKILL.md

Matt Pocock engineering flow:
https://github.com/mattpocock/skills/blob/main/skills/engineering/ask-matt/SKILL.md

------------------------------------------------------------------------

## Resume instruction for Codex

Read this document first.

Then inspect the current primary sources for JEV, Laya, KEV and the
candidate datasets before writing implementation code. Correct any stale
assumptions in this handoff explicitly.

Start with the smallest end-to-end benchmark slice rather than designing
the entire framework upfront.

The immediate milestone is:

> **Run the same reproducible BANKING77 subset through at least one
> decision model, persist every prediction, and produce classification
> accuracy plus latency.**

Once that vertical slice is trustworthy, generalize the harness to the
other models and tracks.
