# Claim-Level Hallucination Verification Module — Project Specification

**Project:** NIT Trichy Internship — Claim-Level Hallucination Detection in RAG Systems
**Team:** Shifan Vadakkungara, Vanshaj Gupta, Sushmit Nakhate
**Supervisor:** Dr. Usha Kiruthika S, Dept. of CSE, NIT Trichy

This document is a complete build specification. It should be read alongside the
project's literature-survey/proposal report (provided separately as a PDF/DOCX).
Everything described here as "validated" has already been implemented and tested
in a Google Colab notebook — the goal now is to rebuild it as a properly
structured application with a real frontend/backend split and a polished UI,
instead of a single notebook script.

---

## 1. What This Project Is

RAG (Retrieval-Augmented Generation) systems retrieve documents and generate
answers grounded in them, but the generator can still "hallucinate" — state
things not actually supported by (or contradicted by) the retrieved evidence.

This project is **not** a RAG system itself. It is a **verification module** that
sits *after* a RAG pipeline's generator and *before* the answer is shown to the
user. It takes (a) the generated answer and (b) the retrieved passages, and
returns a structured, claim-by-claim verdict: which parts of the answer are
actually backed by evidence, which are unsupported (model invented them), and
which directly contradict the evidence.

Existing approaches score an entire answer with one number (e.g. RAGAS). This
project's contribution is keeping the verdict **per-claim** and **three-way**
(supported / unsupported / contradicted) instead of one aggregate score, and
doing so with a **deterministic encoder model**, not another LLM call — see
Section 3 of the project report for the literature justification.

---

## 2. High-Level Architecture (target)

```
┌─────────────┐      ┌──────────────────────────────────────────┐      ┌────────────┐
│  Frontend   │ ───▶ │              Backend API                  │ ───▶ │  ML models │
│ (web UI)    │ ◀─── │  (FastAPI, wraps the 4-stage pipeline)     │ ◀─── │ (local, GPU)│
└─────────────┘      └──────────────────────────────────────────┘      └────────────┘
```

- **Backend**: Python (FastAPI recommended). Exposes a single main endpoint,
  `POST /verify`, that runs the 4-stage pipeline (below) and returns a JSON
  verification report. Should also expose `GET /health` for a simple liveness
  check.
- **Frontend**: A real web UI (React or plain HTML/CSS/JS — Antigravity's
  choice), calling the backend's `/verify` endpoint. NOT the Gradio prototype
  used during development (see Section 7 for what the prototype looked like
  and what it was missing — use that only as a reference for required
  features, not as code to copy).
- **Models run once at backend startup** (not per-request) and stay loaded in
  memory/GPU — request latency should only include inference, not model
  loading.

---

## 3. The Core Pipeline (validated, do not redesign — port faithfully)

This exact 4-stage design was tested and works. Preserve the logic; only the
surrounding application structure (API, UI) needs to be (re)built properly.

### Stage 1 — Claim Decomposition
Input: the generated answer (string).
Output: a list of atomic claims (strings), each a complete, standalone,
grammatically correct sentence expressing exactly one fact.

Two implementations were validated; **support both, selectable via config**:

- **Rule-based** (fast, free, always available, no external dependency):
  1. Split on sentence boundaries: regex `(?<=[.!?])\s+(?=[A-Z])`
  2. Further split each sentence on compounding cues: regex
     `,\s+(?:with|and|which|while)\s+|;\s+`
  3. Trim, drop fragments ≤3 chars.
  - Known limitation: can produce grammatically incomplete fragments from
    aggressive splitting (e.g. "profit margin of 11%." with no subject),
    which measurably hurt downstream entailment accuracy in evaluation
    (see Section 6). Keep this option since it has zero latency/cost, but do
    not present it as the "default best" option in the UI.

- **LLM-based** (better claim quality, requires an API key + network call):
  - Use Google Gemini via the `google-genai` SDK (NOT the deprecated
    `google-generativeai` package).
  - **Model name matters and changes over time** — as of Aug 2026,
    `gemini-2.0-flash` and `gemini-2.5-flash-lite` are both already
    retired/restricted for new users. The working model at time of writing
    is `gemini-3.5-flash-lite`. **Before hardcoding a model name, verify it
    is currently available** (Google deprecates Flash-tier models roughly
    every 2-4 months) — make the model name a config value, not a constant
    buried in code, so it's a one-line change when it's retired again.
  - Free tier rate limits are aggressive and vary by model (as low as
    15 requests/minute, 20-1500 requests/day depending on model tier).
    Backend must handle `429 RESOURCE_EXHAUSTED` and `503 UNAVAILABLE`
    gracefully with exponential backoff (2-3 retries), and on final failure,
    fall back to treating the entire answer as a single claim rather than
    erroring out the whole request.
  - Prompt used (validated, produces clean atomic claims):
    ```
    Break the following text into a list of independent, atomic factual claims.

    Rules:
    - Each claim must be a complete, grammatically correct sentence on its own
      (it must make sense with no other context).
    - Each claim should express exactly ONE fact.
    - Do not add any information that isn't in the text.
    - Output ONLY the claims, one per line, with no numbering, bullets, or
      extra commentary.

    Text: "{text}"
    ```
  - Guard against `response.text` being `None` (happens on safety-filter
    blocks or empty candidates) — fall back to single-claim rather than
    crashing.

### Stage 2 — Retrieval Alignment
Input: one claim + the full list of retrieved passages.
Output: the top-k (k=2, validated) most relevant passages for that claim.

- Model: `sentence-transformers/all-MiniLM-L6-v2` (small, fast, free, loads
  once).
- Method: embed the claim and all candidate passages, rank by cosine
  similarity (`sentence_transformers.util.cos_sim`), take top-k.
- **Critical, previously-missing step — do not skip:** before embedding,
  each retrieved passage/document must be **chunked into small windows
  (~2 sentences each)**, not passed in as one long document. Without this,
  a long context document gets truncated by the downstream NLI model's
  256-token limit and relevant evidence silently gets cut off, which was
  empirically observed to cause a strong systematic bias toward
  "unsupported" verdicts even for genuinely supported claims. Chunking
  function (validated):
  ```python
  def chunk_context(context: str, sentences_per_chunk: int = 2) -> list[str]:
      sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', context.strip())
      sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
      chunks = []
      for i in range(0, len(sentences), sentences_per_chunk):
          chunk = " ".join(sentences[i:i + sentences_per_chunk])
          if chunk:
              chunks.append(chunk)
      return chunks if chunks else [context[:1000]]
  ```
  Run every raw passage through this before alignment.
- Ensure the embedding model is explicitly placed on GPU when available:
  `SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available()
  else "cpu")`. This was missed in an early version and silently fell back to
  CPU, causing a >10x slowdown.

### Stage 3 — Entailment Classification (the core ML model)
Input: one claim + its aligned evidence passages (joined into one string).
Output: one of `supported` / `unsupported` / `contradicted`, plus a confidence
score.

- Model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (HuggingFace,
  `transformers` library — `AutoModelForSequenceClassification` +
  `AutoTokenizer`).
- Raw model outputs `entailment` / `neutral` / `contradiction`; map to
  `supported` / `unsupported` / `contradicted` respectively.
- Tokenize with `evidence` as the premise (first argument) and `claim` as the
  hypothesis (second argument), `truncation=True`, `max_length=256`.
- **Explicitly move both model and tokenized inputs to GPU** —
  `model.to(device)` and `{k: v.to(device) for k, v in inputs.items()}`.
  Same GPU-placement bug as Stage 2 was found here too; do not repeat it.
- Run with `model.eval()` and `torch.no_grad()`.

### Stage 4 — Aggregation
Input: all claim verdicts for one answer.
Output: one of three answer-level decisions, plus the retained per-claim list
(never collapse to a single score — preserving individual verdicts is the
whole point of the project, see report Section 1 and 4.3).

Rule (validated, matches report Section 4.3):
- Any claim `contradicted` → `REJECT / REGENERATE`
- Else, any claim `unsupported` → `PASS WITH FLAGS`
- Else (all `supported`) → `PASS`

**Known limitation to flag in the UI/docs, not silently hide:** this is a
worst-case rule — one bad claim fails the whole answer. Evaluation showed
that as the number of claims per answer increases (e.g. with the LLM
decomposer producing 3-4 claims per answer vs. 1-2 from the simpler
rule-based splitter), the probability that at least one claim is
misclassified rises, which mechanically lowers the "PASS" rate even when
most individual claims are correct. This is a legitimate design trade-off,
not a bug — document it, don't try to silently "fix" the numbers.

---

## 4. Backend API Contract

`POST /verify`
```jsonc
// Request
{
  "answer": "string — the RAG-generated answer to verify",
  "passages": ["string", "string", ...],   // retrieved passages, raw (unchunked) is fine — backend chunks internally
  "decomposition_mode": "rule_based" | "llm"   // optional, default "rule_based"
}

// Response
{
  "claims": [
    {
      "claim": "string",
      "label": "supported" | "unsupported" | "contradicted",
      "confidence": 0.0-1.0,
      "evidence": "string — the aligned passage text used for the verdict"
    }
  ],
  "decision": "PASS (fully supported)" | "PASS WITH FLAGS (some claims unverifiable)" | "REJECT / REGENERATE (contradiction found)",
  "latency_ms": 123
}
```

`GET /health` → `{"status": "ok", "device": "cuda" | "cpu"}`

---

## 5. Frontend Requirements

Build a proper, polished single-page UI (not a Gradio prototype). Required
features:

1. **Input panel**: textarea for the generated answer, textarea for retrieved
   passages (one per line is acceptable input format), a toggle/dropdown for
   decomposition mode (rule-based vs LLM), a "Verify" submit button.
2. **Output panel**:
   - A prominent top banner showing the final decision, color-coded:
     green = PASS, amber = PASS WITH FLAGS, red = REJECT.
   - One card per claim, color-coded to match its label, showing: the claim
     text, the label with an icon, the confidence score, and the evidence
     text used.
   - A small chart/breakdown showing the count of claims per label (bar or
     donut).
3. **Example loader**: at least 2 preset example (answer, passages) pairs the
   user can load with one click for demos — reuse the two examples validated
   during development:
   - Example A (revenue/margin — designed to show a mixed PASS WITH FLAGS
     result):
     - Answer: `"The company announced quarterly revenue of 4.2 billion
       rupees, with profit margins increasing as a result of cutting costs in
       manufacturing."`
     - Passages: `"The company reported quarterly revenue of 4.2 billion
       rupees with a net profit margin of 11 percent."`, `"The company
       operates in the consumer electronics sector across South Asia."`
   - Example B (Eiffel Tower — designed to show a CONTRADICTED result, since
     it's in Paris, not Rome):
     - Answer: `"The Eiffel Tower was built in 1889 and is located in Rome."`
     - Passages: `"The Eiffel Tower was completed in 1889 for the World's
       Fair. It is one of the most visited monuments in Paris, France."`
4. Loading state while the backend call is in flight (entailment inference
   plus, if LLM decomposition is selected, a network round trip — can take
   several seconds).
5. Clear error state if the backend is unreachable or returns an error.

Visual style: clean, modern, presentation-ready (this will be demoed in an
academic viva) — plenty of white space, a legible sans-serif font, subtle
shadows/rounded corners on cards, not a raw/default framework look.

---

## 6. Evaluation Results Already Obtained (for reference / do not repeat blindly)

Dataset used: `wandb/RAGTruth-processed` (HuggingFace `datasets`), a cleaned
parquet version of the RAGTruth benchmark, 2700 test examples, each with a
`context` (retrieved passage(s)), `output` (generated answer), and
`hallucination_labels_processed` (struct with `evident_conflict` and
`baseless_info` span counts) used to derive an answer-level ground truth:
`evident_conflict > 0` → contradicted; `elif baseless_info > 0` →
unsupported; else supported.

150-example run, LLM-based decomposition (`gemini-3.5-flash-lite`), fixed
GPU placement + context chunking:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Supported | 0.833 | 0.091 | 0.164 | 110 |
| Unsupported | 0.086 | 0.545 | 0.148 | 11 |
| Contradicted | 0.221 | 0.517 | 0.309 | 29 |

Accuracy: 0.207. Macro-F1: 0.207. Avg latency: 9.53s/answer (dominated by
LLM decomposition network calls, not the entailment model itself — the
DeBERTa classification step alone is sub-100ms on a T4 GPU).

This run is a **preliminary single-sample evaluation**, not a final
benchmark number — see report Section 5.5 for the full discussion this spec
is based on. Do not treat 20.7% as a target to silently "optimize away"
without understanding why (see the aggregation-policy note in Stage 4 above)
— any change to the pipeline's accuracy behavior should be re-evaluated on
this same dataset/methodology so numbers stay comparable.

Full-dataset (634 examples) and cross-baseline (LLM-judge, RAGAS-style,
binary-classification ablation) evaluation as described in report Section
5.3 is still outstanding — worth building the evaluation script as a
reusable CLI tool (`evaluate.py --n-samples N --dataset ragtruth`) rather
than a one-off notebook cell, so it can be re-run against the finished
backend.

---

## 7. What Was Already Built (development artifacts, for reference)

During development, this was validated as a set of standalone Python scripts
run cell-by-cell in Google Colab (chosen only for free GPU access — the code
has no Colab-specific dependency and runs anywhere with Python + the listed
packages):

- `hallucination_verifier.py` — Stages 1 (rule-based), 2, 3, 4 as plain
  classes/functions (`decompose_into_claims`, `RetrievalAligner`,
  `EntailmentClassifier`, `VerificationReport`, `HallucinationVerifier`).
- `llm_decomposition.py` — overrides Stage 1 with the Gemini-based version.
- `evaluate_on_ragtruth.py` — the evaluation harness described in Section 6.
- `demo_ui_advanced.py` — a Gradio-based quick prototype UI (color-coded
  cards + bar chart). This proved the *feature set* users want (see Section
  5) but is not proper application code — it's a single monolithic script
  with no API layer, not something to extend. Use it only to understand the
  intended look/feel, not as a code base.

Dependencies used: `torch`, `transformers`, `sentence-transformers`,
`datasets`, `scikit-learn` (for evaluation metrics), `google-genai` (for LLM
decomposition), `gradio` + `matplotlib` (prototype UI only, not needed for
the real frontend).

---

## 8. Build Phases (for tracking)

- [ ] **Phase 1 — Backend core**: Port Stages 1-4 into a proper Python
      package (not notebook cells). Unit-testable functions, config for
      model names / decomposition mode / device.
- [ ] **Phase 2 — Backend API**: Wrap the package in FastAPI per the
      contract in Section 4. Models load once at startup. `/health` endpoint.
- [ ] **Phase 3 — Frontend**: Build the UI per Section 5, calling the
      backend API (not embedding the model in the frontend).
- [ ] **Phase 4 — Integration test**: Run both examples from Section 5 end
      to end through the real UI → API → model path; confirm outputs match
      what was validated during development (PASS WITH FLAGS for Example A,
      REJECT for Example B).
- [ ] **Phase 5 — Evaluation CLI**: Turn `evaluate_on_ragtruth.py` into a
      reusable script that hits the new backend API (not the old in-notebook
      classes directly), so evaluation exercises the same code path as
      production.
- [ ] **Phase 6 — Deployment**: Deploy backend + frontend somewhere with a
      permanent URL (e.g. Hugging Face Spaces, or backend on a small GPU
      cloud instance + frontend on Vercel/Netlify) so it doesn't depend on a
      Colab session being open for demos/viva.
- [ ] **Phase 7 — Report integration**: Once Phase 5 produces a full
      634-example (or larger) run plus baseline comparisons (report Section
      5.3), replace the preliminary numbers in report Section 5.5 with final
      ones.

---

## 9. Known Gotchas (do not rediscover these the hard way)

1. **GPU device placement is not automatic for `transformers` models.**
   `SentenceTransformer(...)` picks up GPU on its own; a plain
   `AutoModelForSequenceClassification.from_pretrained(...)` does not — you
   must call `.to(device)` yourself, and move tokenized inputs too.
2. **Long context passed as one string silently loses information** to the
   NLI model's max token length. Always chunk before embedding/aligning (see
   Stage 2).
3. **Aggressive rule-based sentence splitting can create sub-sentence
   fragments** that confuse the entailment model into false "unsupported"
   verdicts. If keeping the rule-based path, keep splits at clause boundaries
   that still read as complete thoughts.
4. **Gemini free-tier model names and quotas change frequently** (multiple
   models were deprecated mid-development: `gemini-2.0-flash`,
   `gemini-2.5-flash-lite` both became unavailable to new users within the
   same week). Keep the model name in config, add retry/backoff, and add a
   graceful single-claim fallback on repeated failure — never let a
   third-party API hiccup break the whole verification request.
5. **Worst-case aggregation amplifies decomposition noise.** More/finer
   claims per answer is not automatically better if it increases the chance
   of one misclassification tanking the whole answer's verdict — this is a
   genuine design trade-off to discuss in the report, not a bug to chase away.
