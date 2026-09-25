# LearnForge support assistant

A compact RAG prototype for the Applied AI/LLM Engineer take-home. It ingests the supplied Markdown corpus, retrieves individual FAQ/policy/ticket records, applies source-authority and recency weighting, carries forward the previous user question, and escalates uncertain or purchase-specific cases.

## Run

Python 3.11+ is the only requirement.

```bash
python app.py --question "Can I download a course on my laptop?"
python -m unittest -v
```

The default corpus directory is the supplied assignment location. For another location, set `KNOWLEDGE_BASE_DIR` or pass `--data-dir`.

## AI-generated responses

The assistant always retrieves company evidence first, then an OpenAI model writes the customer-facing reply from that evidence. This is not a hard-coded response: the AI creates the wording while retrieval keeps policy answers grounded.

Set a replacement API key in your terminal only—never paste it into `app.py`, `learnforge.py`, Git, or a committed `.env` file:

```bash
export OPENAI_API_KEY="your-replacement-key"
export LLM_MODEL="gpt-5-mini"
python app.py --require-ai --question "My course progress is not saving."
```

The application uses the OpenAI Responses API with `store: false`. Each result visibly reports one of these modes:

- `AI LLM grounded answer — Provider (model)` — an LLM wrote the response using retrieved company evidence.
- `Retrieval fallback — AI unavailable` — no API key was configured, so a source excerpt was returned.
- `Escalated — ...` — evidence was insufficient/conflicting or the AI request failed.

`OPENAI_BASE_URL` can be set for a compatible endpoint. Without a key, the app remains runnable and auditable in retrieval-fallback mode.

### Groq

Groq works automatically with its own key; do not set `OPENAI_API_KEY` for this mode:

```bash
export GROQ_API_KEY="your-new-groq-key"
export LLM_MODEL="openai/gpt-oss-20b"
python app.py --require-ai --question "My course progress is not saving."
```

The app detects `GROQ_API_KEY`, uses `https://api.groq.com/openai/v1`, and calls its compatible Responses API. `llama-3.3-70b-versatile` is deprecated, so use `openai/gpt-oss-20b` or `openai/gpt-oss-120b` instead.

The request includes an explicit application `User-Agent`, which avoids Groq edge-layer rejection of Python's default `urllib` signature.

## Architecture

```mermaid
flowchart LR
  U[User + prior turn] --> C[Query contextualizer]
  C --> R[Lexical retrieval]
  KB[(FAQ / policies / tickets)] --> R
  R --> G[Authority + freshness ranker]
  G --> Q{Confidence / conflict gate}
  Q -->|safe| L[Grounded LLM or offline answer]
  Q -->|uncertain, stale, account-specific| H[Human-support escalation]
  L --> U
  H --> U
```

## Data schema

Each parsed record is a `Document`:

| Field | Purpose |
| --- | --- |
| `id`, `kind`, `title`, `text` | retrieval identity and content |
| `reviewed` | explicit policy recency where supplied |
| `stale_language` | ingestion signal for superseded material |
| `score`, `source` | retrieval metadata returned to the answer layer |

In production this maps directly to a vector-table record with `embedding`, `content_hash`, `source_uri`, `published_at`, `reviewed_at`, `authority`, and `superseded_by` added for change management.

## Failure handling

- Low retrieval score: do not answer; escalate.
- High-risk billing/refund language without an authoritative policy, or mixed with historical tickets: escalate for purchase-specific review.
- Stale/conflicting text: policies rank above FAQs and tickets; dated policy language is preserved as metadata. The system never treats a historical ticket as current policy.
- Bad retrieval / prompt injection: only retrieved source text is passed to the optional generator, with an explicit grounded-answer instruction. Answers display their sources.
- Account actions: this prototype never performs cancellations, refunds, or account changes; it routes them to a human workflow.

## Evaluation plan

Build a labeled set from the supplied questions plus adversarial variants: direct FAQ questions, policy conflicts, unsupported questions, account-specific claims, stale-help prompts, and multi-turn follow-ups. Track:

- grounded-answer correctness and citation precision;
- hallucination rate (claims unsupported by retrieved passages);
- retrieval recall@k for the expected authoritative source;
- escalation precision/recall for conflict and low-confidence cases;
- latency and the rate of unnecessary escalation.

Review errors by source type and publish the evaluation set with expected `answer` or `escalate` decisions.

## Trade-offs

This uses dependency-free lexical retrieval because the corpus is tiny, transparent, fast, and easy to run in a take-home environment. At scale, I would add hybrid dense + lexical retrieval, reranking, document versioning/tombstones, a real conversation store, offline evaluation, and observability. The deliberate choice is conservative escalation over plausible guesses: it trades coverage for lower policy and billing hallucination risk.
