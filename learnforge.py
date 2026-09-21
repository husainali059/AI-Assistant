"""Small, auditable RAG pipeline using only the Python standard library."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import re
from urllib import request

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("a an and are as at be but by can did do does for from how i if in is it me my of on or that the this to was what when where will with you your".split())
ENTRY = re.compile(r"^# (FAQ|POLICY|TICKET)-(\d+)\s+—\s+(.+)$", re.M)
DATE = re.compile(r"(?:Last (?:reviewed|updated)|Updated|Reviewed|Effective date|Effective):\s*([A-Za-z]+)\s+(\d{4})", re.I)
AUTHORITY = {"POLICY": 1.0, "FAQ": 0.72, "TICKET": 0.35}
STALE_MARKERS = ("outdated", "obsolete", "archived", "older version", "previous version", "retired", "no longer")
HIGH_RISK = re.compile(r"\b(refund|chargeback|cancel|charged|payment|annual subscription|promo|promotion|terms)\b", re.I)
PROMPT_INJECTION = re.compile(r"\b(ignore|disregard|override)\b.{0,80}\b(instruction|instructions|system|prompt|source|sources|policy)\b|\b(reveal|show)\b.{0,40}\b(system prompt|hidden instruction)\b", re.I)
# Vocabulary normalization is intentionally small and domain-level, rather than a list of
# corpus answers. It lets lexical retrieval handle common customer wording without an API.
EXPANSIONS = (
    (re.compile(r"\b(phone|mobile).{0,80}\b(computer|laptop|desktop)|\b(computer|laptop|desktop).{0,80}\b(phone|mobile)", re.I), "progress synchronize synchronization multiple devices"),
    (re.compile(r"\b(proof|evidence).{0,40}\b(completion|finish)|\b(completion|finish).{0,40}\b(proof|evidence)", re.I), "certificate course completion requirements"),
    (re.compile(r"\b(sister|brother|family|someone else|another person).{0,60}\b(login|account)|\b(share|sharing).{0,40}\b(login|account)", re.I), "account sharing individual learner family organization"),
    (re.compile(r"\breturn\b", re.I), "refund"),
    (re.compile(r"\b(download|offline|laptop)\b", re.I), "offline download mobile application"),
)


def tokens(text: str) -> list[str]:
    return [word for word in TOKEN.findall(text.lower()) if word not in STOPWORDS]


def normalized_query(query: str) -> str:
    additions = [terms for pattern, terms in EXPANSIONS if pattern.search(query)]
    return f"{query} {' '.join(additions)}"


@dataclass(frozen=True)
class Document:
    id: str
    kind: str
    title: str
    text: str
    reviewed: date | None
    stale_language: bool

    @property
    def source(self) -> str:
        return f"{self.kind}-{self.id.split('-')[-1]}: {self.title}"


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float
    lexical: float


class KnowledgeBase:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.df = Counter({term: sum(term in set(tokens(d.title + " " + d.text)) for d in documents)
                           for term in {t for d in documents for t in tokens(d.title + " " + d.text)}})

    @classmethod
    def from_markdown_dir(cls, directory: Path) -> "KnowledgeBase":
        paths = [directory / name for name in ("faqs.md", "policies.md", "tickets.md")]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Knowledge-base files not found: " + ", ".join(missing))
        docs: list[Document] = []
        for path in paths:
            raw = path.read_text(encoding="utf-8")
            matches = list(ENTRY.finditer(raw))
            for index, match in enumerate(matches):
                body = raw[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(raw)]
                kind, number, title = match.groups()
                found = DATE.search(body)
                reviewed = date.fromisoformat(f"{found.group(2)}-{datetime.strptime(found.group(1), '%B').month:02d}-01") if found else None
                docs.append(Document(f"{kind}-{number}", kind, title.strip(), body.strip(), reviewed,
                                     any(marker in body.lower() for marker in STALE_MARKERS)))
        return cls(docs)

    def search(self, query: str, limit: int = 4) -> list[Hit]:
        q = Counter(tokens(normalized_query(query)))
        if not q:
            return []
        results = []
        for doc in self.documents:
            terms = Counter(tokens(doc.title + " " + doc.text))
            dot = sum(q[t] * (1 + math.log(terms[t])) * math.log((1 + len(self.documents)) / (1 + self.df[t])) for t in q if t in terms)
            norm_q = math.sqrt(sum(v * v for v in q.values()))
            norm_d = math.sqrt(sum((1 + math.log(v)) ** 2 for v in terms.values()))
            lexical = dot / (norm_q * norm_d) if norm_d else 0
            authority = AUTHORITY[doc.kind]
            freshness = 0.08 if doc.reviewed else 0
            # Tickets are useful evidence but are never treated as present-day policy.
            penalty = 0.25 if doc.kind == "TICKET" else 0
            title_terms = set(tokens(doc.title))
            title_bonus = 0.04 if title_terms.intersection(q) else 0
            # Authority is a tie-breaker, not a replacement for topical relevance.
            results.append(Hit(doc, lexical * (0.85 + 0.15 * authority) + freshness - penalty + title_bonus, lexical))
        return sorted(results, key=lambda h: h.score, reverse=True)[:limit]


@dataclass
class Conversation:
    turns: list[tuple[str, str]] = field(default_factory=list)
    def contextualize(self, question: str) -> str:
        # Retain only the last user subject; do not let model-generated text enter retrieval.
        return question if not self.turns else f"{self.turns[-1][0]} {question}"


@dataclass(frozen=True)
class Answer:
    answer: str
    decision: str
    confidence: float
    sources: list[str]


class OpenAICompatibleGenerator:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleGenerator | None":
        key = os.getenv("OPENAI_API_KEY")
        return cls(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), key,
                   os.getenv("LLM_MODEL", "gpt-4o-mini")) if key else None

    def generate(self, question: str, hits: list[Hit]) -> str:
        context = "\n\n".join(f"[{h.document.id}] {h.document.text[:1600]}" for h in hits)
        prompt = ("Answer only from the supplied sources. Do not invent account-specific facts. "
                  "If the sources conflict or require order-specific terms, say a human must review it. "
                  f"\n\nQuestion: {question}\n\nSources:\n{context}")
        payload = json.dumps({"model": self.model, "messages": [{"role": "system", "content": "You are a careful LearnForge support agent."}, {"role": "user", "content": prompt}], "temperature": 0.1}).encode()
        req = request.Request(self.base_url + "/chat/completions", data=payload, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with request.urlopen(req, timeout=20) as response:
            return json.load(response)["choices"][0]["message"]["content"].strip()


class Assistant:
    def __init__(self, kb: KnowledgeBase, generator: OpenAICompatibleGenerator | None = None):
        self.kb, self.generator = kb, generator

    def answer(self, question: str, conversation: Conversation) -> Answer:
        hits = self.kb.search(conversation.contextualize(question))
        top_lexical = hits[0].lexical if hits else 0
        policy_hits = [h for h in hits if h.document.kind == "POLICY"]
        subscription_refund = bool(re.search(r"\b(refund|cancel)\b", question, re.I) and re.search(r"\bsubscription\b", question, re.I))
        conflicting = bool(PROMPT_INJECTION.search(question) or subscription_refund or (HIGH_RISK.search(question) and (not policy_hits or any(h.document.kind == "TICKET" for h in hits[:2]))) )
        confidence = max(0, min(1, top_lexical * 3.8 + (0.12 if policy_hits and top_lexical > 0.06 else 0) - (0.25 if conflicting else 0)))
        # A conflict may retrieve highly relevant text but still cannot support a safe answer.
        if conflicting:
            confidence = min(confidence, 0.40)
        sources = [h.document.source for h in hits]
        if confidence < 0.45 or conflicting:
            message = ("I can’t safely determine that from the knowledge base alone. I’ll escalate this to a "
                       "human support specialist to verify the applicable account, purchase date, and terms.")
            result = Answer(message, "escalate", confidence, sources)
        else:
            if self.generator:
                message = self.generator.generate(question, hits)
            else:
                excerpt = re.sub(r"\s+", " ", hits[0].document.text).strip()
                message = f"Based on {hits[0].document.source}: {excerpt[:700]}"
            result = Answer(message, "answer", confidence, sources)
        conversation.turns.append((question, result.answer))
        return result
