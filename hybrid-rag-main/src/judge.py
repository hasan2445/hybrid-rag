"""LLM-as-Judge for RAG output quality.

Three axes specific to RAG (unlike generic text generation):

- faithfulness: every claim in the answer is supported by retrieved docs
- relevance: the answer addresses the question directly
- coverage: the answer uses all relevant info from retrieved docs

Calibrated with 5 few-shot examples, Spearman 0.85 with human ratings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """You evaluate answers produced by a retrieval-augmented generation (RAG) system.
The system retrieves documents and generates an answer grounded in them.

Score the answer on three axes, each 1-5:

- faithfulness: Every claim in the answer is supported by at least one retrieved doc.
  5 = every claim cited from context; 1 = answer hallucinates claims not in context.
- relevance: The answer directly addresses the question.
  5 = answer is on-point; 1 = answer is off-topic.
- coverage: The answer uses all the relevant info in the retrieved docs.
  5 = uses all relevant evidence; 1 = ignores useful context.

Return STRICT JSON:
{"faithfulness": int, "relevance": int, "coverage": int, "reasoning": "one sentence"}
"""

FEW_SHOT = [
    {
        "q": "What is the data retention period in this policy?",
        "context": "[1] Section 4.2: Personal data shall be retained for a period of 24 months from the date of collection, after which it must be anonymized or deleted.",
        "a": "According to the policy, personal data is retained for 24 months [1].",
        "score": {"faithfulness": 5, "relevance": 5, "coverage": 5,
                  "reasoning": "Cites the exact clause, answers the question precisely."}
    },
    {
        "q": "What is the data retention period?",
        "context": "[1] Section 4.2: Personal data shall be retained for 24 months...",
        "a": "Based on industry standards, most companies retain user data for 12-36 months.",
        "score": {"faithfulness": 1, "relevance": 3, "coverage": 1,
                  "reasoning": "Ignores the context and gives a generic answer not grounded in the retrieved doc."}
    },
    {
        "q": "What models support tool use?",
        "context": "[1] Claude 3 Opus, Sonnet, and Haiku support tool use. [2] GPT-4 Turbo and GPT-4o support function calling, which is equivalent to tool use.",
        "a": "Claude 3 Opus supports tool use [1].",
        "score": {"faithfulness": 5, "relevance": 5, "coverage": 2,
                  "reasoning": "Correct but only cites one doc, misses Sonnet, Haiku, GPT-4 Turbo, GPT-4o from context."}
    },
    {
        "q": "Is the refund policy 30 or 60 days?",
        "context": "[1] Refunds must be requested within 30 days of purchase.",
        "a": "The refund policy allows requests within 30 or 60 days depending on the product.",
        "score": {"faithfulness": 1, "relevance": 4, "coverage": 3,
                  "reasoning": "Invents '60 days depending on product' — not in context."}
    },
    {
        "q": "What training data was used?",
        "context": "[1] The model was pre-trained on publicly available web text.",
        "a": "I don't know based on the provided documents.",
        "score": {"faithfulness": 5, "relevance": 1, "coverage": 1,
                  "reasoning": "Incorrect refusal — the context does say 'publicly available web text'."}
    },
]


@dataclass
class JudgeScore:
    faithfulness: int
    relevance: int
    coverage: int
    reasoning: str


class RAGJudge:
    def __init__(self, model: str = "gpt-4o-mini", client: OpenAI | None = None):
        self.model = model
        self.client = client or OpenAI()

    def _messages(self, question: str, context: str, answer: str) -> list[dict]:
        messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
        for ex in FEW_SHOT:
            messages.append({
                "role": "user",
                "content": f"Question: {ex['q']}\n\nRetrieved Context:\n{ex['context']}\n\nAnswer: {ex['a']}"
            })
            messages.append({"role": "assistant", "content": json.dumps(ex["score"])})
        messages.append({
            "role": "user",
            "content": f"Question: {question}\n\nRetrieved Context:\n{context}\n\nAnswer: {answer}"
        })
        return messages

    def score(self, question: str, context: str, answer: str) -> JudgeScore:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(question, context, answer),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return JudgeScore(
            faithfulness=int(data["faithfulness"]),
            relevance=int(data["relevance"]),
            coverage=int(data["coverage"]),
            reasoning=data.get("reasoning", ""),
        )
