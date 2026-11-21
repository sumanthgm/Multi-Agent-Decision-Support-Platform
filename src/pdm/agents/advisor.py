"""Advisor Agent: alarm code -> knowledge-base entry -> short, cited checklist.

- `advise(code)`: code-filtered lookup (our method). Optional LLM rewording via a local Ollama
  server (set OLLAMA_URL); the LLM may only rephrase the retrieved entry and must keep the source.
- `search(text)`: TF-IDF retrieval over the whole knowledge base, used for the README-style
  baseline that queries with raw sensor values.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).resolve().parents[3] / "knowledge" / "fault_kb.yaml"


class AdvisorAgent:
    def __init__(self, kb_path: Path | str = KB_PATH, llm: bool | None = None):
        self.llm = llm
        self.kb = yaml.safe_load(Path(kb_path).read_text())
        self.codes = list(self.kb)
        docs = [" ".join([e["title"], e["symptoms"], " ".join(e["causes"])]) for e in self.kb.values()]
        self.vec = TfidfVectorizer(ngram_range=(1, 2)).fit(docs)
        self.mat = self.vec.transform(docs)

    def search(self, text: str, k: int = 3) -> list[str]:
        sims = cosine_similarity(self.vec.transform([text]), self.mat)[0]
        return [self.codes[i] for i in sims.argsort()[::-1][:k]]

    def advise(self, code: str, context: str = "", llm: bool | None = None) -> dict:
        """llm (else the agent's default; else whether OLLAMA_URL is set): True asks the LLM
        (local Ollama when OLLAMA_URL is unset), False never does."""
        if code not in self.kb:
            return {
                "code": code,
                "title": "Unknown alarm",
                "steps": ["Call service: no manual entry for this code."],
                "source": None,
            }
        e = self.kb[code]
        answer = {
            "code": code,
            "title": e["title"],
            "causes": e["causes"],
            "checks": e["checks"],
            "steps": e["steps"],
            "safety": e["safety"],
            "source": e["source"],
        }
        url = os.environ.get("OLLAMA_URL")
        if llm is None:
            llm = self.llm if self.llm is not None else bool(url)
        if llm:
            answer["llm_summary"] = self._llm(e, context, url or "http://127.0.0.1:11434")
        return answer

    @staticmethod
    def _llm(entry: dict, context: str, url: str) -> str:
        prompt = (
            "Rewrite these maintenance steps as a short numbered checklist for a technician. "
            "Use the live readings in Context to put the most relevant checks first. "
            "Use ONLY the facts given. End with 'Source: " + str(entry["source"]) + "'.\n\n"
            f"Context: {context}\nEntry: {json.dumps(entry)}"
        )
        body = json.dumps(
            {"model": os.environ.get("OLLAMA_MODEL", "llama3.2"), "prompt": prompt, "stream": False}
        ).encode()
        try:
            req = urllib.request.Request(url.rstrip("/") + "/api/generate", body, {"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())["response"]
        except Exception as exc:  # LLM is optional; never block the advice
            return f"(LLM unavailable: {exc})"


def format_checklist(ans: dict) -> str:
    lines = [f"{ans['code']} - {ans['title']}"]
    lines += [f"{i}. {s}" for i, s in enumerate(ans.get("steps", []), 1)]
    if ans.get("safety"):
        lines.append(f"Safety: {ans['safety']}")
    lines.append(f"Source: {ans.get('source')}")
    return "\n".join(lines)
