"""
Upgrade generation backend for ARC while keeping RDE + signal extraction intact.

Run examples:
  python scripts/09_upgrade_llm.py
  python scripts/09_upgrade_llm.py --backend ollama --ollama-model llama3.2
  python scripts/09_upgrade_llm.py --backend hf_small --hf-model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "rde"

DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_HF_SMALL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_HF_API_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_SIGNAL_LEN = 256


MODE_PARAMS = {
    "fast": {"temperature": 0.15, "max_tokens": 24, "n_samples": 1, "use_cot": False},
    "cot": {"temperature": 0.35, "max_tokens": 128, "n_samples": 1, "use_cot": True},
    "best_of_n": {"temperature": 0.65, "max_tokens": 192, "n_samples": 2, "use_cot": True},
}

PROOF_KEYWORDS = {
    "prove", "proof", "show that", "theorem", "lemma", "corollary",
    "contradiction", "induction", "derive", "irrational", "recurrence",
    "integer solutions", "if and only if", "iff",
}
MATH_SYMBOLS_RE = re.compile(
    r"[\^\u2200\u2203\u2208\u2211\u221a\u221e\u2264\u2265\u2260\u2261\u00b1\u00d7\u00f7]|\\[a-zA-Z]+|\d+\^\d+|[a-z]\([a-z]\)"
)
ABSTRACT_NOUNS = {
    "theorem", "lemma", "corollary", "proof", "polynomial", "integer",
    "rational", "irrational", "prime", "matrix", "eigenvalue", "derivative",
    "integral", "convergence",
}


def _extract_final_answer_span(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned[-60:]


def _sanitize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?:^|\n)\s*(question|q)\s*:\s*.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"(?:^|\n)\s*answer\s*:\s*", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines[:6])


def _rule_based_fast_answer(question: str) -> Optional[str]:
    q = question.strip()
    ql = q.lower()

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×])\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        a = float(m.group(1))
        op = m.group(2)
        b = float(m.group(3))
        if op in ("x", "×", "*"):
            val = a * b
        elif op == "+":
            val = a + b
        elif op == "-":
            val = a - b
        else:
            if abs(b) < 1e-12:
                return "Undefined (division by zero)."
            val = a / b
        return str(int(round(val))) if abs(val - round(val)) < 1e-9 else f"{val:.6g}"

    capitals = {
        "india": "New Delhi",
        "france": "Paris",
        "australia": "Canberra",
        "japan": "Tokyo",
    }
    if "capital of" in ql:
        for c, cap in capitals.items():
            if c in ql:
                return cap

    if "time complexity" in ql and "merge sort" in ql:
        return "O(n log n)"

    return None


def _compact_fast_answer(text: str) -> str:
    txt = _sanitize_answer(text)
    if not txt:
        return txt
    line = txt.splitlines()[0].strip()
    # Prefer concise first sentence.
    sent = re.split(r"(?<=[.!?])\s+", line)[0]
    return sent[:140].strip()


class RDENet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BackendConfig:
    backend: str
    ollama_model: str
    ollama_url: str
    hf_small_model: str
    hf_api_model: str
    hf_token: str


class UpgradedARCPipeline:
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_tok = None
        self.hf_model = None

        print(f"Backend: {cfg.backend}")
        if cfg.backend == "ollama":
            print(f"Ollama model: {cfg.ollama_model}")
            self._check_ollama()
        elif cfg.backend == "hf_small":
            self._load_hf_small()
        elif cfg.backend == "hf_api":
            if not cfg.hf_token:
                raise RuntimeError("HF API backend selected, but HF_TOKEN env var is missing.")
            print(f"HF API model: {cfg.hf_api_model}")
        else:
            raise ValueError(f"Unsupported backend: {cfg.backend}")

        self._load_rde()
        self._load_signal_extractor()

    def _check_ollama(self) -> None:
        try:
            import requests

            resp = requests.get("http://localhost:11434", timeout=3)
            if resp.status_code >= 400:
                raise RuntimeError(f"Ollama unhealthy HTTP status: {resp.status_code}")
        except Exception as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start it with `ollama serve` and pull a model, e.g. "
                "`ollama pull llama3.2`."
            ) from exc

    def _load_hf_small(self) -> None:
        print(f"Loading local HF model: {self.cfg.hf_small_model}")
        self.hf_tok = AutoTokenizer.from_pretrained(self.cfg.hf_small_model)
        if self.hf_tok.pad_token is None:
            self.hf_tok.pad_token = self.hf_tok.eos_token
        self.hf_model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_small_model).to(self.device)
        self.hf_model.eval()
        print("HF small model ready.")

    def _load_rde(self) -> None:
        ckpt_path = MODELS_DIR / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing RDE checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.rde = RDENet(
            ckpt["input_dim"], ckpt["hidden_dims"], ckpt["num_classes"], ckpt["dropout"]
        ).to(self.device)
        self.rde.load_state_dict(ckpt["model_state"])
        self.rde.eval()

        self.classes = list(ckpt["label_classes"])
        self.sc_mean = np.load(MODELS_DIR / "scalar_mean.npy")
        self.sc_scale = np.load(MODELS_DIR / "scalar_scale.npy")
        self.em_mean = np.load(MODELS_DIR / "emb_mean.npy")
        self.em_scale = np.load(MODELS_DIR / "emb_scale.npy")
        print("RDE loaded.")

    def _load_signal_extractor(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.sig_tok = AutoTokenizer.from_pretrained("distilgpt2")
        if self.sig_tok.pad_token is None:
            self.sig_tok.pad_token = self.sig_tok.eos_token
        # Important: do not set output_attentions at model init (causes warning).
        self.sig_lm = AutoModelForCausalLM.from_pretrained("distilgpt2").to(self.device)
        self.sig_lm.eval()
        print("Signal extractor loaded (distilgpt2).")

    def _math_features(self, text: str) -> Tuple[float, float, float, float, float, float]:
        tl = text.lower()
        toks = tl.split()
        n = max(len(toks), 1)

        pk = sum(1 for kw in PROOF_KEYWORDS if kw in tl) / n
        ms = len(MATH_SYMBOLS_RE.findall(text)) / n
        eq = len(re.findall(r"[a-zA-Z]\^?\d+|[a-zA-Z]\([a-zA-Z0-9,\s]+\)", text)) / n
        ar = sum(1 for t in toks if t.strip(".,;:") in ABSTRACT_NOUNS) / n
        clean = [t.strip(".,;:!?()[]{}") for t in toks if t.strip(".,;:!?()[]{}")]
        aw = float(np.mean([len(t) for t in clean])) if clean else 0.0
        nd = sum(1 for t in toks if re.match(r"^\d+\.?\d*$", t)) / n
        return pk, ms, eq, ar, aw, nd

    def extract_features(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        inputs = self.sig_tok(
            text,
            return_tensors="pt",
            max_length=MAX_SIGNAL_LEN,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            out = self.sig_lm(**inputs, output_hidden_states=True, output_attentions=True)

        emb = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

        ents = []
        if out.attentions:
            for la in out.attentions:
                p = la.squeeze(0) + 1e-9
                h = -torch.sum(p * torch.log(p), dim=-1)
                ents.append(h.mean().item())
        attn_ent = float(np.mean(ents)) if ents else 0.0

        logits = out.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1) + 1e-9
        var_ent = float((-torch.sum(probs * torch.log(probs), dim=-1)).var().item())

        # Compute perplexity manually (without passing labels, avoids loss_type warning path).
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = inputs["input_ids"][:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.sig_tok.pad_token_id,
        )
        ppl = float(torch.exp(ce).item())

        doc = self.nlp(text[:1000])
        sents = list(doc.sents)
        max_d = 0
        for tok in doc:
            d = 0
            cur = tok
            while cur.head != cur:
                cur = cur.head
                d += 1
            max_d = max(max_d, d)
        cl_cnt = sum(1 for t in doc if t.dep_ in ("relcl", "advcl", "ccomp", "xcomp"))
        avg_sl = float(np.mean([len(s) for s in sents])) if sents else 0.0

        pk, ms, eq, ar, aw, nd = self._math_features(text)
        scalars = np.array(
            [attn_ent, var_ent, ppl, float(max_d), float(cl_cnt), avg_sl, pk, ms, eq, ar, aw, nd],
            dtype=np.float32,
        )
        return scalars, emb

    def rde_predict(self, question: str) -> Tuple[str, float, Dict[str, float]]:
        scalars, emb = self.extract_features(question)
        s_norm = (scalars - self.sc_mean) / (self.sc_scale + 1e-8)
        e_norm = (np.array(emb, dtype=np.float32) - self.em_mean) / (self.em_scale + 1e-8)
        x = torch.tensor(np.concatenate([s_norm, e_norm]), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.rde(x), dim=-1).squeeze().cpu().numpy()
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx]), dict(zip(self.classes, probs.tolist()))

    @staticmethod
    def select_mode(difficulty: str, conf: float) -> Tuple[str, bool]:
        mode = {"easy": "fast", "medium": "cot", "hard": "best_of_n"}.get(difficulty, "cot")
        escalated = False
        if conf < 0.55:
            if mode == "fast":
                mode = "cot"
                escalated = True
            elif mode == "cot":
                mode = "best_of_n"
                escalated = True
        return mode, escalated

    @staticmethod
    def build_prompt(question: str, use_cot: bool) -> str:
        if use_cot:
            return (
                "You are a precise reasoning assistant.\n"
                "Reason step-by-step, then give a concise final answer.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        return f"Answer briefly.\n\nQuestion: {question}\nAnswer:"

    def _generate_ollama_once(self, prompt: str, mode: str) -> Tuple[str, int]:
        import requests

        params = MODE_PARAMS[mode]
        payload = {
            "model": self.cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params["temperature"],
                "num_predict": params["max_tokens"],
                "stop": ["\nQuestion:", "\nQ:", "\nInput:"],
            },
        }
        resp = requests.post(self.cfg.ollama_url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        txt = _sanitize_answer(data.get("response", ""))
        tokens = int(data.get("eval_count", max(len(txt.split()), 1)))
        return txt, tokens

    def _generate_hf_small_once(self, prompt: str, mode: str) -> Tuple[str, int]:
        assert self.hf_tok is not None and self.hf_model is not None
        params = MODE_PARAMS[mode]
        inputs = self.hf_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        do_sample = True
        gen_cfg = self.hf_model.generation_config.clone()
        gen_cfg.do_sample = True
        gen_cfg.temperature = max(params["temperature"], 0.15)
        gen_cfg.top_p = 0.92
        gen_cfg.top_k = 40
        with torch.no_grad():
            out = self.hf_model.generate(
                **inputs,
                max_new_tokens=params["max_tokens"],
                generation_config=gen_cfg,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                pad_token_id=self.hf_tok.eos_token_id,
                eos_token_id=self.hf_tok.eos_token_id,
                use_cache=True,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        decoded = self.hf_tok.decode(gen, skip_special_tokens=True)
        txt = _compact_fast_answer(decoded) if mode == "fast" else _sanitize_answer(decoded)
        return txt, int(len(gen))

    def _generate_hf_api_once(self, prompt: str, mode: str) -> Tuple[str, int]:
        import requests

        params = MODE_PARAMS[mode]
        headers = {"Authorization": f"Bearer {self.cfg.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": params["temperature"],
                "max_new_tokens": params["max_tokens"],
                "return_full_text": False,
            },
        }
        url = f"https://api-inference.huggingface.co/models/{self.cfg.hf_api_model}"
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            txt = str(data[0].get("generated_text", ""))
        else:
            txt = json.dumps(data)
        txt = _sanitize_answer(txt)
        return txt, max(len(txt.split()), 1)

    def generate_answer(self, prompt: str, mode: str) -> Tuple[str, int]:
        n = MODE_PARAMS[mode]["n_samples"] if mode == "best_of_n" else 1
        candidates: List[Tuple[str, int]] = []

        for _ in range(n):
            if self.cfg.backend == "ollama":
                answer, tokens = self._generate_ollama_once(prompt, mode)
            elif self.cfg.backend == "hf_small":
                answer, tokens = self._generate_hf_small_once(prompt, mode)
            else:
                answer, tokens = self._generate_hf_api_once(prompt, mode)
            candidates.append((answer, tokens))

        if len(candidates) == 1:
            return candidates[0]

        finals = [_extract_final_answer_span(c[0]) for c in candidates]
        counts: Dict[str, int] = {}
        for f in finals:
            counts[f] = counts.get(f, 0) + 1
        majority = max(counts.items(), key=lambda kv: kv[1])[0]

        best_idx = 0
        best_score = -1e9
        for i, (ans, tok) in enumerate(candidates):
            score = 0.0
            if _extract_final_answer_span(ans) == majority:
                score += 1.0
            score += min(len(ans), 400) / 4000.0
            score += min(tok, 400) / 4000.0
            if score > best_score:
                best_score = score
                best_idx = i
        return candidates[best_idx]

    def run(self, question: str) -> Dict[str, object]:
        t0 = time.time()
        difficulty, conf, probs = self.rde_predict(question)
        mode, escalated = self.select_mode(difficulty, conf)

        # Fast deterministic fallback for simple questions.
        if mode == "fast":
            quick = _rule_based_fast_answer(question)
            if quick is not None:
                return {
                    "question": question,
                    "difficulty": difficulty,
                    "rde_confidence": conf,
                    "rde_probs": probs,
                    "mode": mode,
                    "escalated": escalated,
                    "answer": quick,
                    "tokens": max(len(quick.split()), 1),
                    "time": round(time.time() - t0, 2),
                }

        prompt = self.build_prompt(question, MODE_PARAMS[mode]["use_cot"])
        answer, out_tokens = self.generate_answer(prompt, mode)
        return {
            "question": question,
            "difficulty": difficulty,
            "rde_confidence": conf,
            "rde_probs": probs,
            "mode": mode,
            "escalated": escalated,
            "answer": answer,
            "tokens": out_tokens,
            "time": round(time.time() - t0, 2),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upgrade ARC generation backend to a capable LLM.")
    p.add_argument("--backend", choices=["ollama", "hf_small", "hf_api"], default=DEFAULT_BACKEND)
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--hf-model", default=DEFAULT_HF_SMALL, help="Used for --backend hf_small")
    p.add_argument("--hf-api-model", default=DEFAULT_HF_API_MODEL, help="Used for --backend hf_api")
    p.add_argument("--no-interactive", action="store_true", help="Run benchmark only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BackendConfig(
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        hf_small_model=args.hf_model,
        hf_api_model=args.hf_api_model,
        hf_token=os.environ.get("HF_TOKEN", ""),
    )

    try:
        pipeline = UpgradedARCPipeline(cfg)
    except Exception as exc:
        print(f"Startup error: {exc}")
        print("Hint: install/start Ollama and run `ollama pull llama3.2`, or switch backend with --backend.")
        return

    benchmark = [
        "What is 15 + 27?",
        "Explain the difference between a stack and a queue.",
        "Prove that sqrt(2) is irrational.",
        "Find all integer solutions to x^3 + y^3 = z^3.",
    ]

    total_tokens = 0
    print("\n" + "=" * 78)
    print(f"{'Question':<46} {'Diff':>7} {'Mode':>10} {'Tok':>6} {'Time':>8}")
    print("=" * 78)
    for q in benchmark:
        try:
            res = pipeline.run(q)
        except Exception as exc:
            print(f"Run error for benchmark question: {exc}")
            continue
        total_tokens += int(res["tokens"])
        print(
            f"{q[:44]:<46} {str(res['difficulty']).upper():>7} {str(res['mode']):>10} "
            f"{int(res['tokens']):>6} {float(res['time']):>7.2f}s"
        )
        print(f"  -> {str(res['answer'])[:240]}")
        print("-" * 78)

    baseline = MODE_PARAMS["cot"]["max_tokens"] * len(benchmark)
    savings = 100.0 * (baseline - total_tokens) / max(baseline, 1)
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Always-CoT baseline (est.): {baseline}")
    print(f"Token savings: {savings:.1f}%")

    if args.no_interactive:
        return

    print("\nInteractive mode (type 'quit' to exit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        try:
            res = pipeline.run(q)
        except Exception as exc:
            print(f"Run error: {exc}")
            continue
        print(
            f"[{str(res['difficulty']).upper()} | {res['mode']} | "
            f"conf={float(res['rde_confidence']):.1%}]"
        )
        print(str(res["answer"])[:700])


if __name__ == "__main__":
    main()
