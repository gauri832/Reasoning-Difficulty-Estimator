import argparse
import copy
import json
import os
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np
import spacy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


if not hasattr(GenerationConfig, "clone"):
    def _compat_clone(self):
        return copy.deepcopy(self)

    GenerationConfig.clone = _compat_clone


ALWAYS_COT_EST_TOKENS = 512
PROOF_KW = {
    "prove",
    "proof",
    "show that",
    "theorem",
    "lemma",
    "contradiction",
    "induction",
    "iff",
    "derive",
    "irrational",
    "integer solutions",
}
MATH_RE = re.compile(r"[\^]|\\[a-zA-Z]+|\d+\^\d+|[a-z]\([a-z]\)")
ABS_N = {
    "theorem",
    "lemma",
    "polynomial",
    "integer",
    "irrational",
    "prime",
    "matrix",
    "eigenvalue",
    "derivative",
}
HARD_CUES = re.compile(
    r"\b(prove|proof|contradiction|induction|theorem|lemma|corollary|"
    r"irrational|integer solutions?|recurrence|show that|if and only if|iff)\b",
    re.IGNORECASE,
)
MEDIUM_CUES = re.compile(
    r"\b(explain|difference between|compare|complexity|time complexity|algorithm|"
    r"average speed|merge sort|quicksort|queue|stack)\b",
    re.IGNORECASE,
)
EASY_ARITH = re.compile(
    r"^\s*(what is|calculate|compute|find)\b.{0,80}(\d+\s*[+\-*/]\s*\d+)",
    re.IGNORECASE,
)
EASY_FACT = re.compile(
    r"\b(capital of|what is the capital|who is|when did|define)\b",
    re.IGNORECASE,
)


class RDENet(nn.Module):
    def __init__(self, d, h, nc, dr):
        super().__init__()
        layers = []
        prev = d
        for hh in h:
            layers.extend([
                nn.Linear(prev, hh),
                nn.BatchNorm1d(hh),
                nn.ReLU(),
                nn.Dropout(dr),
            ])
            prev = hh
        layers.append(nn.Linear(prev, nc))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ARCEngine:
    def __init__(
        self,
        backend: str,
        hf_model_name: str,
        ollama_model: str,
        cpu_fast_max_new_tokens: int,
        cpu_cot_max_new_tokens: int,
        cpu_best_max_new_tokens: int,
    ):
        self.backend = backend
        self.hf_model_name = hf_model_name
        self.ollama_model = ollama_model
        self.cpu_token_caps = {
            "fast": cpu_fast_max_new_tokens,
            "cot": cpu_cot_max_new_tokens,
            "best_of_n": cpu_best_max_new_tokens,
        }
        self.ollama_url = "http://localhost:11434/api/generate"
        self.hf_api_token = os.environ.get("HF_TOKEN", "")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_len = 256

        print(f"Backend: {self.backend} | Device: {self.device}")

        self.nlp = spacy.load("en_core_web_sm")

        self.sig_tok = AutoTokenizer.from_pretrained("distilgpt2")
        if self.sig_tok.pad_token is None:
            self.sig_tok.pad_token = self.sig_tok.eos_token

        self.sig_lm = AutoModelForCausalLM.from_pretrained("distilgpt2").to(self.device)
        self.sig_lm.eval()

        ckpt = torch.load("models/rde/best_model.pt", map_location=self.device, weights_only=False)
        self.rde = RDENet(ckpt["input_dim"], ckpt["hidden_dims"], ckpt["num_classes"], ckpt["dropout"]).to(self.device)
        self.rde.load_state_dict(ckpt["model_state"])
        self.rde.eval()

        self.classes = list(ckpt["label_classes"])
        self.sc_mean = np.load("models/rde/scalar_mean.npy")
        self.sc_scale = np.load("models/rde/scalar_scale.npy")
        self.em_mean = np.load("models/rde/emb_mean.npy")
        self.em_scale = np.load("models/rde/emb_scale.npy")

        self.hf_tok = None
        self.hf_model = None
        if self.backend == "hf_small":
            print(f"Loading answer model: {self.hf_model_name}")
            self.hf_tok = AutoTokenizer.from_pretrained(self.hf_model_name, trust_remote_code=True)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.device)
            self.hf_model.eval()
        elif self.backend == "hf_api" and not self.hf_api_token:
            raise RuntimeError("HF_TOKEN environment variable is required for hf_api backend")

    def _mode_params(self, mode: str) -> dict:
        params = {
            "fast": {"temperature": 0.1, "max_new_tokens": 150, "do_sample": False},
            "cot": {"temperature": 0.5, "max_new_tokens": 512, "do_sample": True},
            "best_of_n": {"temperature": 0.8, "max_new_tokens": 600, "do_sample": True},
        }.get(mode, {"temperature": 0.5, "max_new_tokens": 300, "do_sample": True})

        if self.device == "cpu":
            params = dict(params)
            params["do_sample"] = False
            params["temperature"] = 0.0
            cpu_cap = self.cpu_token_caps.get(mode, self.cpu_token_caps["cot"])
            params["max_new_tokens"] = min(params["max_new_tokens"], int(cpu_cap))
        return params

    def build_prompt(self, question: str, mode: str) -> str:
        if self.hf_tok is not None and hasattr(self.hf_tok, "apply_chat_template"):
            try:
                if mode == "fast":
                    system = "Answer in one sentence or less. Be direct and concise."
                else:
                    system = (
                        "You are a precise reasoning assistant. Think step by step, "
                        "show your work clearly, then state the final answer explicitly."
                    )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question},
                ]
                return self.hf_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        if mode == "fast":
            return f"Answer concisely.\nQ: {question}\nA:"
        return f"You are a concise, precise reasoning assistant.\nQuestion: {question}\n\nStep-by-step:"

    def generate_hf_small(self, prompt: str, mode: str):
        params = self._mode_params(mode)
        clean_cfg = GenerationConfig(
            pad_token_id=self.hf_tok.eos_token_id,
            eos_token_id=self.hf_tok.eos_token_id,
            bos_token_id=getattr(self.hf_tok, "bos_token_id", self.hf_tok.eos_token_id),
            max_new_tokens=params["max_new_tokens"],
            do_sample=params["do_sample"],
            no_repeat_ngram_size=3,
        )
        if params["do_sample"]:
            clean_cfg.temperature = params["temperature"]
            clean_cfg.top_p = 0.92

        try:
            self.hf_model.generation_config = clean_cfg
        except Exception:
            pass

        inputs = self.hf_tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        ).to(self.device)

        with torch.no_grad():
            out = self.hf_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=clean_cfg,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        answer = self.hf_tok.decode(new_ids, skip_special_tokens=True).strip()
        return answer, len(new_ids)

    def generate_ollama(self, prompt: str, mode: str):
        import requests

        p = self._mode_params(mode)
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": p["temperature"],
                "num_predict": p["max_new_tokens"],
            },
        }
        r = requests.post(self.ollama_url, json=payload, timeout=120)
        data = r.json()
        answer = data.get("response", "").strip()
        return answer, data.get("eval_count", len(answer.split()))

    def generate_hf_api(self, prompt: str, mode: str):
        import requests

        p = self._mode_params(mode)
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": p["temperature"],
                "max_new_tokens": 300,
                "return_full_text": False,
            },
        }
        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        data = r.json()
        answer = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
        return answer.strip(), len(answer.split())

    def generate_answer(self, prompt: str, mode: str):
        if self.backend == "hf_small":
            return self.generate_hf_small(prompt, mode)
        if self.backend == "ollama":
            return self.generate_ollama(prompt, mode)
        if self.backend == "hf_api":
            return self.generate_hf_api(prompt, mode)
        raise RuntimeError(f"Unsupported backend: {self.backend}")

    def _math_features(self, text: str):
        tl = text.lower()
        toks = tl.split()
        n = max(len(toks), 1)
        pk = sum(1 for kw in PROOF_KW if kw in tl) / n
        ms = len(MATH_RE.findall(text)) / n
        eq = len(re.findall(r"[a-zA-Z]\^?\d+|[a-zA-Z]\([a-zA-Z0-9]+\)", text)) / n
        ar = sum(1 for t in toks if t.strip(".,;:") in ABS_N) / n
        cleaned = [t.strip(".,;:!?()") for t in toks if t.strip(".,;:!?()")]
        avg_word_len = float(np.mean([len(t) for t in cleaned])) if cleaned else 0.0
        num_density = sum(1 for t in toks if re.match(r"^\d+\.?\d*$", t)) / n
        return pk, ms, eq, ar, avg_word_len, num_density

    def extract_features(self, text: str):
        inputs = self.sig_tok(
            text,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            out = self.sig_lm(
                **inputs,
                labels=inputs["input_ids"],
                output_hidden_states=True,
                output_attentions=True,
            )

        emb = out.hidden_states[-1].mean(1).squeeze().cpu().numpy()
        ents = [
            (-torch.sum((layer.squeeze(0) + 1e-9) * torch.log(layer.squeeze(0) + 1e-9), dim=-1)).mean().item()
            for layer in (out.attentions or [])
        ]
        attn_ent = float(np.mean(ents)) if ents else 0.0
        logits = out.logits.squeeze(0)
        probs = torch.softmax(logits, -1) + 1e-9
        var_ent = float((-torch.sum(probs * torch.log(probs), -1)).var().item())
        ppl = float(torch.exp(out.loss).item())

        doc = self.nlp(text[:1000])
        sents = list(doc.sents)
        max_depth = 0
        for tok in doc:
            depth = 0
            cur = tok
            while cur.head != cur:
                cur = cur.head
                depth += 1
            max_depth = max(max_depth, depth)

        clause_count = sum(1 for t in doc if t.dep_ in ("relcl", "advcl", "ccomp", "xcomp"))
        avg_sent_len = float(np.mean([len(s) for s in sents])) if sents else 0.0

        pk, ms, eq, ar, aw, nd = self._math_features(text)
        scalars = np.array(
            [attn_ent, var_ent, ppl, float(max_depth), float(clause_count), avg_sent_len, pk, ms, eq, ar, aw, nd],
            dtype=np.float32,
        )
        return scalars, emb

    def _rule_override(self, question: str):
        q = question.lower()
        if HARD_CUES.search(q):
            return "hard"
        if MEDIUM_CUES.search(q):
            return "medium"
        if EASY_ARITH.search(q) or EASY_FACT.search(q):
            return "easy"
        return None

    def _rule_based_answer(self, question: str) -> Optional[str]:
        q = question.strip()
        ql = q.lower()

        # Basic arithmetic: +, -, *, /, x, ×
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

            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return f"{val:.6g}"

        # Exponent phrasing: "2 to the power of 8"
        p = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to the power of|\^)\s*(-?\d+(?:\.\d+)?)", ql)
        if p:
            base = float(p.group(1))
            exp = float(p.group(2))
            val = base ** exp
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return f"{val:.6g}"

        return None

    def rde_predict(self, question: str):
        scalars, emb = self.extract_features(question)
        sn = (scalars - self.sc_mean) / (self.sc_scale + 1e-8)
        en = (np.array(emb, dtype=np.float32) - self.em_mean) / (self.em_scale + 1e-8)
        x = torch.tensor(np.concatenate([sn, en]), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.rde(x), -1).squeeze().cpu().numpy()

        override = self._rule_override(question)
        if override and override in self.classes:
            idx_ov = self.classes.index(override)
            if probs[idx_ov] < 0.60:
                probs[idx_ov] = 0.60
                probs = probs / probs.sum()
            idx = idx_ov
        else:
            idx = int(probs.argmax())

        return self.classes[idx], float(probs[idx]), dict(zip(self.classes, probs.tolist()))

    def select_mode(self, difficulty: str, confidence: float):
        mode = {"easy": "fast", "medium": "cot", "hard": "best_of_n"}.get(difficulty, "cot")
        escalated = False
        reason = f"difficulty={difficulty}, confidence={confidence:.2f}"

        if confidence < 0.55:
            if mode == "fast":
                mode = "cot"
                escalated = True
            elif mode == "cot":
                mode = "best_of_n"
                escalated = True
            reason += ", escalated due to low confidence"

        return mode, escalated, reason

    def run(self, question: str):
        t0 = time.time()
        difficulty, confidence, probs = self.rde_predict(question)
        mode, escalated, reason = self.select_mode(difficulty, confidence)

        rule_answer = self._rule_based_answer(question)
        if rule_answer is not None:
            answer = rule_answer
            if self.hf_tok is not None:
                tokens = len(self.hf_tok.encode(answer))
            else:
                tokens = len(answer.split())
            mode = "fast"
            escalated = False
            reason = f"{reason}, deterministic arithmetic override"
        else:
            prompt = self.build_prompt(question, mode)
            answer, tokens = self.generate_answer(prompt, mode)

        savings = (ALWAYS_COT_EST_TOKENS - tokens) / ALWAYS_COT_EST_TOKENS * 100.0

        return {
            "question": question,
            "difficulty": difficulty,
            "rde_conf": confidence,
            "difficulty_probs": probs,
            "mode": mode,
            "mode_reason": reason,
            "escalated": escalated,
            "answer": answer,
            "tokens": int(tokens),
            "always_cot_tokens": ALWAYS_COT_EST_TOKENS,
            "query_savings_pct": round(savings, 2),
            "time": round(time.time() - t0, 2),
        }


class ARCRequestHandler(BaseHTTPRequestHandler):
    engine = None
    static_root = None

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = self.path.split("?", 1)[0]

        if route == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if route == "/" or route == "":
            self._send_file(self.static_root / "index.html", "text/html; charset=utf-8")
            return

        mapping = {
            "/app.js": "application/javascript; charset=utf-8",
            "/styles.css": "text/css; charset=utf-8",
        }
        if route in mapping:
            self._send_file(self.static_root / route.lstrip("/"), mapping[route])
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            return

        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "question is required"})
            return

        try:
            result = self.engine.run(question)
            self._send_json(HTTPStatus.OK, result)
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def parse_args():
    parser = argparse.ArgumentParser(description="ARC React UI server")
    parser.add_argument("--backend", default="hf_small", choices=["hf_small", "ollama", "hf_api"])
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--cpu-fast-max-new-tokens", type=int, default=64)
    parser.add_argument("--cpu-cot-max-new-tokens", type=int, default=192)
    parser.add_argument("--cpu-best-max-new-tokens", type=int, default=320)
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    static_root = root / "webui"
    if not static_root.exists():
        raise FileNotFoundError(f"Missing static UI directory: {static_root}")

    engine = ARCEngine(
        args.backend,
        args.hf_model,
        args.ollama_model,
        args.cpu_fast_max_new_tokens,
        args.cpu_cot_max_new_tokens,
        args.cpu_best_max_new_tokens,
    )

    ARCRequestHandler.engine = engine
    ARCRequestHandler.static_root = static_root

    server = ThreadingHTTPServer((args.host, args.port), ARCRequestHandler)
    print(f"ARC web UI ready at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
