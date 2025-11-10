#!/usr/bin/env python3
"""
chat_hf.py — Interactive CLI for local/HF models with persona + live switch + fast SMS + bench.

Example:
  python chat_hf.py --model meta-llama/Llama-3.1-8B-Instruct \
    --trust-remote-code --persona persona/kermit.json
"""

import argparse, threading, json, os, re, time
from typing import Optional, List, Dict, Any
import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TextIteratorStreamer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
)

# ---------- quantization helpers ----------
def build_bnb(bits: Optional[int]):
    if not bits:
        return None
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError("--bits must be 4, 8, or omitted")


def _first_word(name: Optional[str], default="Assistant") -> str:
    if not name:
        return default
    return re.sub(r"[^A-Za-z0-9_.\\-]", "", name.strip().split()[0]) or default


# ---------- persona loader ----------
def load_persona(path_or_name: str, base_dir: str = "persona") -> Dict[str, Any]:
    """Load a persona JSON either by path or short name."""
    if not path_or_name:
        return {}
    if os.path.isfile(path_or_name):
        path = path_or_name
    else:
        guess = os.path.join(base_dir, f"{path_or_name}.json")
        path = guess if os.path.isfile(guess) else path_or_name
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load persona '{path_or_name}': {e}")
        return {}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Chat with a HF model (local or hub).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, choices=[4, 8], default=None)
    ap.add_argument("--device-map", default='{"":0}')
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--trust-remote-code", action="store_true", default=True)

    ap.add_argument("--persona", default=None)
    ap.add_argument("--system", default="You are a helpful assistant. Be concise.")
    ap.add_argument("--name", default=None)
    ap.add_argument("--assistant-label", default=None)

    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)

    ap.add_argument("--fast-sms", action="store_true", help="Greedy + 2-sentence early stop.")
    ap.add_argument("--bench", default=None, help="Run fast-sms bench on this prompt and exit.")
    args = ap.parse_args()

    # ---------- persona / system text ----------
    persona = load_persona(args.persona) if args.persona else {}
    system_text = (args.system or "").strip()
    persona_text = ""

    if persona:
        name = persona.get("name", "Assistant")
        profession = persona.get("profession", "")
        personality = persona.get("personality", "")
        style = persona.get("style", "")
        facts_guard = persona.get("facts_guard", False)
        instructions = persona.get("instructions", "")

        persona_lines = [
            f"You are {name}, {profession}." if profession else f"You are {name}.",
            f"Your personality: {personality}" if personality else "",
            f"Your communication style: {style}" if style else "",
            "Keep facts accurate." if facts_guard else "",
            instructions,
        ]
        persona_text = " ".join(line for line in persona_lines if line)

    system_text = (persona_text + " " + system_text).strip()
    if not system_text:
        system_text = "You are a helpful assistant."

    asst_label = args.assistant_label or _first_word(args.name or persona.get("name"), "Assistant")

    # ---------- model load ----------
    try:
        device_map = eval(args.device_map, {"__builtins__": {}})
    except Exception:
        device_map = args.device_map

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    has_qconf = getattr(cfg, "quantization_config", None) is not None
    bnb = build_bnb(args.bits)
    dtype = None if (has_qconf or bnb) else (
        torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    qkwargs = {} if (has_qconf or not bnb) else {"quantization_config": bnb}

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn,
        low_cpu_mem_usage=True,
        **qkwargs,
    ).eval()

    try:
        mdl.config.use_cache = True
    except Exception:
        pass

    # ---------- fast SMS ----------
    def _sms(prompt: str, max_new=48, max_sents=2):
        class _Stop(StoppingCriteria):
            def __init__(self, tok, n):
                self.tok, self.n, self.s = tok, n, ""
            def __call__(self, input_ids, scores, **kw):
                tid = int(input_ids[0, -1])
                self.s += self.tok.decode([tid], skip_special_tokens=True)
                return (self.s.count(".") + self.s.count("?") + self.s.count("!")) >= self.n

        inputs = tok([prompt], return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = mdl.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([_Stop(tok, max_sents)]),
            )
        return tok.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # ---------- bench ----------
    if args.bench:
        msgs = [
            {"role": "system", "content": "Reply in 2–3 sentences, SMS-friendly."},
            {"role": "user", "content": args.bench},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        out = _sms(prompt, max_new=48, max_sents=2)
        dt = time.time() - t0
        ntok = len(tok(out)["input_ids"])
        print(
            json.dumps(
                {
                    "new_tokens": ntok,
                    "latency_s": round(dt, 3),
                    "tok_per_s": round(ntok / max(dt, 1e-9), 2),
                    "output": out,
                },
                indent=2,
            )
        )
        return

    # ---------- interactive loop ----------
    print("Type your message. Commands: /reset, /persona <name>, /exit\n")
    history: List[Dict[str, str]] = []

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history.clear()
            print("🔄 history cleared.")
            continue

        if user.strip() in ("/personae", "/personas"):
            base = os.path.dirname(args.persona) or "."
            print("🎭 Available personae:")
            for f in os.listdir(base):
                if f.endswith(".json"):
                    print("  -", f.replace(".json", ""))
            continue

        # persona switching command
        if user.startswith("/persona"):
            parts = user.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /persona <name_or_path>")
                continue
            new_name = parts[1].strip()
            persona = load_persona(new_name, base_dir=os.path.dirname(args.persona) or ".")
            if not persona:
                print(f"⚠️  Persona '{new_name}' not found.")
                continue

            name = persona.get("name", "Assistant")
            profession = persona.get("profession", "")
            personality = persona.get("personality", "")
            style = persona.get("style", "")
            facts_guard = persona.get("facts_guard", False)
            instructions = persona.get("instructions", "")

            persona_lines = [
                f"You are {name}, {profession}." if profession else f"You are {name}.",
                f"Your personality: {personality}" if personality else "",
                f"Your communication style: {style}" if style else "",
                "Keep facts accurate." if facts_guard else "",
                instructions,
            ]
            persona_text = " ".join(line for line in persona_lines if line)
            system_text = (persona_text + " " + (args.system or "")).strip()
            asst_label = _first_word(name, "Assistant")
            print(f"🎭 Persona switched to {name}")
            continue


        history.append({"role": "user", "content": user})

        # fast SMS shortcut
        if args.fast_sms:
            msgs = [
                {"role": "system", "content": "Reply in 2–3 sentences, SMS-friendly."},
                {"role": "user", "content": user},
            ]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            text = _sms(prompt, max_new=48, max_sents=2)
            print(f"{asst_label}: {text}")
            history.append({"role": "assistant", "content": text})
            continue

        # ---------- normal generation ----------
        msgs = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = tok([prompt], return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        do_sample = args.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
            if args.top_k is not None:
                gen_kwargs["top_k"] = args.top_k
        if args.repetition_penalty and args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        print(f"{asst_label}: ", end="", flush=True)
        t = threading.Thread(
            target=lambda: mdl.generate(**inputs, streamer=streamer, **gen_kwargs),
            daemon=True,
        )
        t.start()
        full = ""
        try:
            for piece in streamer:
                full += piece
                print(piece, end="", flush=True)
        except KeyboardInterrupt:
            print("\n⛔️ Interrupted.")
        print()
        history.append({"role": "assistant", "content": full.strip() or "(no output)"})


if __name__ == "__main__":
    main()

