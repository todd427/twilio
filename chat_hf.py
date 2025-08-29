#!/usr/bin/env python3
# chat_hf.py — interactive CLI for HF models (local or hub)
# Features:
# - Persona JSON (--persona) with name/profession/personality/style/guards
# - Optional --name override; assistant label from first word (or --assistant-label)
# - Robust prompting with custom labels ("You:" and "<Name>:")
# - Identity opener stripping + identity lines anywhere (prevents mirroring)
# - Role-label stripping ("Assistant:", "You:", "User:", "Me:", "Human:", "<Name>:") to prevent loops
# - Default stop guards to cut off transcript loops (You:/User:/Me:/Human:/System:/Assistant:/<Name>:)
# - Context limiter (--max-turns)
# - URL hygiene: block, allowlist, validate, inline/footnote rendering
# - Output hygiene: --answer-first, --no-greetings, --no-praise, --no-emojis
# - Safe: no sampling knobs passed when --temperature 0
#
# pip install -U transformers accelerate bitsandbytes sentencepiece

import argparse, threading, re, json, os
import urllib.parse, urllib.request, urllib.error
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    TextIteratorStreamer, BitsAndBytesConfig,
)

# ===================== Quantization =====================
def build_bnb(bits: Optional[int]):
    if not bits:
        return None
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError("--bits must be 4, 8, or omitted")

# ===================== Labels & Prompt =====================
def first_word(name: Optional[str], default="Assistant") -> str:
    if not name: return default
    w = re.sub(r"[^A-Za-z0-9_.\-]", "", name.strip().split()[0])
    return w or default

def render_prompt(system_text: str, messages: List[Dict[str, str]], you_label: str, asst_label: str, max_turns: int) -> str:
    # Keep last `max_turns` assistant replies and their preceding user turns + the latest user turn
    msgs = messages
    if max_turns and max_turns > 0:
        kept: List[Dict[str, str]] = []
        assistants = 0
        for m in reversed(messages):
            kept.append(m)
            if m["role"] == "assistant":
                assistants += 1
                if assistants >= max_turns:
                    break
        msgs = list(reversed(kept))
        # Ensure we end on the latest user (prompt ends with "<asst_label>:")
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": ""})

    parts = []
    if system_text:
        parts += [system_text.strip(), ""]
    for m in msgs:
        if m["role"] == "user":
            parts += [f"{you_label}: {m['content']}", f"{asst_label}:"]
        elif m["role"] == "assistant":
            parts += [m["content"], ""]
    if not parts or not parts[-1].endswith(f"{asst_label}:"):
        parts.append(f"{asst_label}:")
    return "\n".join(parts).rstrip()

# ===================== Persona =====================
def load_persona(path: Optional[str]) -> dict:
    if not path: return {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # allow inline JSON
    try:
        return json.loads(path)
    except Exception:
        raise ValueError("--persona must be a JSON file path or a JSON string")

def build_system_text(base_system: str, persona: dict, cli_name: Optional[str]) -> str:
    """
    Persona schema (all fields optional):
    {
      "name": "Jimi Hendrix",
      "profession": "Musician",
      "personality": "Laid back, kindly, thoughtful, a bit stoned.",
      "style": "concise, plain",
      "facts_guard": true,
      "instructions": "Avoid exaggerated praise; answer in 1–2 sentences.",
      "extra": "Any extra freeform text."
    }
    """
    sys = (base_system or "").strip()
    name = cli_name or persona.get("name")
    if name and not sys.lower().startswith("your name is"):
        sys = f"Your name is {name}. " + sys
    if persona.get("profession"):
        sys += f" You are a {persona['profession']}."
    if persona.get("personality"):
        sys += f" Personality: {persona['personality']}"
    if persona.get("style"):
        sys += f" Style: {persona['style']}."
    if persona.get("facts_guard", True):
        sys += " Be accurate; if unsure about real people or events, say 'I don't know' or ask for a source. Do not invent facts or links."
    if persona.get("instructions"):
        sys += " " + persona["instructions"].strip()
    if persona.get("extra"):
        sys += " " + str(persona["extra"]).strip()
    return " ".join(sys.split())

# ===================== URL hygiene =====================
URL_RE = re.compile(r'https?://[^\s)\]}>"]+', re.I)

def _netloc(host: str) -> str:
    return host.lower().lstrip(".") if host else ""

def _allowed(netloc: str, allow: Optional[list]) -> bool:
    if not allow: return True
    n = _netloc(netloc)
    return any(n == d.lower() or n.endswith("." + d.lower()) for d in allow)

def _head_ok(url: str, timeout: float) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= getattr(resp, "status", 200) < 300
        except Exception:
            return False

def sanitize_urls(text: str, *, no_urls: bool, allow_domains: Optional[list],
                  validate: bool, timeout: float, style: str) -> str:
    if no_urls:
        return URL_RE.sub("", text)

    urls = []
    for m in URL_RE.finditer(text):
        url = m.group(0)
        try:
            parsed = urllib.parse.urlparse(url)
        except Exception:
            continue
        if not _allowed(parsed.netloc, allow_domains):
            continue
        ok = True
        if validate:
            ok = _head_ok(url, timeout)
        urls.append((m.span(), url, ok))

    if style == "inline":
        out, last = [], 0
        url_map = {(span, url): ok for (span, url, ok) in urls}
        for m in URL_RE.finditer(text):
            span = m.span(); u = m.group(0)
            out.append(text[last:span[0]])
            ok = url_map.get((span, u), None)
            if ok is None:
                pass  # stripped (not allowed)
            else:
                out.append(u if ok else (u + " (broken)"))
            last = span[1]
        out.append(text[last:])
        return "".join(out)

    # footnote
    body = URL_RE.sub("", text)
    if not urls: return body
    notes, seen, idx = [], set(), 1
    for _, url, ok in urls:
        if url in seen: continue
        seen.add(url)
        notes.append(f"[{idx}] {url}" + ("" if ok else " (broken)"))
        idx += 1
    return body.rstrip() + ("\n\n" + "\n".join(notes))

# ===================== Cleaners =====================
def infer_identity_from_system(system_text: str) -> Optional[str]:
    m = re.search(r"Your name is\s+([A-Za-z0-9 _.\-]{2,64})", system_text, re.I)
    return m.group(1).strip() if m else None

def make_identity_regex(name: Optional[str]):
    if not name: return None
    if name.strip().lower() in {"*", "any", "generic"}:
        return re.compile(r"^\s*(?:i am|i’m|i'm)\s+[^\n.]+[^\n]*\n?", re.I)
    n = re.escape(name.strip())
    return re.compile(rf"^\s*(?:i am|i’m|i'm)\s+{n}\b[^\n]*\n?", re.I)

def strip_leading_role_labels(s: str, labels: List[str]) -> str:
    # remove leading "Assistant:" / "<Name>:" / "You:" / "User:" / "Me:" / "Human:" / "System:" repeatedly
    role_re = re.compile(r"^\s*(?:" + "|".join(re.escape(x) for x in labels) + r")\s*:\s*", re.I)
    before = None
    while s and s != before:
        before = s
        s = role_re.sub("", s, count=1)
    return s

def strip_llm_disclaimers(s: str) -> str:
    s = re.sub(r"^\s*(?:i am|i’m|i'm)\s+(?:a|an)\s+.*?(?:model|assistant)\b[^\n]*\n?", "", s, flags=re.I)
    s = re.sub(r"^\s*trained by [^\n]*\n?", "", s, flags=re.I)
    return s

def strip_system_echoes(s: str, system_text: str) -> str:
    # Stronger scrub: drop obvious persona/system lines or long lines appearing in system text
    sys = (system_text or "")
    sys_lines = [ln.strip() for ln in sys.splitlines() if ln.strip()]
    kill_prefixes = ("Your name is", "You are a", "You are an", "Personality:", "Style:", "Instructions:", "Be accurate;")
    out = []
    for ln in s.splitlines():
        raw = ln.strip()
        if not raw:
            continue
        if raw.startswith(kill_prefixes):
            continue
        if any((raw in sl) or (sl in raw) for sl in sys_lines) and len(raw) >= 20:
            continue
        out.append(ln)
    return "\n".join(out)

def trim_on_stops(s: str, stops: List[str]) -> str:
    if not stops: return s
    earliest = None
    for t in stops:
        idx = s.find(t)
        if idx <= 0: continue
        if s[:idx].strip() == "":  # avoid trimming to empty
            continue
        if earliest is None or idx < earliest: earliest = idx
    return s[:earliest] if earliest is not None else s

# Sentence + chatter filters
SENT_SPLIT_RE = re.compile(r"([.!?])(\s+|$)")
def first_sentence(text: str) -> str:
    m = SENT_SPLIT_RE.search(text)
    if not m: return text.strip()
    return text[:m.end(1)].strip()

GREET_RE = re.compile(
    r"^\s*(?:hi|hello|hey|nice to (?:meet|see) you|pleased to meet you|thanks(?: a lot)?|thank you)\b[^\n]*",
    re.I,
)
PRAISE_RE = re.compile(
    r"\b(?:admire(?:d)? your work|you(?:'re| are) (?:my )?favorite(?: musician)?|"
    r"inspiration to (?:me|so many)|so proud of you|i(?:’m|'m) a big fan)\b",
    re.I,
)

# Emojis (basic ranges); safe-fallback for narrow Python builds
try:
    EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF]+")
except re.error:
    EMOJI_RE = re.compile("$^")  # matches nothing on narrow builds

def strip_greetings(s: str) -> str:
    lines = s.splitlines()
    out = []
    for ln in lines:
        if GREET_RE.match(ln): continue
        out.append(ln)
    return "\n".join(out).strip()

def strip_praise(s: str) -> str:
    lines = s.splitlines()
    kept = []
    for ln in lines:
        if PRAISE_RE.search(ln): continue
        kept.append(ln)
    return "\n".join(kept).strip()

def strip_emojis(s: str) -> str:
    return EMOJI_RE.sub("", s)

def strip_identity_lines_anywhere(s: str, name_token: Optional[str]) -> str:
    """Remove lines like 'My name is NAME...' / 'I am NAME...' anywhere in the reply."""
    if not name_token: return s
    pat = re.compile(rf"^\s*(?:my name is|i am|i’m|i'm)\s+{re.escape(name_token)}\b[^\n]*$", re.I | re.M)
    return pat.sub("", s)

def postprocess_full(
    text: str,
    *,
    id_regex,
    stops,
    system_text,
    last_user,
    you_label,
    asst_label,
    drop_first_line,
    strip_roles,
    url_opts,
    answer_first=False,
    no_greetings=False,
    no_praise=False,
    no_emojis=False,
    prev_assistant: str = "",
) -> str:
    s = text

    # Strip leading role labels first
    if strip_roles:
        s = strip_leading_role_labels(s, [asst_label, "Assistant", you_label, "You", "User", "Me", "Human", "System", "system"])

    # Identity opener (start-of-text)
    if id_regex:
        stripped = id_regex.sub("", s, count=1)
        if stripped.strip(): s = stripped

    s = strip_llm_disclaimers(s)

    # Remove identity lines anywhere (prevents "My name is Nancy..." mirroring)
    s = strip_identity_lines_anywhere(s, asst_label)

    if drop_first_line and "\n" in s:
        s = s.split("\n", 1)[1]

    s = trim_on_stops(s, stops)
    s = strip_system_echoes(s, system_text)

    # Anti-echo last user (also catches literal "User: ...")
    lu = last_user.strip()
    removed_user_echo = False
    if lu:
        for pref in ("", f"{you_label}: ", "You: ", "User: "):
            if s.strip().startswith(pref + lu):
                s = s.strip()[len(pref + lu):].lstrip("\n :")
                removed_user_echo = True
                break
    # If we removed the echo and are empty, try to salvage first non-empty non-system line from raw
    if removed_user_echo and not s.strip():
        for ln in text.splitlines():
            tln = ln.strip()
            if not tln: continue
            if tln == lu: continue
            if tln.startswith(("Your name is", "You are a", "You are an", "Personality:", "Style:", "Instructions:")):
                continue
            if system_text and tln in system_text:
                continue
            s = tln
            break

    # Optional trims
    if no_greetings:
        s = strip_greetings(s)
    if no_praise:
        s = strip_praise(s)
    if answer_first:
        for ln in s.splitlines():
            if ln.strip():
                s = first_sentence(ln)
                break
    if no_emojis:
        s = strip_emojis(s)

    s = s.lstrip()

    # De-dup against previous assistant reply
    if prev_assistant and s.strip() == prev_assistant.strip():
        short = first_sentence(s)
        s = short if short else "(no output)"

    # Never return empty
    if not s.strip():
        for ln in text.splitlines():
            ln2 = strip_leading_role_labels(ln, [asst_label, "Assistant", you_label, "You", "User", "Me", "Human", "System", "system"]).strip()
            if id_regex: ln2 = id_regex.sub("", ln2, count=1).strip()
            if ln2:
                s = ln2
                break
        if not s.strip():
            s = "(no output)"

    # URL sanitizing
    s = sanitize_urls(
        s,
        no_urls=url_opts["no_urls"],
        allow_domains=url_opts["allow_domains"],
        validate=url_opts["validate_urls"],
        timeout=url_opts["url_timeout"],
        style=url_opts["link_style"],
    )
    return s

# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="Chat with a Hugging Face model (local or hub).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, choices=[4, 8], default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true")

    # Persona/system
    ap.add_argument("--persona", default=None, help="Path to JSON persona or inline JSON string.")
    ap.add_argument("--system", default="You are a helpful assistant. Speak plainly. No HTML/markdown. Be concise.")
    ap.add_argument("--name", default=None, help="Identity (overrides persona.name).")
    ap.add_argument("--assistant-label", default=None, help='Override assistant label (default = first word of name/persona).')

    # Decoding
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--no-stream", action="store_true")

    # Context & stops / cleanup
    ap.add_argument("--max-turns", type=int, default=3, help="How many past assistant turns to include (1 = single-turn).")
    ap.add_argument("--stop", nargs="*", default=None, help="Extra stop strings. Defaults apply unless --no-stops.")
    ap.add_argument("--no-stops", action="store_true", help="Disable default stop strings.")
    ap.add_argument("--drop-first-line", action="store_true")
    ap.add_argument("--strip-identity", default=None, help='Remove identity opener. Use "*" to remove any.')
    ap.add_argument("--no-role-strip", action="store_true", help="Do NOT strip leading role labels from replies.")
    ap.add_argument("--answer-first", action="store_true", help="Keep only the first sentence of each reply.")
    ap.add_argument("--no-greetings", action="store_true", help="Strip hello/thanks niceties.")
    ap.add_argument("--no-praise", action="store_true", help="Strip user-directed praise/familiarity.")
    ap.add_argument("--no-emojis", action="store_true", help="Remove emojis from the reply.")

    # URL hygiene
    ap.add_argument("--no-urls", action="store_true", help="Remove all URLs from the assistant output.")
    ap.add_argument("--allow-domains", nargs="*", default=None,
                    help="Only keep URLs whose netloc ends with one of these domains (e.g. youtube.com spotify.com).")
    ap.add_argument("--validate-urls", action="store_true", help="HEAD/GET-check URLs; tag '(broken)' if non-2xx.")
    ap.add_argument("--url-timeout", type=float, default=2.5, help="Seconds per URL check.")
    ap.add_argument("--link-style", choices=["inline", "footnote"], default="inline",
                    help="Show kept links inline (default) or as footnotes at the end.")

    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    persona = load_persona(args.persona)
    system_text = build_system_text(args.system, persona, args.name)
    name_for_label = args.name or persona.get("name")
    you_label = "You"
    asst_label = args.assistant_label or first_word(name_for_label, "Assistant")

    # Load model/tokenizer with clean quantization logic
    if not args.quiet:
        print(f"Loading model: {args.model} (bits={args.bits or 'none'}, device_map={args.device_map})")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    has_qconfig = getattr(cfg, "quantization_config", None) is not None

    bnb = build_bnb(args.bits)
    if has_qconfig and bnb and not args.quiet:
        print("ℹ️  Model already includes quantization_config; ignoring --bits.")
    dtype = None if (has_qconfig or bnb) else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    qkwargs = {}
    if bnb and not has_qconfig:
        qkwargs["quantization_config"] = bnb

    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=args.device_map, torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code, **qkwargs
    ).eval()

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # Robust default stops to prevent role-echo loops (unless disabled)
    default_stops = [] if args.no_stops else [
        "\nYou:", "\nUser:", "\nMe:", "\nHuman:", "\nSystem:", "\nAssistant:", f"\n{you_label}:", f"\n{asst_label}:"
    ]
    # de-duplicate while keeping order
    seen = set(); default_stops = [s for s in default_stops if not (s in seen or seen.add(s))]
    stops = (args.stop or []) + default_stops

    # Identity stripping: explicit > name/persona > inferred from system
    identity = args.strip_identity
    if identity is None:
        identity = name_for_label or infer_identity_from_system(system_text)
    id_regex = make_identity_regex(identity) if identity else None

    url_opts = {
        "no_urls": args.no_urls,
        "allow_domains": args.allow_domains,
        "validate_urls": args.validate_urls,
        "url_timeout": args.url_timeout,
        "link_style": args.link_style,
    }

    messages: List[Dict[str, str]] = []
    if not args.quiet:
        print("\nType your message. Commands: /reset, /system <text>, /exit\n")

    while True:
        print()  # blank line before prompt
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break

        if not user: continue
        if user in ("/exit", "/quit"): break
        if user.startswith("/reset"):
            messages.clear(); print("🔄 history cleared."); continue
        if user.startswith("/system "):
            system_text = user[len("/system "):].strip()
            messages.clear()
            if args.strip_identity is None:
                identity = infer_identity_from_system(system_text)
                id_regex = make_identity_regex(identity) if identity else None
            print(f"✅ system set to: {system_text!r} (history cleared)")
            continue

        messages.append({"role": "user", "content": user})
        prompt = render_prompt(system_text, messages, you_label, asst_label, args.max_turns)
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        # Deterministic vs sampling
        do_sample = args.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            eos_token_id=getattr(tok, "eos_token_id", None),
            pad_token_id=getattr(tok, "pad_token_id", None),
        )
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            if args.top_p is not None: gen_kwargs["top_p"] = args.top_p
            if args.top_k is not None: gen_kwargs["top_k"] = args.top_k
        if args.repetition_penalty and args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty

        # previous assistant reply for de-dup
        prev_assistant = ""
        for m in reversed(messages):
            if m["role"] == "assistant":
                prev_assistant = m["content"]
                break

        if args.no_stream:
            with torch.inference_mode():
                out = mdl.generate(**inputs, **gen_kwargs)
            raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            text = postprocess_full(
                raw,
                id_regex=id_regex, stops=stops, system_text=system_text, last_user=user,
                you_label=you_label, asst_label=asst_label,
                drop_first_line=args.drop_first_line, strip_roles=not args.no_role_strip,
                url_opts=url_opts,
                answer_first=args.answer_first, no_greetings=args.no_greetings, no_praise=args.no_praise,
                no_emojis=args.no_emojis, prev_assistant=prev_assistant,
            )
            print(f"{asst_label}: {text}")
        else:
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            print(f"{asst_label}: ", end="", flush=True)
            t = threading.Thread(target=lambda: mdl.generate(**inputs, streamer=streamer, **gen_kwargs), daemon=True)
            t.start()

            chunks_all, buf, started = [], "", False
            try:
                for piece in streamer:
                    chunks_all.append(piece)
                    if not started:
                        buf += piece
                        clean = postprocess_full(
                            buf,
                            id_regex=id_regex, stops=[], system_text=system_text, last_user=user,
                            you_label=you_label, asst_label=asst_label,
                            drop_first_line=args.drop_first_line, strip_roles=not args.no_role_strip,
                            url_opts=url_opts,
                            answer_first=args.answer_first, no_greetings=args.no_greetings, no_praise=args.no_praise,
                            no_emojis=args.no_emojis, prev_assistant=prev_assistant,
                        )
                        if clean != buf or "\n" in buf:
                            print(clean, end="", flush=True)
                            started = True
                            buf = ""
                        continue
                    print(piece, end="", flush=True)
            except KeyboardInterrupt:
                print("\n⛔️ Interrupted.")
            print()
            raw = "".join(chunks_all)
            text = postprocess_full(
                raw,
                id_regex=id_regex, stops=stops, system_text=system_text, last_user=user,
                you_label=you_label, asst_label=asst_label,
                drop_first_line=args.drop_first_line, strip_roles=not args.no_role_strip,
                url_opts=url_opts,
                answer_first=args.answer_first, no_greetings=args.no_greetings, no_praise=args.no_praise,
                no_emojis=args.no_emojis, prev_assistant=prev_assistant,
            )
            print(f"{asst_label}: {text}")
        messages.append({"role": "assistant", "content": text})

if __name__ == "__main__":
    main()
