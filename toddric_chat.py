# toddric_chat.py
# Reusable chat engine with persona + URL hygiene + reply cleanup.
# pip install -U transformers accelerate bitsandbytes sentencepiece

from __future__ import annotations

import re, json, os, urllib.parse, urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ---------- URL hygiene ----------
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
                pass
            else:
                out.append(u if ok else (u + " (broken)"))
            last = span[1]
        out.append(text[last:])
        return "".join(out)

    body = URL_RE.sub("", text)
    if not urls: return body
    notes, seen, idx = [], set(), 1
    for _, url, ok in urls:
        if url in seen: continue
        seen.add(url)
        notes.append(f"[{idx}] {url}" + ("" if ok else " (broken)"))
        idx += 1
    return body.rstrip() + ("\n\n" + "\n".join(notes))

# ---------- text cleaners ----------
def first_word(name: Optional[str], default="Assistant") -> str:
    if not name: return default
    import re as _re
    w = _re.sub(r"[^A-Za-z0-9_.\-]", "", name.strip().split()[0])
    return w or default

def strip_leading_role_labels(s: str, labels: List[str]) -> str:
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

SENT_SPLIT_RE = re.compile(r"([.!?])(\s+|$)")
def first_sentence(text: str) -> str:
    m = SENT_SPLIT_RE.search(text)
    if not m: return text.strip()
    return text[:m.end(1)].strip()

GREET_RE  = re.compile(r"^\s*(?:hi|hello|hey|nice to (?:meet|see) you|pleased to meet you|thanks(?: a lot)?|thank you)\b[^\n]*", re.I)
PRAISE_RE = re.compile(r"\b(?:admire(?:d)? your work|you(?:'re| are) (?:my )?favorite(?: musician)?|inspiration to (?:me|so many)|so proud of you|i(?:’m|'m) a big fan)\b", re.I)
try:
    EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF]+")
except re.error:
    EMOJI_RE = re.compile("$^")

def strip_greetings(s: str) -> str:
    return "\n".join(ln for ln in s.splitlines() if not GREET_RE.match(ln)).strip()
def strip_praise(s: str) -> str:
    return "\n".join(ln for ln in s.splitlines() if not PRAISE_RE.search(ln)).strip()
def strip_emojis(s: str) -> str:
    return EMOJI_RE.sub("", s)

def infer_identity_from_system(system_text: str) -> Optional[str]:
    m = re.search(r"Your name is\s+([A-Za-z0-9 _.\-]{2,64})", system_text, re.I)
    return m.group(1).strip() if m else None

def make_identity_regex(name: Optional[str]):
    if not name: return None
    if name.strip().lower() in {"*", "any", "generic"}:
        return re.compile(r"^\s*(?:i am|i’m|i'm)\s+[^\n.]+[^\n]*\n?", re.I)
    n = re.escape(name.strip())
    return re.compile(rf"^\s*(?:i am|i’m|i'm)\s+{n}\b[^\n]*\n?", re.I)

def strip_identity_lines_anywhere(s: str, name_token: Optional[str]) -> str:
    if not name_token: return s
    pat = re.compile(rf"^\s*(?:my name is|i am|i’m|i'm)\s+{re.escape(name_token)}\b[^\n]*$", re.I | re.M)
    return pat.sub("", s)

# ---------- persona & system ----------
def load_persona(path_or_json: Optional[str]) -> dict:
    if not path_or_json: return {}
    if os.path.isfile(path_or_json):
        with open(path_or_json, "r", encoding="utf-8") as f:
            return json.load(f)
    try:
        return json.loads(path_or_json)
    except Exception:
        raise ValueError("persona must be a JSON file path or an inline JSON string")

def build_system_text(base_system: str, persona: dict, cli_name: Optional[str]) -> str:
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

# ---------- config & engine ----------
@dataclass
class URLSettings:
    no_urls: bool = False
    allow_domains: Optional[List[str]] = None
    validate_urls: bool = False
    url_timeout: float = 2.5
    link_style: str = "inline"  # "inline" | "footnote"

@dataclass
class ReplySettings:
    answer_first: bool = False
    no_greetings: bool = False
    no_praise: bool = False
    no_emojis: bool = False
    strip_identity: Optional[str] = "*"  # None, "*", or a name to strip

@dataclass
class DecodeSettings:
    max_new_tokens: int = 200
    temperature: float = 0.5
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    repetition_penalty: float = 1.05

@dataclass
class EngineConfig:
    model: str
    device_map: str = "auto"
    trust_remote_code: bool = False
    bits: Optional[int] = None  # 4, 8 or None
    system: str = "You are a helpful assistant. Speak plainly. No HTML/markdown. Be concise."
    persona: Optional[dict] = None
    name: Optional[str] = None
    you_label: str = "You"
    assistant_label: Optional[str] = None
    max_turns: int = 3
    url: URLSettings = field(default_factory=URLSettings)
    reply: ReplySettings = field(default_factory=ReplySettings)
    decode: DecodeSettings = field(default_factory=DecodeSettings)
    stops: Optional[List[str]] = None  # extra stops; defaults applied if None

class ChatEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        # Build system text
        persona = cfg.persona or {}
        self.system_text = build_system_text(cfg.system, persona, cfg.name)
        self.asst_label = cfg.assistant_label or first_word(cfg.name or persona.get("name"), "Assistant")
        self.you_label = cfg.you_label

        # Stops to prevent role-echo loops
        default_stops = [
            "\nYou:", "\nUser:", "\nMe:", "\nHuman:", "\nSystem:", "\nAssistant:",
            f"\n{self.you_label}:", f"\n{self.asst_label}:"
        ]
        # dedupe keep order
        seen=set(); default_stops=[s for s in default_stops if not (s in seen or seen.add(s))]
        self.stops = (cfg.stops or []) + default_stops

        # Identity stripping
        ident = cfg.reply.strip_identity
        if ident is None:
            ident = cfg.name or persona.get("name") or infer_identity_from_system(self.system_text)
        self.id_regex = make_identity_regex(ident) if ident else None

        # Load model
        if cfg.bits == 4:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            qkwargs = {"quantization_config": bnb}
            dtype = None
        elif cfg.bits == 8:
            bnb = BitsAndBytesConfig(load_in_8bit=True)
            qkwargs = {"quantization_config": bnb}
            dtype = None
        else:
            qkwargs = {}
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Respect baked-in quantization_config if present
        acfg = AutoConfig.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
        if getattr(acfg, "quantization_config", None) is not None:
            qkwargs = {}
            dtype = None

        self.tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code, use_fast=True)
        self.mdl = AutoModelForCausalLM.from_pretrained(
            cfg.model, device_map=cfg.device_map, torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code, **qkwargs
        ).eval()
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token

        self.messages: List[Dict[str, str]] = []

    # -------- prompt & postprocess --------
    def _render_prompt(self) -> str:
        msgs = self.messages
        if self.cfg.max_turns and self.cfg.max_turns > 0:
            kept_rev = []
            assistants = 0
            for m in reversed(self.messages):
                kept_rev.append(m)
                if m["role"] == "assistant":
                    assistants += 1
                    if assistants >= self.cfg.max_turns:
                        break
            msgs = list(reversed(kept_rev))
            if not msgs or msgs[-1]["role"] != "user":
                msgs.append({"role":"user","content":""})

        parts = []
        if self.system_text:
            parts += [self.system_text.strip(), ""]
        for m in msgs:
            if m["role"] == "user":
                parts += [f"{self.you_label}: {m['content']}", f"{self.asst_label}:"]
            elif m["role"] == "assistant":
                parts += [m["content"], ""]
        if not parts or not parts[-1].endswith(f"{self.asst_label}:"):
            parts.append(f"{self.asst_label}:")
        return "\n".join(parts).rstrip()

    def _postprocess(self, raw: str, last_user: str, prev_assistant: str) -> str:
        s = raw
        # Strip leading role labels
        s = strip_leading_role_labels(s, [self.asst_label, "Assistant", self.you_label, "You", "User", "Me", "Human", "System", "system"])
        # Identity opener (start)
        if self.id_regex:
            stripped = self.id_regex.sub("", s, count=1)
            if stripped.strip(): s = stripped
        s = strip_llm_disclaimers(s)
        # Remove identity lines anywhere
        s = strip_identity_lines_anywhere(s, self.asst_label)
        # Stops & system-echo
        s = self._trim_on_stops(s)
        s = strip_system_echoes(s, self.system_text)

        # Anti-echo last user
        lu = last_user.strip()
        removed_user_echo = False
        if lu:
            for pref in ("", f"{self.you_label}: ", "You: ", "User: "):
                if s.strip().startswith(pref + lu):
                    s = s.strip()[len(pref + lu):].lstrip("\n :")
                    removed_user_echo = True
                    break
        if removed_user_echo and not s.strip():
            for ln in raw.splitlines():
                tln = ln.strip()
                if not tln: continue
                if tln == lu: continue
                if tln.startswith(("Your name is", "You are a", "You are an", "Personality:", "Style:", "Instructions:")):
                    continue
                if self.system_text and tln in self.system_text:
                    continue
                s = tln
                break

        # Optional trims
        r = self.cfg.reply
        if r.no_greetings: s = strip_greetings(s)
        if r.no_praise:    s = strip_praise(s)
        if r.answer_first:
            for ln in s.splitlines():
                if ln.strip():
                    s = first_sentence(ln); break
        if r.no_emojis:    s = strip_emojis(s)

        s = s.lstrip()

        # dedup
        if prev_assistant and s.strip() == prev_assistant.strip():
            s = first_sentence(s) or "(no output)"

        # never empty
        if not s.strip():
            for ln in raw.splitlines():
                ln2 = strip_leading_role_labels(ln, [self.asst_label, "Assistant", self.you_label, "You", "User", "Me", "Human", "System", "system"]).strip()
                if self.id_regex: ln2 = self.id_regex.sub("", ln2, count=1).strip()
                if ln2:
                    s = ln2; break
            if not s.strip():
                s = "(no output)"

        # URLs
        u = self.cfg.url
        s = sanitize_urls(s, no_urls=u.no_urls, allow_domains=u.allow_domains,
                          validate=u.validate_urls, timeout=u.url_timeout,
                          style=u.link_style)
        return s

    def _trim_on_stops(self, s: str) -> str:
        earliest = None
        for t in self.stops:
            idx = s.find(t)
            if idx <= 0: continue
            if s[:idx].strip() == "":  # avoid trimming to empty
                continue
            if earliest is None or idx < earliest:
                earliest = idx
        return s[:earliest] if earliest is not None else s

    # -------- public API --------
    def reset(self):
        self.messages.clear()

    def system(self, text: str):
        self.system_text = text.strip()
        self.reset()
        # recompute identity if using inferred
        if self.cfg.reply.strip_identity is None:
            ident = infer_identity_from_system(self.system_text)
            self.id_regex = make_identity_regex(ident) if ident else None

    def chat(self, user_text: str) -> str:
        user_text = (user_text or "").strip()
        if not user_text:
            return ""
        self.messages.append({"role":"user","content":user_text})
        prompt = self._render_prompt()
        inputs = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.mdl.device) for k, v in inputs.items()}

        d = self.cfg.decode
        do_sample = d.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=d.max_new_tokens,
            do_sample=do_sample,
            eos_token_id=getattr(self.tok, "eos_token_id", None),
            pad_token_id=getattr(self.tok, "pad_token_id", None),
        )
        if do_sample:
            gen_kwargs["temperature"] = d.temperature
            if d.top_p is not None: gen_kwargs["top_p"] = d.top_p
            if d.top_k is not None: gen_kwargs["top_k"] = d.top_k
        if d.repetition_penalty and d.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = d.repetition_penalty

        with torch.inference_mode():
            out = self.mdl.generate(**inputs, **gen_kwargs)
        raw = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        prev_assistant = ""
        for m in reversed(self.messages):
            if m["role"] == "assistant":
                prev_assistant = m["content"]; break

        text = self._postprocess(raw, last_user=user_text, prev_assistant=prev_assistant)
        self.messages.append({"role":"assistant","content":text})
        return text
