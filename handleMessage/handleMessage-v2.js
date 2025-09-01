// Env Vars (Twilio Console → Functions → Settings):
//   TODDRIC_API_URL          e.g. https://your-host
//   TAG                      optional, default "[toddric]"
//   SMS_MAX_NEW              optional, default "40" (patched lower for speed)
//   TODDRIC_TIMEOUT_MS       optional, default "12000" (patched higher)
//   OVERALL_TIMEOUT_MS       optional, default "14000" (patched higher)
//   BEARER_TOKEN             the token your API expects (required for auth)
//   ENABLE_OPENAI_FALLBACK   optional, "true" to enable (off by default)
//   OPENAI_API_P1 / OPENAI_API_P2 / OPENAI_MODEL  (if fallback enabled)
//   ECHO_OPT_OUT             optional "true" to echo STOP confirmation (Twilio already sends one)

exports.handler = function (context, event, callback) {
    const MessagingResponse = Twilio.twiml.MessagingResponse;
    const twiml = new MessagingResponse();

    const TAG = String(context.TAG || "[toddric]");
    const userMsgRaw = String(event.Body || event.message || "");
    const userMsg = userMsgRaw.trim();
    const userMsgU = userMsg.toUpperCase();
    const from = String(event.From || "").trim();
    const sessionId = from || require("crypto").randomBytes(8).toString("hex");

    const TODDRIC = String(context.TODDRIC_API_URL || "");
    const MAX_NEW_DEFAULT = parseInt(context.SMS_MAX_NEW || "40", 10); // patched default lower
    const TODDRIC_TIMEOUT = parseInt(context.TODDRIC_TIMEOUT_MS || "12000", 10); // patched higher
    const OVERALL_TIMEOUT = parseInt(context.OVERALL_TIMEOUT_MS || "14000", 10); // patched higher
    const BEARER = String(context.BEARER_TOKEN || "");

    const ENABLE_FALLBACK = String(context.ENABLE_OPENAI_FALLBACK || "false") === "true";
    const OPENAI_API_KEY = (context.OPENAI_API_P1 || "") + (context.OPENAI_API_P2 || "");
    const OPENAI_MODEL = context.OPENAI_MODEL || "gpt-4o-mini";

    const ECHO_OPT_OUT = String(context.ECHO_OPT_OUT || "false") === "true";

    const STOP_WORDS  = ["STOP","STOPALL","UNSUBSCRIBE","CANCEL","END","QUIT"];
    const START_WORDS = ["START","YES","UNSTOP"];
    const HELP_WORDS  = ["HELP","INFO"];

    function send(xml) {
      const res = new Twilio.Response();
      res.appendHeader("Content-Type", "text/xml");
      res.setBody(xml.toString());
      return callback(null, res);
    }
    function reply(msg) {
      twiml.message(String(msg || "").slice(0, 1400));
      return send(twiml);
    }
    function empty() { return send(new MessagingResponse()); }

    // --- Compliance keywords ---
    if (STOP_WORDS.includes(userMsgU)) {
      if (ECHO_OPT_OUT) return reply("You’re unsubscribed. Reply START to resubscribe.");
      return empty();
    }
    if (START_WORDS.includes(userMsgU)) {
      return reply("You’re resubscribed. Text HELP for help. " + TAG);
    }
    if (HELP_WORDS.includes(userMsgU)) {
      return reply("Help: Ask questions conversationally. Reply STOP to unsubscribe; START to rejoin. " + TAG);
    }

    // quick diags
    if (userMsgU === "DIAG") {
      return reply(`diag: toddric=${TODDRIC ? "yes" : "no"} overall=${OVERALL_TIMEOUT}ms t_api=${TODDRIC_TIMEOUT}ms max_new=${MAX_NEW_DEFAULT}`);
    }
    if (userMsgU === "WHOAMI") {
      if (!TODDRIC) return reply("model: (no server)");
      return fetch(TODDRIC.replace(/\/+$/,"") + "/whoami")
        .then(r => r.json()).then(j => reply(j && j.model ? `model: ${j.model}` : "model: unknown"))
        .catch(() => reply("model: unknown"));
    }
    if (userMsgU === "RESET") {
      if (TODDRIC) {
        fetch(TODDRIC.replace(/\/+$/,"") + "/reset", {
          method: "POST",
          headers: {"Content-Type":"application/json", ...(BEARER ? {"Authorization":"Bearer "+BEARER} : {})},
          body: JSON.stringify({ session_id: sessionId })
        }).catch(()=>{});
      }
      return reply(`Okay, cleared our chat. ${TAG}`);
    }

    // Heuristic: knowledgey prompts → shrink SMS
    const isKnowledge = /\b(who|what|tell me about|know about|history|biography|bio|explain)\b/i.test(userMsg);
    const MAX_NEW_LOCAL = isKnowledge ? Math.min(60, MAX_NEW_DEFAULT) : MAX_NEW_DEFAULT;
    const INSTRUCTION = isKnowledge
      ? "Reply in 2–3 concise sentences, SMS-friendly, no markdown, no disclaimers."
      : "Be concise and SMS-friendly.";

    function withTimeout(promiseFactory, ms, label) {
      const AbortController = global.AbortController || require("abort-controller");
      const ctl = new AbortController();
      const timer = setTimeout(() => { try { ctl.abort(); } catch(e){} }, ms);
      return promiseFactory(ctl.signal).finally(() => clearTimeout(timer))
        .catch(e => { throw new Error((label||"timeout")+":"+(e && e.message || e)); });
    }

    function callToddric(signal) {
      if (!TODDRIC) return Promise.reject(new Error("toddric:not_configured"));
      const url = TODDRIC.replace(/\/+$/,"") + "/chat";
      const payload = {
        message: userMsg,
        session_id: sessionId,
        max_new_tokens: MAX_NEW_LOCAL,
        temperature: 0.0,
        style: "sms_short",
        instruction: INSTRUCTION
      };
      return fetch(url, {
        method: "POST",
        headers: {
          "Content-Type":"application/json",
          ...(BEARER ? {"Authorization":"Bearer "+BEARER} : {}),
        },
        body: JSON.stringify(payload),
        signal
      })
      .then(r => r.ok ? r.json() : r.text().then(t => { throw new Error("toddric:"+r.status+":"+String(t).slice(0,300)); }))
      .then(j => {
        const txt = (j && (j.text || j.reply || j.output)) ? String(j.text || j.reply || j.output).trim() : "";
        if (!txt) throw new Error("toddric:empty");
        return txt;
      });
    }

    function callOpenAI(signal) {
      if (!ENABLE_FALLBACK || !OPENAI_API_KEY) return Promise.reject(new Error("fallback:disabled"));
      const payload = {
        model: OPENAI_MODEL,
        messages: [
          { role: "system", content: INSTRUCTION },
          { role: "user", content: userMsg }
        ],
        max_tokens: Math.max(40, Math.min(120, MAX_NEW_LOCAL)),
        temperature: 0.5
      };
      return fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type":"application/json", "Authorization":"Bearer "+OPENAI_API_KEY },
        body: JSON.stringify(payload),
        signal
      })
      .then(r => r.ok ? r.json() : r.text().then(t => { throw new Error("openai:"+r.status+":"+String(t).slice(0,300)); }))
      .then(j => (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) ? j.choices[0].message.content.trim() : "OK");
    }

    let finished = false;
    // Progress ping earlier (patched ~8s)
    const progressTimer = setTimeout(() => {
      if (!finished) {
        reply("Thinking… give me a moment. " + TAG);
        finished = true;
      }
    }, 8000);

    withTimeout(callToddric, TODDRIC_TIMEOUT, "toddric_timeout")
      .then(txt => { if (!finished) { finished = true; clearTimeout(progressTimer); reply(`${txt} ${TAG}`); } })
      .catch(() => {
        if (!ENABLE_FALLBACK) {
          if (!finished) { finished = true; clearTimeout(progressTimer); reply("Sorry, that took too long. Try a shorter ask. " + TAG); }
          return;
        }
        const fbTimeout = Math.min(4000, Math.max(2000, OVERALL_TIMEOUT - 2000));
        withTimeout(callOpenAI, fbTimeout, "fallback_timeout")
          .then(txt => { if (!finished) { finished = true; clearTimeout(progressTimer); reply(`${txt} ${TAG}`); } })
          .catch(() => { if (!finished) { finished = true; clearTimeout(progressTimer); reply("Timed out. Try again with fewer details. " + TAG); } });
      });
  };
