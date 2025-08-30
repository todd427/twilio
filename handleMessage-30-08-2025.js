// Env Vars:
//   TODDRIC_API_URL          e.g. https://your-host
//   TAG                      optional, e.g. "[toddric]"
//   SMS_MAX_NEW              optional, default "120"
//   TODDRIC_TIMEOUT_MS       optional, default "7000"
//   OVERALL_TIMEOUT_MS       optional, default "8500"
//   ENABLE_OPENAI_FALLBACK   optional, "true" to enable
//   OPENAI_API_P1 / OPENAI_API_P2 / OPENAI_MODEL (if fallback enabled)

exports.handler = function (context, event, callback) {
  const MessagingResponse = Twilio.twiml.MessagingResponse;
  const twiml = new MessagingResponse();

  const TAG = String(context.TAG || "[toddric]");
  const userMsg = String(event.Body || event.message || "").trim();
  const from = String(event.From || "").trim();
  const sessionId = from || require("crypto").randomBytes(8).toString("hex");

  const TODDRIC = String(context.TODDRIC_API_URL || "");
  const MAX_NEW = parseInt(context.SMS_MAX_NEW || "120", 10);
  const TODDRIC_TIMEOUT = parseInt(context.TODDRIC_TIMEOUT_MS || "7000", 10);     // < 10s
  const OVERALL_TIMEOUT = parseInt(context.OVERALL_TIMEOUT_MS || "8500", 10);     // global guard

  const ENABLE_FALLBACK = String(context.ENABLE_OPENAI_FALLBACK || "false") === "true";
  const OPENAI_API_KEY = (context.OPENAI_API_P1 || "") + (context.OPENAI_API_P2 || "");
  const OPENAI_MODEL = context.OPENAI_MODEL || "gpt-4o-mini";

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

  // quick diags
  if (userMsg.toLowerCase() === "diag") {
    return reply(
      `diag: toddric=${TODDRIC ? "yes" : "no"} overall=${OVERALL_TIMEOUT}ms t_api=${TODDRIC_TIMEOUT}ms max_new=${MAX_NEW}`
    );
  }
  if (userMsg.toLowerCase() === "whoami") {
    if (!TODDRIC) return reply("model: (no server)");
    return fetch(TODDRIC.replace(/\/+$/,"") + "/whoami")
      .then(r => r.json()).then(j => reply(j && j.model ? `model: ${j.model}` : "model: unknown"))
      .catch(() => reply("model: unknown"));
  }
  if (userMsg.toLowerCase() === "reset") {
    if (TODDRIC) {
      fetch(TODDRIC.replace(/\/+$/,"") + "/reset", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ session_id: sessionId })
      }).catch(()=>{});
    }
    return reply(`Okay, cleared our chat. ${TAG}`);
  }

  function withTimeout(promise, ms, label) {
    const AbortController = global.AbortController || require("abort-controller");
    const ctl = new AbortController();
    const timer = setTimeout(() => { try { ctl.abort(); } catch(e){} }, ms);
    return promise(ctl.signal).finally(() => clearTimeout(timer))
      .catch(e => { throw new Error((label||"timeout") + ":" + (e && e.message || e)); });
  }

  function callToddric(signal) {
    if (!TODDRIC) return Promise.reject(new Error("toddric:not_configured"));
    const url = TODDRIC.replace(/\/+$/,"") + "/chat";
    const payload = {
      // keep SMS responses tight
      max_new_tokens: MAX_NEW,
      temperature: 0.2,
      top_p: 0.9,
      message: userMsg,
      session_id: sessionId
    };
    return fetch(url, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
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
        { role: "system", content: "You are a concise, SMS-friendly assistant." },
        { role: "user", content: userMsg }
      ],
      max_tokens: Math.max(80, Math.min(180, MAX_NEW)),
      temperature: 0.5
    };
    return fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type":"application/json",
        "Authorization":"Bearer " + OPENAI_API_KEY
      },
      body: JSON.stringify(payload),
      signal
    })
    .then(r => r.ok ? r.json() : r.text().then(t => { throw new Error("openai:"+r.status+":"+String(t).slice(0,300)); }))
    .then(j => (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) ? j.choices[0].message.content.trim() : "OK");
  }

  // overall watchdog to guarantee TwiML in time
  let finished = false;
  const overallTimer = setTimeout(() => {
    if (!finished) {
      finished = true;
      reply("Still working—try rephrasing or ask a shorter question. " + TAG);
    }
  }, OVERALL_TIMEOUT - 100); // a small safety margin

  // run Toddric with its own cap; only try fallback if enabled and time remains
  withTimeout(callToddric, TODDRIC_TIMEOUT, "toddric_timeout")
    .then(txt => {
      if (finished) return;
      finished = true;
      clearTimeout(overallTimer);
      reply(`${txt} ${TAG}`);
    })
    .catch(() => {
      if (!ENABLE_FALLBACK) {
        if (!finished) { finished = true; clearTimeout(overallTimer); reply("Sorry, that one took too long. Try a shorter ask. " + TAG); }
        return;
      }
      // leave a little headroom for fallback
      const slack = Math.max(1200, OVERALL_TIMEOUT - (Date.now() % 1000000)); // conservative
      const fbTimeout = Math.min(3000, slack);
      withTimeout(callOpenAI, fbTimeout, "fallback_timeout")
        .then(txt => {
          if (finished) return;
          finished = true; clearTimeout(overallTimer);
          reply(`${txt} ${TAG}`);
        })
        .catch(() => {
          if (!finished) { finished = true; clearTimeout(overallTimer); reply("Timed out. Try again with fewer details. " + TAG); }
        });
    });
};
