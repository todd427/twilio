/* global Twilio, fetch */
// Twilio Functions ultra-compatible build (Node 18.x)
// - No arrow funcs, no template strings, no spread, no finally, no AbortController
// - Single webhook for SMS + WhatsApp
//
// Env Vars (Functions -> Settings):
//   TODDRIC_API_URL, TAG, BEARER_TOKEN
//   TODDRIC_TIMEOUT_MS (default 12000), OVERALL_TIMEOUT_MS (default 14000)
//   SMS_MAX_NEW (default 40), WA_MAX_NEW (default 80)
//   ENABLE_OPENAI_FALLBACK ("true"/"false"), OPENAI_API_P1, OPENAI_API_P2, OPENAI_MODEL
//   ECHO_OPT_OUT ("true"/"false")

exports.handler = function (context, event, callback) {
    var MessagingResponse = Twilio.twiml.MessagingResponse;
    var twiml = new MessagingResponse();
  
    function send(xml) {
      var res = new Twilio.Response();
      res.appendHeader("Content-Type", "text/xml");
      res.setBody(xml.toString());
      return callback(null, res);
    }
    function reply(msg) {
      twiml.message(String(msg || "").slice(0, 1400));
      return send(twiml);
    }
    function empty() { return send(new MessagingResponse()); }
  
    // --- config ---
    var TAG = String(context.TAG || "[toddric]");
    var TODDRIC = String(context.TODDRIC_API_URL || "");
    var BEARER = String(context.BEARER_TOKEN || "");
    var TODDRIC_TIMEOUT = parseInt(context.TODDRIC_TIMEOUT_MS || "12000", 10);
    var OVERALL_TIMEOUT = parseInt(context.OVERALL_TIMEOUT_MS || "14000", 10);
    var SMS_MAX_NEW = parseInt(context.SMS_MAX_NEW || "40", 10);
    var WA_MAX_NEW = parseInt(context.WA_MAX_NEW || "80", 10);
    var OPENAI_ENABLED = String(context.ENABLE_OPENAI_FALLBACK || "false") === "true";
    var OPENAI_KEY = String(context.OPENAI_API_P1 || "") + String(context.OPENAI_API_P2 || "");
    var OPENAI_MODEL = String(context.OPENAI_MODEL || "gpt-4o-mini");
    var ECHO_OPT_OUT = String(context.ECHO_OPT_OUT || "false") === "true";
  
    // --- inbound ---
    var userMsgRaw = String(event.Body || event.message || "");
    var userMsg = userMsgRaw.trim();
    var userMsgU = userMsg.toUpperCase();
    var from = String(event.From || "").trim();
    var isWhatsApp = /^whatsapp:/i.test(from);
    var sessionId = from || require("crypto").randomBytes(8).toString("hex");
  
    if (!userMsg) { return reply("Say something to begin. " + TAG); }
  
    // --- compliance (SMS only for STOP/START echo) ---
    var STOP_WORDS = ["STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT"];
    var START_WORDS = ["START", "YES", "UNSTOP"];
    var HELP_WORDS = ["HELP", "INFO"];
  
    if (!isWhatsApp && STOP_WORDS.indexOf(userMsgU) !== -1) {
      if (ECHO_OPT_OUT) { return reply("You're unsubscribed. Reply START to resubscribe."); }
      return empty();
    }
    if (!isWhatsApp && START_WORDS.indexOf(userMsgU) !== -1) { return reply("You're resubscribed. Text HELP for help. " + TAG); }
    if (HELP_WORDS.indexOf(userMsgU) !== -1) {
      var helpText = isWhatsApp ? "Help: Ask questions on WhatsApp. Send STOP on SMS to unsubscribe. " + TAG : "Help: Ask questions conversationally. Reply STOP to unsubscribe; START to rejoin. " + TAG;
      return reply(helpText);
    }
  
    // --- diagnostics ---
    if (userMsgU === "DIAG") {
      return reply("diag: toddric=" + (TODDRIC ? "yes" : "no") + " overall=" + OVERALL_TIMEOUT + "ms t_api=" + TODDRIC_TIMEOUT + "ms max_new=" + (isWhatsApp ? WA_MAX_NEW : SMS_MAX_NEW) + " channel=" + (isWhatsApp ? "wa" : "sms"));
    }
    if (userMsgU === "WHOAMI") {
      if (!TODDRIC) return reply("model: (no server)");
      return fetch(TODDRIC.replace(/\/+$/, "") + "/whoami")
        .then(function (r) { return r.json(); })
        .then(function (j) { return reply(j && j.model ? "model: " + j.model : "model: unknown"); })
        .catch(function () { return reply("model: unknown"); });
    }
    if (userMsgU === "RESET") {
      if (TODDRIC) {
        var hdrR = { "Content-Type": "application/json" };
        if (BEARER) { hdrR.Authorization = "Bearer " + BEARER; }
        fetch(TODDRIC.replace(/\/+$/, "") + "/reset", { method: "POST", headers: hdrR, body: JSON.stringify({ session_id: sessionId }) }).catch(function () {});
      }
      return reply("Okay, cleared our chat. " + TAG);
    }
  
    // --- style & caps ---
    var isKnowledge = /\b(who|what|tell me about|know about|history|biography|bio|explain)\b/i.test(userMsg);
    var MAX_NEW = isWhatsApp ? WA_MAX_NEW : SMS_MAX_NEW;
    var MAX_NEW_LOCAL = isKnowledge ? Math.min(isWhatsApp ? 80 : 60, MAX_NEW) : MAX_NEW;
    var INSTRUCTION = isKnowledge ? "Reply in 2-3 concise sentences, messaging-friendly, no markdown, no disclaimers." : "Be concise and messaging-friendly.";
  
    // --- tiny timeout wrapper (no abort, no finally) ---
    function withTimeout(promiseFactory, ms, label) {
      return new Promise(function (resolve, reject) {
        var finished = false;
        var timer = setTimeout(function () { if (!finished) { finished = true; reject(new Error(label || "timeout")); } }, ms);
        promiseFactory().then(function (v) {
          if (!finished) { finished = true; clearTimeout(timer); resolve(v); }
        }, function (e) {
          if (!finished) { finished = true; clearTimeout(timer); reject(e); }
        });
      });
    }
  
    function callToddric() {
      if (!TODDRIC) { return Promise.reject(new Error("toddric:not_configured")); }
      var url = TODDRIC.replace(/\/+$/, "") + "/chat";
      var payload = { message: userMsg, session_id: sessionId, max_new_tokens: MAX_NEW_LOCAL, temperature: 0.0, style: isWhatsApp ? "wa_short" : "sms_short", instruction: INSTRUCTION };
      var hdr = { "Content-Type": "application/json" };
      if (BEARER) { hdr.Authorization = "Bearer " + BEARER; }
      return fetch(url, { method: "POST", headers: hdr, body: JSON.stringify(payload) })
        .then(function (r) { if (r.ok) { return r.json(); } return r.text().then(function (t) { throw new Error("toddric:" + r.status + ":" + String(t).slice(0, 300)); }); })
        .then(function (j) { var txt = (j && (j.text || j.reply || j.output)) ? String(j.text || j.reply || j.output).trim() : ""; if (!txt) { throw new Error("toddric:empty"); } return txt; });
    }
  
    function callOpenAI() {
      if (!OPENAI_ENABLED || !OPENAI_KEY) { return Promise.reject(new Error("fallback:disabled")); }
      var payload = { model: OPENAI_MODEL, messages: [ { role: "system", content: INSTRUCTION }, { role: "user", content: userMsg } ], max_tokens: Math.max(40, Math.min(isWhatsApp ? 160 : 120, MAX_NEW_LOCAL)), temperature: 0.5 };
      return fetch("https://api.openai.com/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json", Authorization: "Bearer " + OPENAI_KEY }, body: JSON.stringify(payload) })
        .then(function (r) { if (r.ok) { return r.json(); } return r.text().then(function (t) { throw new Error("openai:" + r.status + ":" + String(t).slice(0, 300)); }); })
        .then(function (j) { return (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) ? j.choices[0].message.content.trim() : "OK"; });
    }
  
    // --- run with budgets ---
    var startMs = Date.now();
    var responded = false;
    function safeReplyOnce(text) {
      if (!responded) { responded = true; reply(String(text) + " " + TAG); }
    }
    // Overall guard: send timeout if nothing returned by OVERALL_TIMEOUT
    setTimeout(function () { if (!responded) { safeReplyOnce("Sorry, that took too long. Try a shorter ask."); } }, OVERALL_TIMEOUT);
  
    withTimeout(function () { return callToddric(); }, TODDRIC_TIMEOUT, "toddric_timeout").then(function (txt) {
      if (!responded) { safeReplyOnce(txt); }
    }, function () {
      // Toddric failed/slow: attempt fallback within remaining budget
      var elapsed = Date.now() - startMs;
      var remaining = OVERALL_TIMEOUT - elapsed - 200; // save a little cushion
      if (!OPENAI_ENABLED || remaining < 1500) { if (!responded) { safeReplyOnce("Sorry, that took too long. Try a shorter ask."); } return; }
      var fbBudget = Math.max(1500, Math.min(4000, remaining));
      withTimeout(function () { return callOpenAI(); }, fbBudget, "fallback_timeout").then(function (txt2) {
        if (!responded) { safeReplyOnce(txt2); }
      }, function () {
        if (!responded) { safeReplyOnce("Timed out. Try again with fewer details."); }
      });
    });
  };
  