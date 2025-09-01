/* global Twilio, fetch */
// Twilio Functions (Node 18.x) - ES5/ES2015-safe syntax
// Single webhook for SMS + WhatsApp. Detect channel via From: "whatsapp:+...".
//
// Env Vars (Twilio Console → Functions → Settings):
//   TODDRIC_API_URL          e.g. https://your-host
//   TAG                      optional, default "[toddric]"
//   BEARER_TOKEN             required if your API expects Authorization: Bearer
//   // Timeouts (stay < 15s total for Twilio):
//   TODDRIC_TIMEOUT_MS       optional, default "12000"
//   OVERALL_TIMEOUT_MS       optional, default "14000"
//   // Token caps per channel (shorter = faster):
//   SMS_MAX_NEW              optional, default "40"
//   WA_MAX_NEW               optional, default "80"
//   // Optional fallback to OpenAI:
//   ENABLE_OPENAI_FALLBACK   optional, "true" to enable (off by default)
//   OPENAI_API_P1 / OPENAI_API_P2 / OPENAI_MODEL
//   // Compliance echo for SMS STOP (Twilio auto-handles STOP on SMS):
//   ECHO_OPT_OUT             optional, "true" to also echo STOP confirmation

exports.handler = function (context, event, callback) {
  var MessagingResponse = Twilio.twiml.MessagingResponse;
  var twiml = new MessagingResponse();

  // --- helpers ---
  function send(xml) {
    var res = new Twilio.Response();
    res.appendHeader("Content-Type", "text/xml");
    res.setBody(xml.toString());
    return callback(null, res);
  }
  function reply(msg) {
    twiml.message(String(msg || "").slice(0, 1400)); // conservative length (works for SMS & WA)
    return send(twiml);
  }
  function empty() { return send(new MessagingResponse()); }

  // --- request context ---
  var TAG = String(context.TAG || "[toddric]");
  var TODDRIC = String(context.TODDRIC_API_URL || "");
  var BEARER = String(context.BEARER_TOKEN || "");

  var TODDRIC_TIMEOUT = parseInt(context.TODDRIC_TIMEOUT_MS || "12000", 10);
  var OVERALL_TIMEOUT = parseInt(context.OVERALL_TIMEOUT_MS || "14000", 10);

  var OPENAI_ENABLED = String(context.ENABLE_OPENAI_FALLBACK || "false") === "true";
  var OPENAI_API_KEY = String(context.OPENAI_API_P1 || "") + String(context.OPENAI_API_P2 || "");
  var OPENAI_MODEL = String(context.OPENAI_MODEL || "gpt-4o-mini");

  var ECHO_OPT_OUT = String(context.ECHO_OPT_OUT || "false") === "true";

  // --- inbound message ---
  var userMsgRaw = String(event.Body || event.message || "");
  var userMsg = userMsgRaw.trim();
  var userMsgU = userMsg.toUpperCase();
  var from = String(event.From || "").trim();
  var isWhatsApp = /^whatsapp:/i.test(from);
  var sessionId = from || require("crypto").randomBytes(8).toString("hex");

  // token caps per channel
  var SMS_MAX_NEW = parseInt(context.SMS_MAX_NEW || "40", 10);
  var WA_MAX_NEW = parseInt(context.WA_MAX_NEW || "80", 10);
  var MAX_NEW_DEFAULT = isWhatsApp ? WA_MAX_NEW : SMS_MAX_NEW;

  // --- compliance keywords ---
  var STOP_WORDS = ["STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT"];
  var START_WORDS = ["START", "YES", "UNSTOP"];
  var HELP_WORDS = ["HELP", "INFO"];

  if (!userMsg) { return reply("Say something to begin. " + TAG); }

  if (!isWhatsApp && STOP_WORDS.indexOf(userMsgU) !== -1) {
    // SMS: Twilio auto-handles STOP; only echo if configured
    if (ECHO_OPT_OUT) { return reply("You’re unsubscribed. Reply START to resubscribe."); }
    return empty();
  }
  if (!isWhatsApp && START_WORDS.indexOf(userMsgU) !== -1) {
    return reply("You’re resubscribed. Text HELP for help. " + TAG);
  }
  if (HELP_WORDS.indexOf(userMsgU) !== -1) {
    var helpText = isWhatsApp
      ? "Help: Ask questions conversationally on WhatsApp. Send 'STOP' on SMS to unsubscribe. " + TAG
      : "Help: Ask questions conversationally. Reply STOP to unsubscribe; START to rejoin. " + TAG;
    return reply(helpText);
  }

  // --- quick diagnostics ---
  if (userMsgU === "DIAG") {
    return reply(
      "diag: toddric=" + (TODDRIC ? "yes" : "no") +
      " overall=" + OVERALL_TIMEOUT + "ms" +
      " t_api=" + TODDRIC_TIMEOUT + "ms" +
      " max_new=" + MAX_NEW_DEFAULT +
      " channel=" + (isWhatsApp ? "wa" : "sms")
    );
  }
  if (userMsgU === "WHOAMI") {
    if (!TODDRIC) return reply("model: (no server)");
    return fetch(TODDRIC.replace(/\/+$/, "") + "/whoami")
      .then(function (r) { return r.json(); })
      .then(function (j) { return reply(j && j.model ? ("model: " + j.model) : "model: unknown"); })
      .catch(function () { return reply("model: unknown"); });
  }
  if (userMsgU === "RESET") {
    if (TODDRIC) {
      fetch(TODDRIC.replace(/\/+$/, "") + "/reset", {
        method: "POST",
        headers: Object.assign({ "Content-Type": "application/json" }, BEARER ? { Authorization: "Bearer " + BEARER } : {}),
        body: JSON.stringify({ session_id: sessionId })
      }).catch(function () { /* noop */ });
    }
    return reply("Okay, cleared our chat. " + TAG);
  }

  // --- SMS vs WA output style ---
  var isKnowledge = /\b(who|what|tell me about|know about|history|biography|bio|explain)\b/i.test(userMsg);
  var MAX_NEW_LOCAL = isKnowledge ? Math.min(isWhatsApp ? 80 : 60, MAX_NEW_DEFAULT) : MAX_NEW_DEFAULT;
  var INSTRUCTION = isKnowledge
    ? "Reply in 2–3 concise sentences, messaging-friendly, no markdown, no disclaimers."
    : "Be concise and messaging-friendly.";

  function withTimeout(promiseFactory, ms, label) {
    var AbortC = (typeof AbortController !== "undefined" ? AbortController : require("abort-controller"));
    var ctl = new AbortC();
    var timer = setTimeout(function () { try { ctl.abort(); } catch (e) {} }, ms);
    return promiseFactory(ctl.signal)
      .finally(function () { clearTimeout(timer); })
      .catch(function (e) { throw new Error((label || "timeout") + ":" + (e && e.message || e)); });
  }

  function callToddric(signal) {
    if (!TODDRIC) { return Promise.reject(new Error("toddric:not_configured")); }
    var url = TODDRIC.replace(/\/+$/, "") + "/chat";
    var payload = {
      message: userMsg,
      session_id: sessionId,
      max_new_tokens: MAX_NEW_LOCAL,
      temperature: 0.0,
      style: isWhatsApp ? "wa_short" : "sms_short",
      instruction: INSTRUCTION
    };
    return fetch(url, {
      method: "POST",
      headers: Object.assign({ "Content-Type": "application/json" }, BEARER ? { Authorization: "Bearer " + BEARER } : {}),
      body: JSON.stringify(payload),
      signal: signal
    }).then(function (r) {
      if (r.ok) return r.json();
      return r.text().then(function (t) { throw new Error("toddric:" + r.status + ":" + String(t).slice(0, 300)); });
    }).then(function (j) {
      var txt = (j && (j.text || j.reply || j.output)) ? String(j.text || j.reply || j.output).trim() : "";
      if (!txt) { throw new Error("toddric:empty"); }
      return txt;
    });
  }

  function callOpenAI(signal) {
    if (!OPENAI_ENABLED || !OPENAI_API_KEY) { return Promise.reject(new Error("fallback:disabled")); }
    var payload = {
      model: OPENAI_MODEL,
      messages: [
        { role: "system", content: INSTRUCTION },
        { role: "user", content: userMsg }
      ],
      max_tokens: Math.max(40, Math.min(isWhatsApp ? 160 : 120, MAX_NEW_LOCAL)),
      temperature: 0.5
    };
    return fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + OPENAI_API_KEY },
      body: JSON.stringify(payload),
      signal: signal
    }).then(function (r) {
      if (r.ok) return r.json();
      return r.text().then(function (t) { throw new Error("openai:" + r.status + ":" + String(t).slice(0, 300)); });
    }).then(function (j) {
      return (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content)
        ? j.choices[0].message.content.trim()
        : "OK";
    });
  }

  var finished = false;
  // Early progress ping (~8s) so Twilio has *some* response
  var progressTimer = setTimeout(function () {
    if (!finished) {
      reply("Thinking… give me a moment. " + TAG);
      finished = true;
    }
  }, 8000);

  withTimeout(callToddric, TODDRIC_TIMEOUT, "toddric_timeout").then(function (txt) {
    if (!finished) {
      finished = true;
      clearTimeout(progressTimer);
      reply(String(txt) + " " + TAG);
    }
  }).catch(function () {
    if (!OPENAI_ENABLED) {
      if (!finished) {
        finished = true;
        clearTimeout(progressTimer);
        reply("Sorry, that took too long. Try a shorter ask. " + TAG);
      }
      return;
    }
    var fbTimeout = Math.min(4000, Math.max(2000, OVERALL_TIMEOUT - 2000));
    withTimeout(callOpenAI, fbTimeout, "fallback_timeout").then(function (txt) {
      if (!finished) {
        finished = true;
        clearTimeout(progressTimer);
        reply(String(txt) + " " + TAG);
      }
    }).catch(function () {
      if (!finished) {
        finished = true;
        clearTimeout(progressTimer);
        reply("Timed out. Try again with fewer details. " + TAG);
      }
    });
  });
};
