/* global Twilio, fetch */
// hm-twilio-safe2.js — SMS + WhatsApp, word-safe chunking, MORE pagination,
// House Rules on HELP, RESET gives welcome, no TAG.
// Runtime: Twilio Functions (Node 18.x)

exports.handler = function (context, event, callback) {
    var MessagingResponse = Twilio.twiml.MessagingResponse;
    var twiml = new MessagingResponse();
  
    // ----- helpers -----
    function send(xml) {
      var res = new Twilio.Response();
      res.appendHeader("Content-Type", "text/xml");
      res.setBody(xml.toString());
      return callback(null, res);
    }
  
    function isGSM7(s) {
      for (var i = 0; i < s.length; i++) if (s.charCodeAt(i) > 127) return false;
      return true;
    }
  
    // Word-safe segmenting for SMS/WA
    function sendChunked(text) {
      var s = String(text || "").replace(/\s+/g, " ").trim();
      if (!s) { twiml.message(""); return; }
      var gsm = isGSM7(s);
      var seg = gsm ? 153 : 67;
      while (s.length > 0) {
        var take = Math.min(seg, s.length);
        var part = s.slice(0, take);
        if (take < s.length && /[A-Za-z0-9]/.test(s.charAt(take))) {
          var back = part.search(/[ \t\n\r.,;:!?][^ \t\n\r.,;:!?]*$/);
          if (back > 0) part = part.slice(0, back + 1);
        }
        twiml.message(part.trim());
        s = s.slice(part.length);
      }
    }
  
    function reply(msg) { sendChunked(msg); return send(twiml); }
  
    function rejectNow(msg) {
      return new Promise(function (_, reject) { reject(new Error(msg)); });
    }
  
    function withTimeout(promiseFactory, ms, label) {
      return new Promise(function (resolve, reject) {
        var finished = false;
        var timer = setTimeout(function () {
          if (!finished) { finished = true; reject(new Error(label || "timeout")); }
        }, ms);
        promiseFactory().then(function (v) {
          if (!finished) { finished = true; clearTimeout(timer); resolve(v); }
        }, function (e) {
          if (!finished) { finished = true; clearTimeout(timer); reject(e); }
        });
      });
    }
  
    // ----- config -----
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
  
    // ----- inbound / session -----
    var userMsgRaw = String(event.Body || event.message || "");
    var userMsg = userMsgRaw.trim();
    var userMsgU = userMsg.toUpperCase();
    var from = String(event.From || "").trim();
    var isWhatsApp = /^whatsapp:/i.test(from);
    var sessionId = from || require("crypto").randomBytes(8).toString("hex");
  
    if (!userMsg) { return reply("Say something to begin."); }
  
    // ----- compliance -----
    var STOP_WORDS = ["STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT"];
    var START_WORDS = ["START", "YES", "UNSTOP"];
  
    if (!isWhatsApp && STOP_WORDS.indexOf(userMsgU) !== -1) {
      if (ECHO_OPT_OUT) { return reply("You’re unsubscribed. Reply START to resubscribe."); }
      return send(new MessagingResponse());
    }
    if (!isWhatsApp && START_WORDS.indexOf(userMsgU) !== -1) {
      return reply("You’re resubscribed. Text HELP for help.");
    }
  
    // ----- HELP / House Rules -----
    if (/^\/?help$/i.test(userMsg)) {
      var rules =
        "House Rules:\n" +
        "• TIME = current time\n" +
        "• TZ Europe/Dublin = set timezone\n" +
        "• MORE = continue long reply\n" +
        "• STATS = usage metrics\n" +
        "• RESET = clear session\n" +
        "• STOP/START = opt out/in";
      return reply(rules);
    }
  
    // ----- diagnostics -----
    if (userMsgU === "DIAG") {
      return reply(
        "diag: toddric=" + (TODDRIC ? "yes" : "no") +
        " overall=" + OVERALL_TIMEOUT + "ms" +
        " t_api=" + TODDRIC_TIMEOUT + "ms" +
        " max_new=" + (isWhatsApp ? WA_MAX_NEW : SMS_MAX_NEW) +
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
  
    // ----- RESET -----
    if (/^\/?reset$/i.test(userMsg)) {
      if (TODDRIC) {
        var hdrR = { "Content-Type": "application/json" };
        if (BEARER) { hdrR.Authorization = "Bearer " + BEARER; }
        fetch(TODDRIC.replace(/\/+$/, "") + "/reset", {
          method: "POST",
          headers: hdrR,
          body: JSON.stringify({ session_id: sessionId })
        }).catch(function () {});
      }
      return reply("Welcome to Toddric! 👋 Ask me anything in 1–2 sentences. Text HELP for House Rules.");
    }
  
    // ----- metrics over chat -----
    if (userMsgU === "STATS" || userMsgU === "METRICS") {
      if (!TODDRIC) return reply("No server.");
      return fetch(TODDRIC.replace(/\/+$/, "") + "/metrics")
        .then(function (r) { return r.json(); })
        .then(function (m) {
          var lat = m && m.chat_latency_ms ? m.chat_latency_ms : { avg:0,p50:0,p95:0,p99:0,samples:0 };
          var ch = m && m.channels ? m.channels : { sms:{chat_requests_total:0,chat_errors_total:0}, wa:{chat_requests_total:0,chat_errors_total:0} };
          var txt =
            "Reqs:" + m.requests_total + " Chat:" + m.chat_requests_total + " Err:" + m.chat_errors_total + " | " +
            "ms avg/p50/p95/p99:" + lat.avg + "/" + lat.p50 + "/" + lat.p95 + "/" + lat.p99 + " (n=" + lat.samples + ") | " +
            "SMS:" + ch.sms.chat_requests_total + "/" + ch.sms.chat_errors_total + " " +
            "WA:"   + ch.wa.chat_requests_total  + "/" + ch.wa.chat_errors_total;
          return reply(txt);
        })
        .catch(function () { return reply("metrics unavailable"); });
    }
  
    // ----- MORE pagination -----
    function fetchMore(sessionId, maxChars) {
      var url = TODDRIC.replace(/\/+$/, "") + "/more";
      var body = JSON.stringify({ session_id: sessionId, max_chars: maxChars || (isWhatsApp ? 400 : 300) });
      var hdr = { "Content-Type": "application/json" };
      if (BEARER) hdr.Authorization = "Bearer " + BEARER;
      return fetch(url, { method: "POST", headers: hdr, body: body }).then(function (r) { return r.json(); });
    }
  
    if (userMsgU === "MORE") {
      if (!TODDRIC) return reply("No server.");
      var maxChars = isWhatsApp ? 400 : 300;
      return fetchMore(sessionId, maxChars).then(function (j) {
        if (!j || !j.chunk) return reply("Nothing more.");
        sendChunked(j.chunk);
        if (!j.done) twiml.message("Reply MORE for the next part.");
        return send(twiml);
      }).catch(function () { return reply("Couldn’t fetch more right now."); });
    }
  
    // ----- style & caps -----
    var isKnowledge = /\b(who|what|tell me about|know about|history|biography|bio|explain)\b/i.test(userMsg);
    var MAX_NEW = isWhatsApp ? WA_MAX_NEW : SMS_MAX_NEW;
    var MAX_NEW_LOCAL = isKnowledge ? Math.min(isWhatsApp ? 80 : 60, MAX_NEW) : MAX_NEW;
    var INSTRUCTION = isKnowledge
      ? "Be concise and messaging-friendly. 1–2 sentences. No markdown or disclaimers."
      : "Be concise and messaging-friendly.";
  
    // ----- API calls -----
    function callToddric() {
      if (!TODDRIC) { return rejectNow("toddric:not_configured"); }
      var url = TODDRIC.replace(/\/+$/, "") + "/chat";
      var payload = {
        message: userMsg,
        session_id: sessionId,
        channel: isWhatsApp ? "wa" : "sms",
        max_new_tokens: MAX_NEW_LOCAL,
        temperature: 0.0,
        style: isWhatsApp ? "wa_short" : "sms_short",
        instruction: INSTRUCTION
      };
      var hdr = { "Content-Type": "application/json" };
      if (BEARER) { hdr.Authorization = "Bearer " + BEARER; }
      return fetch(url, { method: "POST", headers: hdr, body: JSON.stringify(payload) })
        .then(function (r) {
          if (r.ok) return r.json();
          return r.text().then(function (t) { throw new Error("toddric:" + r.status + ":" + String(t).slice(0, 300)); });
        })
        .then(function (j) {
          var txt = (j && (j.text || j.reply || j.output)) ? String(j.text || j.reply || j.output).trim() : "";
          if (!txt) { throw new Error("toddric:empty"); }
          return txt;
        });
    }
  
    function callOpenAI() {
      if (!OPENAI_ENABLED || !OPENAI_KEY) { return rejectNow("fallback:disabled"); }
      var payload = {
        model: OPENAI_MODEL,
        messages: [
          { role: "system", content: INSTRUCTION },
          { role: "user", content: userMsg }
        ],
        max_tokens: Math.max(40, Math.min(isWhatsApp ? 160 : 120, MAX_NEW_LOCAL)),
        temperature: 0.3
      };
      return fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + OPENAI_KEY },
        body: JSON.stringify(payload)
      })
      .then(function (r) {
        if (r.ok) return r.json();
        return r.text().then(function (t) { throw new Error("openai:" + r.status + ":" + String(t).slice(0, 300)); });
      })
      .then(function (j) {
        return (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content)
          ? j.choices[0].message.content.trim()
          : "OK";
      });
    }
  
    // ----- timeouts + orchestration -----
    var startMs = Date.now();
    var responded = false;
    function safeReplyOnce(text) {
      if (!responded) { responded = true; reply(String(text)); }
    }
  
    setTimeout(function () {
      if (!responded) { safeReplyOnce("Sorry, that took too long. Try a shorter ask."); }
    }, OVERALL_TIMEOUT);
  
    withTimeout(function () { return callToddric(); }, TODDRIC_TIMEOUT, "toddric_timeout")
      .then(function (txt) {
        if (!responded) {
          var compact = String(txt || "").replace(/\s+/g, " ").trim();
          if (compact.length > 300 && !isWhatsApp) {
            sendChunked(compact.slice(0, 300));
            twiml.message("Reply MORE for the next part.");
            responded = true; return send(twiml);
          }
          safeReplyOnce(txt);
        }
      }, function () {
        var elapsed = Date.now() - startMs;
        var remaining = OVERALL_TIMEOUT - elapsed - 200;
        if (!OPENAI_ENABLED || remaining < 1500) {
          if (!responded) { safeReplyOnce("Sorry, that took too long. Try a shorter ask."); }
          return;
        }
        var fbBudget = Math.max(1500, Math.min(4000, remaining));
        withTimeout(function () { return callOpenAI(); }, fbBudget, "fallback_timeout")
          .then(function (txt2) { if (!responded) { safeReplyOnce(txt2); } },
                function () { if (!responded) { safeReplyOnce("Timed out. Try again with fewer details."); } });
      });
  };
  