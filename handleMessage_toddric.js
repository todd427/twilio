// Twilio Function: /handleMessage (TwiML)
// Production: STOP/START/HELP, Toddric primary, OpenAI fallback, friendly errors.

exports.handler = function (context, event, callback) {
  var twiml = new Twilio.twiml.MessagingResponse();

  // ---- Config (env) ----
  var TODDRIC_URL  = (context.TODDRIC_URL || "").trim();      // e.g., https://<tunnel>.trycloudflare.com/v1/reply
  var TODDRIC_KEY  = (context.TODDRIC_KEY || "").trim();      // optional bearer token

  // OpenAI fallback (optional; joined from two env vars)
  var OPENAI_API_KEY = ((context.OPENAI_P1 || "") + (context.OPENAI_P2 || "")).trim();
  var SYSTEM_PROMPT  = context.SYSTEM_PROMPT || "You are Toddric. Be concise, helpful, kind, and SMS-friendly.";
  var MODEL          = context.OPENAI_MODEL || "gpt-4o-mini";
  var TEMP           = Number(context.OPENAI_TEMP || "0.7");
  var MAX_TOKENS     = parseInt(context.OPENAI_MAX_TOKENS || "200", 10);
  var TIMEOUT_MS     = parseInt(context.OPENAI_TIMEOUT_MS || "12000", 10); // keep SMS latency < ~12s
  var REPLY_MAX      = parseInt(context.REPLY_MAX || "480", 10);           // safe SMS cap

  // ---- Inbound ----
  var from     = String(event.From || "");
  var userMsg  = String(event.Body || event.message || "").trim();
  var lowerMsg = userMsg.toLowerCase();

  // ---- Compliance keywords ----
  var STOP_WORDS  = ["stop", "cancel", "end", "quit", "unsubscribe"];
  var START_WORDS = ["start", "unstop"];
  var HELP_WORDS  = ["help", "info"];

  function isOneOf(arr, s) { for (var i=0;i<arr.length;i++){ if (s === arr[i]) return true; } return false; }

  if (!userMsg) {
    twiml.message("Hi! Send me a message and I’ll reply. Text HELP for info or STOP to opt out.");
    return callback(null, twiml);
  }
  if (isOneOf(STOP_WORDS, lowerMsg)) {
    twiml.message("You’re opted out of Toddric SMS. Reply START to rejoin.");
    return callback(null, twiml);
  }
  if (isOneOf(START_WORDS, lowerMsg)) {
    twiml.message("You’re opted back in to Toddric SMS. How can I help?");
    return callback(null, twiml);
  }
  if (isOneOf(HELP_WORDS, lowerMsg)) {
    twiml.message("Help: Toddric SMS. Quick answers. Std msg&data rates may apply. Reply STOP to opt out.");
    return callback(null, twiml);
  }

  // ---- Quick diag command (optional) ----
  if (lowerMsg === "diag") {
    var haveToddric = TODDRIC_URL ? "yes" : "no";
    var haveOA = OPENAI_API_KEY ? "yes" : "no";
    twiml.message("diag: toddric=" + haveToddric + " openai=" + haveOA + " node=" + process.version);
    return callback(null, twiml);
  }

  // ---- Utility: guarded fetch with timeout ----
  function guardedFetch(url, opts, timeoutMs) {
    var controller = new AbortController();
    var timer = setTimeout(function(){ try { controller.abort(); } catch(e){} }, timeoutMs);
    opts = opts || {};
    opts.signal = controller.signal;
    return fetch(url, opts).then(function(r){
      clearTimeout(timer);
      return r;
    }).catch(function(e){
      clearTimeout(timer);
      throw e;
    });
  }

  // ---- Step 1: Try Toddric API ----
  function callToddric(msg) {
    if (!TODDRIC_URL) return Promise.reject(new Error("toddric:missing-url"));
    var headers = { "Content-Type": "application/json" };
    if (TODDRIC_KEY) headers.Authorization = "Bearer " + TODDRIC_KEY;

    var body = {
      user_id: from,
      message: msg,
      channel: "sms"
    };

    return guardedFetch(TODDRIC_URL, {
      method: "POST",
      headers: headers,
      body: JSON.stringify(body)
    }, TIMEOUT_MS).then(function(r){
      if (!r.ok) {
        return r.text().then(function(t){
          throw new Error("toddric:" + r.status + ":" + t.slice(0,300));
        });
      }
      return r.json();
    }).then(function(data){
      var txt = (data && data.reply ? String(data.reply) : "...").trim();
      return txt.slice(0, REPLY_MAX);
    });
  }

  // ---- Step 2: Fallback to OpenAI (optional) ----
  function callOpenAI(msg) {
    if (!OPENAI_API_KEY) return Promise.reject(new Error("openai:missing-key"));
    var payload = {
      model: MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user",   content: msg }
      ],
      temperature: TEMP,
      max_tokens: MAX_TOKENS
    };
    return guardedFetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + OPENAI_API_KEY
      },
      body: JSON.stringify(payload)
    }, TIMEOUT_MS).then(function(r){
      if (!r.ok) {
        return r.text().then(function(t){
          throw new Error("openai:" + r.status + ":" + t.slice(0,300));
        });
      }
      return r.json();
    }).then(function(data){
      var txt = "...";
      if (data && data.choices && data.choices[0] &&
          data.choices[0].message && data.choices[0].message.content) {
        txt = data.choices[0].message.content.trim();
      }
      return txt.slice(0, REPLY_MAX);
    });
  }

  // ---- Step 3: As a last resort, echo ----
  function echoReply(msg) {
    return Promise.resolve(("Echo: " + msg).slice(0, REPLY_MAX));
  }

  // ---- Execute: Toddric → OpenAI → Echo ----
  callToddric(userMsg)
    .catch(function(e1){
      console.error("Toddric error:", (e1 && (e1.stack || e1.message)) || e1);
      return callOpenAI(userMsg);
    })
    .catch(function(e2){
      console.error("OpenAI fallback error:", (e2 && (e2.stack || e2.message)) || e2);
      return echoReply(userMsg);
    })
    .then(function(replyText){
      twiml.message(replyText);
      return callback(null, twiml);
    })
    .catch(function(finalErr){
      console.error("Final error:", (finalErr && (finalErr.stack || finalErr.message)) || finalErr);
      twiml.message("Sorry—Toddric had a hiccup. Please try again in a moment.");
      return callback(null, twiml);
    });
};
