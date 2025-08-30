// Twilio Function: /handleMessage (TwiML)
// Production-ready: STOP/HELP/START handling, OpenAI call, friendly fallback.
// Uses split env vars: OPENAI_P1 + OPENAI_P2

exports.handler = function (context, event, callback) {
    var twiml = new Twilio.twiml.MessagingResponse();
  
    // --- Config (env) ---
    var OPENAI_API_KEY = ((context.OPENAI_API_P1 || "") + (context.OPENAI_API_P2 || "")).trim();
    var SYSTEM_PROMPT  = context.SYSTEM_PROMPT || "You are Toddric. Be concise, helpful, kind, and SMS-friendly.";
    var MODEL          = context.OPENAI_MODEL || "gpt-4o-mini";
    var TEMP           = Number(context.OPENAI_TEMP || "0.7");
    var MAX_TOKENS     = parseInt(context.OPENAI_MAX_TOKENS || "200", 10);
    var TIMEOUT_MS     = parseInt(context.OPENAI_TIMEOUT_MS || "12000", 10); // 12s is SMS-friendly
    var REPLY_MAX      = parseInt(context.REPLY_MAX || "480", 10); // cap length for SMS
  
    // --- Incoming text ---
    var from     = String(event.From || "");
    var userMsg  = String(event.Body || event.message || "").trim();
    var lowerMsg = userMsg.toLowerCase();
  
    // --- Compliance keywords (simple server-side handling) ---
    var STOP_WORDS  = ["stop", "cancel", "end", "quit", "unsubscribe"];
    var START_WORDS = ["start", "unstop"];
    var HELP_WORDS  = ["help", "info"];
    function isOneOf(arr, s) { for (var i=0;i<arr.length;i++){ if (s === arr[i]) return true; } return false; }
  
    if (!userMsg) {
      twiml.message("Hi! Send me a message and I’ll reply. Text HELP for info or STOP to opt out.");
      return callback(null, twiml);
    }
  
    if (isOneOf(STOP_WORDS, lowerMsg)) {
      // Note: Full carrier-grade opt-out is best handled by Messaging Services Advanced Opt-Out.
      twiml.message("You’re opted out of Toddric SMS. Reply START to rejoin.");
      return callback(null, twiml);
    }
  
    if (isOneOf(START_WORDS, lowerMsg)) {
      twiml.message("You’re opted back in to Toddric SMS. How can I help?");
      return callback(null, twiml);
    }
  
    if (isOneOf(HELP_WORDS, lowerMsg)) {
      twiml.message("Help: Toddric SMS. We’ll answer quick questions. Std msg&data rates may apply. Reply STOP to opt out.");
      return callback(null, twiml);
    }
  
    // --- If no API key, stay in echo mode (safe fallback) ---
    if (!OPENAI_API_KEY) {
      twiml.message(("Echo: " + userMsg).slice(0, REPLY_MAX));
      return callback(null, twiml);
    }
  
    // --- OpenAI call with timeout guard ---
    var controller = new AbortController();
    var timer = setTimeout(function(){ try { controller.abort(); } catch(e){} }, TIMEOUT_MS);
  
    var payload = {
      model: MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user",   content: userMsg }
      ],
      temperature: TEMP,
      max_tokens: MAX_TOKENS
    };
  
    fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + OPENAI_API_KEY
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    })
    .then(function(r){
      if (!r.ok) {
        return r.text().then(function(txt){
          throw new Error("openai:"+r.status+":"+txt.slice(0,300));
        });
      }
      return r.json();
    })
    .then(function(data){
      clearTimeout(timer);
      var txt = "...";
      if (data && data.choices && data.choices[0] &&
          data.choices[0].message && data.choices[0].message.content) {
        txt = data.choices[0].message.content.trim();
      }
      twiml.message(txt.slice(0, REPLY_MAX));
      return callback(null, twiml);
    })
    .catch(function(e){
      clearTimeout(timer);
      console.error("LLM error:", (e && (e.stack || e.message)) || e);
      // Friendly fallback (no internal details)
      twiml.message("Sorry—Toddric had a hiccup. Please try again in a moment.");
      return callback(null, twiml);
    });
  };
  