// Twilio Function: /handleMessage
// Purpose: TwiML reply for inbound SMS. Uses Toddric HF API if configured, else OpenAI.
// Works in Classic editor (ES5 style).

exports.handler = function (context, event, callback) {
  var twiml = new Twilio.twiml.MessagingResponse();

  // ---- Config (env) ----
  var TODDRIC_API_URL = String(context.TODDRIC_API_URL || ""); // e.g., https://abc123.ngrok.io
  var TODDRIC_TIMEOUT_MS = parseInt(context.TODDRIC_TIMEOUT_MS || "9000", 10); // keep < Twilio 10s
  var PERSONA_JSON = {};
  try { PERSONA_JSON = context.PERSONA_JSON ? JSON.parse(String(context.PERSONA_JSON)) : {}; } catch (e) { PERSONA_JSON = {}; }

  // OpenAI fallback (unchanged)
  var OPENAI_API_KEY = (context.OPENAI_API_P1 || "") + (context.OPENAI_API_P2 || "");
  var SYSTEM_PROMPT  = context.SYSTEM_PROMPT || 'You are Toddric. Be concise, helpful, kind, and SMS-friendly.';
  var MODEL          = context.OPENAI_MODEL || 'gpt-4o-mini';
  var REPLY_MAX      = parseInt(context.REPLY_MAX || '1400', 10);
  var TEMP           = Number(context.OPENAI_TEMP || '0.7');
  var MAX_TOKENS     = parseInt(context.OPENAI_MAX_TOKENS || '200', 10);
  var OPENAI_TIMEOUT_MS = parseInt(context.OPENAI_TIMEOUT_MS || '12000', 10);

  // ---- Inbound ----
  var userMsg = String(event.Body || event.message || '').trim();
  var from = String(event.From || '').trim();   // e.g., +15551234567 or whatsapp:+1555...
  var sessionId = from || require('crypto').randomBytes(8).toString('hex');

  // ---- Quick diagnostics ----
  if (String(userMsg).toLowerCase() === 'showkey') {
    twiml.message(
      'OPENAI_API_KEY len=' + OPENAI_API_KEY.length +
      ' tail=' + (OPENAI_API_KEY ? OPENAI_API_KEY.slice(-6) : '')
    );
    return callback(null, twiml);
  }

  if (String(userMsg).toLowerCase() === 'diag') {
    var keyTail = OPENAI_API_KEY ? OPENAI_API_KEY.slice(-6) : '';
    twiml.message(
      'diag: toddric=' + (TODDRIC_API_URL ? 'yes(' + TODDRIC_API_URL + ')' : 'no') +
      ' key=' + (OPENAI_API_KEY ? 'yes(*' + keyTail + ')' : 'no') +
      ' node=' + process.version +
      ' model=' + MODEL
    );
    return callback(null, twiml);
  }

  // ---- Helper: bounded reply ----
  function replySMS(txt) {
    twiml.message(String(txt || '').slice(0, REPLY_MAX));
    return callback(null, twiml);
  }

  // ---- Try Toddric HF API first (if configured) ----
  function tryToddric() {
    if (!TODDRIC_API_URL) return Promise.reject(new Error('toddric:not_configured'));

    var url = TODDRIC_API_URL.replace(/\/+$/,'') + '/chat';
    var body = {
      message: userMsg,
      session_id: sessionId
    };
    if (PERSONA_JSON && typeof PERSONA_JSON === 'object' && Object.keys(PERSONA_JSON).length > 0) {
      body.persona = PERSONA_JSON;
    }

    var controller = new AbortController();
    var timer = setTimeout(function () { try { controller.abort(); } catch (e) {} }, TODDRIC_TIMEOUT_MS);

    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal
    })
    .then(function (r) {
      if (!r.ok) {
        return r.text().then(function (txt) {
          throw new Error('toddric:' + r.status + ':' + String(txt).slice(0, 300));
        });
      }
      return r.json();
    })
    .then(function (data) {
      clearTimeout(timer);
      var txt = (data && data.text) ? String(data.text).trim() : '';
      if (!txt) throw new Error('toddric:empty');
      return txt;
    })
    .catch(function (e) {
      clearTimeout(timer);
      throw e;
    });
  }

  // ---- Fallback: OpenAI (your original code) ----
  function tryOpenAI() {
    if (!OPENAI_API_KEY) return Promise.resolve('Echo: ' + userMsg);

    var controller = new AbortController();
    var timer = setTimeout(function () { try { controller.abort(); } catch (e) {} }, OPENAI_TIMEOUT_MS);

    var payload = {
      model: MODEL,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userMsg }
      ],
      temperature: TEMP,
      max_tokens: MAX_TOKENS
    };

    return fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + OPENAI_API_KEY
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    })
    .then(function (r) {
      if (!r.ok) {
        return r.text().then(function (txt) {
          throw new Error('openai:' + r.status + ':' + String(txt).slice(0, 300));
        });
      }
      return r.json();
    })
    .then(function (data) {
      clearTimeout(timer);
      var txt = '...';
      if (data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content) {
        txt = data.choices[0].message.content.trim();
      }
      return txt;
    })
    .catch(function (e) {
      clearTimeout(timer);
      throw e;
    });
  }
  // ---- Simple commands (ES5 style; no async/await) ----

  // Reset server-side session
  if (userMsg.toLowerCase() === 'reset') {
    if (context.TODDRIC_API_URL) {
      fetch(context.TODDRIC_API_URL.replace(/\/+$/,'') + '/reset', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ session_id: sessionId })
      }).catch(function(){ /* ignore */ });
    }
    return replySMS('Okay, cleared our chat.');
  }

  // Ask the server which model is loaded
  if (userMsg.toLowerCase() === 'whoami') {
    if (context.TODDRIC_API_URL) {
      return fetch(context.TODDRIC_API_URL.replace(/\/+$/,'') + '/whoami')
        .then(function(r){ return r.json(); })
        .then(function(j){
          var msg = (j && j.model) ? ('model: ' + j.model) : 'model: unknown';
          return replySMS(msg);
        })
        .catch(function(){ return replySMS('model: unknown'); });
    }
    return replySMS('model: (no server)');
  }

  // ---- Orchestrate: Toddric -> OpenAI -> Echo ----
  tryToddric()
    .then(function (txt) { return replySMS(txt); })
    .catch(function (e1) {
      console.error('Toddric error:', e1 && (e1.message || e1));
      return tryOpenAI()
        .then(function (txt) { return replySMS(txt); })
        .catch(function (e2) {
          console.error('OpenAI error:', e2 && (e2.message || e2));
          return replySMS('diag error: ' + (e2 && (e2.message || e2.toString()) || 'unknown'));
        });
    });
};
