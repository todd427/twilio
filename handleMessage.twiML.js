// Twilio Function: /handleMessage
// Purpose: TwiML reply for inbound SMS. Uses OpenAI if key present, else echoes.
// Works in Classic editor (ES5 style, no optional chaining).

exports.handler = function (context, event, callback) {
  // TwiML response (required for per-number webhook path)
  var twiml = new Twilio.twiml.MessagingResponse();

  // Env config
  //var OPENAI_API_KEY = context.OPENAI_API_P1 + context.OPENAI_API_P2;
  var OPENAI_API_KEY = (context.OPENAI_API_P1 || "") + (context.OPENAI_API_P2 || "");

  
  var SYSTEM_PROMPT  = context.SYSTEM_PROMPT || 'You are Toddric. Be concise, helpful, kind, and SMS-friendly.';
  var MODEL          = context.OPENAI_MODEL || 'gpt-4o-mini';
  var REPLY_MAX      = parseInt(context.REPLY_MAX || '1400', 10);
  var TEMP           = Number(context.OPENAI_TEMP || '0.7');
  var MAX_TOKENS     = parseInt(context.OPENAI_MAX_TOKENS || '200', 10);
  var TIMEOUT_MS     = parseInt(context.OPENAI_TIMEOUT_MS || '12000', 10); // 12s

  // Inbound text
  var userMsg = String(event.Body || event.message || '').trim();
  
  // TEMP diag check
    if (String(userMsg).toLowerCase() === 'showkey') {
      twiml.message(
        'OPENAI_API_KEY len=' + OPENAI_API_KEY.length +
        ' tail=' + OPENAI_API_KEY.slice(-6)
      );
      return callback(null, twiml);
    }


  // Quick SMS diagnostics: send "diag"
  if (String(userMsg).toLowerCase() === 'diag') {
    var keyTail = OPENAI_API_KEY ? OPENAI_API_KEY.slice(-6) : '';
    twiml.message(
      'diag: key=' + (OPENAI_API_KEY ? 'yes(*' + keyTail + ')' : 'no') +
      ' node=' + process.version +
      ' model=' + MODEL
    );
    return callback(null, twiml);
  }

  // If no API key, simple echo
  if (!OPENAI_API_KEY) {
    twiml.message(('Echo: ' + userMsg).slice(0, REPLY_MAX));
    return callback(null, twiml);
  }

  // Abort/timeout guard
  var controller = new AbortController();
  var timer = setTimeout(function () { try { controller.abort(); } catch (e) {} }, TIMEOUT_MS);

  // Build request
  var payload = {
    model: MODEL,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userMsg }
    ],
    temperature: TEMP,
    max_tokens: MAX_TOKENS
  };

  fetch('https://api.openai.com/v1/chat/completions', {
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
      // Read response text so we can show the real reason in SMS (401/403/429/etc.)
      return r.text().then(function (txt) {
        throw new Error('openai:' + r.status + ':' + txt.slice(0, 300));
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
    twiml.message(txt.slice(0, REPLY_MAX));
    return callback(null, twiml);
  })
  .catch(function (e) {
    clearTimeout(timer);
    var msg = (e && (e.message || e.toString())) || 'unknown';
    console.error('LLM error:', msg);
    // TEMP: surface the exact reason to your phone for quick fixing.
    // Once everything is stable, replace this with a friendly fallback.
    twiml.message('diag error: ' + msg);
    return callback(null, twiml);
  });
};
