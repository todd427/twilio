// Local test harness for handleMessage

async function handleMessage(event = {}, context = {}) {
  const userId  = String(event.user_id || event.from || event.From || "unknown");
  const userMsg = String(event.message || event.Body || "").trim();

  const OPENAI_API_KEY = context.OPENAI_API_KEY; // optional
  const SYSTEM_PROMPT =
    context.SYSTEM_PROMPT || "You are Toddric. Be concise, helpful, kind, and SMS-friendly.";
  const REPLY_MAX = parseInt(context.REPLY_MAX || "1400", 10);

  // Diagnostics
  if ((event.diag || "").toString() === "1") {
    return {
      ok: true,
      has_key: Boolean(OPENAI_API_KEY),
      node: process.version,
      got: { userId, userMsg }
    };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 9000);

  async function llmReply(prompt) {
    if (!OPENAI_API_KEY) {
      return `Echo: ${prompt.slice(0, Math.max(0, REPLY_MAX - 6))}`;
    }
    const r = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userMsg }
        ],
        temperature: 0.7,
        max_tokens: 200
      }),
      signal: controller.signal
    });
    if (!r.ok) throw new Error(`OpenAI ${r.status}`);
    const data = await r.json();
    const txt = (data?.choices?.[0]?.message?.content || "…").trim();
    return txt.slice(0, REPLY_MAX);
  }

  try {
    const reply = await llmReply(userMsg);
    clearTimeout(timer);
    return { reply, user_id: userId };
  } catch (err) {
    clearTimeout(timer);
    console.error("LLM error detail:", err);
    return {
      reply: "Toddric hit a hiccup—please try again shortly.",
      user_id: userId
    };
  }
}

// Run directly if called with `node handleMessage.js`
if (require.main === module) {
  const context = { OPENAI_API_KEY: process.env.OPENAI_API_KEY };
  const event = { user_id: "+353872902835", message: "ping" };
  handleMessage(event, context).then(console.log).catch(console.error);
}

module.exports = { handleMessage };

