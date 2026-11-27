// POST /api/team/ask — streams an AI answer about a repository for a Gitsum Team key.
// Authorization: Bearer gsk_…  · Body: same as the web app's /api/chat
//   { owner, repo, branch, question, overview: { description, stack, languages, keyFiles, dirs, readme, readmePath },
//     snippets: [{ path, startLine, endLine, snippet }], history: [{ role, text }] }
// SSE: data: {"text":"…"} … data: {"done":true,"used":n,"limit":m} · data: [DONE]   (or event: error)
// 400 bad_request · 401 bad_key · 402 seat_inactive · 429 rate_limited | quota_exceeded · 503 capacity | not_configured
// CORS: chrome-extension://* and the configured site origins (see server/cors.js).

import Anthropic from "@anthropic-ai/sdk";
import { env, fail, missingEnv, readJson } from "../../server/http.js";
import { preflight, withCors } from "../../server/cors.js";
import { teamFromKey, touchKey } from "../../server/teams.js";
import { reserveTeamMessage } from "../../server/limits.js";
import { buildRequest, validateBody } from "../../server/prompt.js";

export const maxDuration = 60;

const CORS = { methods: "POST, OPTIONS" };
const MAX_BODY = 200_000;

let anthropic = null;
export function __setAnthropic(client) {
  anthropic = client;
}
function anthropicClient() {
  if (anthropic) return anthropic;
  const apiKey = env("ANTHROPIC_API_KEY");
  if (!apiKey) return null;
  anthropic = new Anthropic({ apiKey, maxRetries: 2 });
  return anthropic;
}

const encoder = new TextEncoder();
const sse = (obj) => encoder.encode(`data: ${JSON.stringify(obj)}\n\n`);

export function OPTIONS(req) {
  return preflight(req, CORS);
}

export async function POST(req) {
  return withCors(await handle(req), req, CORS);
}

async function handle(req) {
  if (missingEnv(["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "ANTHROPIC_API_KEY"]).length)
    return fail(503, "not_configured", "Gitsum Team AI isn't switched on for this site yet.");

  let found;
  try {
    found = await teamFromKey(req);
  } catch (err) {
    console.error("[team/ask] key lookup failed", err?.message);
    return fail(503, "capacity", "Couldn't check this key. Try again in a moment.");
  }
  if (found.error) return fail(...found.error);
  const { key, team } = found;

  let body;
  try {
    body = await readJson(req, MAX_BODY);
  } catch (err) {
    return fail(err.status || 400, "bad_request", err.message);
  }
  const invalid = validateBody(body);
  if (invalid.length) return fail(400, "bad_request", `Invalid: ${invalid.join(", ")}`);

  const slot = await reserveTeamMessage(team, key.id);
  if (!slot.ok) {
    if (slot.code === "rate_limited")
      return fail(429, "rate_limited", `Too many questions at once from this key. Try again in ${slot.retryAfter}s.`, { retryAfter: slot.retryAfter });
    if (slot.code === "quota_exceeded")
      return fail(429, "quota_exceeded", `Your team has used this month's ${slot.limit} AI answers. The quota resets on the 1st; add seats to raise it.`, { limit: slot.limit });
    return fail(503, "capacity", "Gitsum is at capacity right now. Try again shortly.");
  }
  touchKey(key.id);

  const client = anthropicClient();
  const request = buildRequest(body);
  const model = env("GITSUM_MODEL", "claude-haiku-5-5");
  const maxTokens = Number(env("GITSUM_MAX_TOKENS", 1200));

  const stream = new ReadableStream({
    async start(controller) {
      let produced = false;
      try {
        const run = client.messages.stream({ model, max_tokens: maxTokens, system: request.system, messages: request.messages });
        req.signal?.addEventListener("abort", () => run.abort?.());
        for await (const event of run) {
          if (event.type === "content_block_delta" && event.delta?.type === "text_delta" && event.delta.text) {
            produced = true;
            controller.enqueue(sse({ text: event.delta.text }));
          }
        }
        const final = await run.finalMessage?.();
        if (final?.usage) console.log("[team/ask] usage", JSON.stringify({ team: team.id, model, ...final.usage }));
        if (!produced) await slot.release().catch(() => {});
        controller.enqueue(sse({ done: true, used: produced ? slot.used : slot.used - 1, limit: slot.limit }));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      } catch (err) {
        console.error("[team/ask] model call failed", err?.status, err?.message);
        if (!produced) await slot.release().catch(() => {});
        const message = err?.status === 529 || err?.status === 429 ? "The AI is busy right now. Try again in a moment." : "The AI couldn't answer this time. Try again.";
        controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify(message)}\n\n`));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Accel-Buffering": "no",
    },
  });
}
