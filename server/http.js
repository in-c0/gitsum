// Small helpers shared by the /api functions (Vercel Node runtime, Web Request/Response signature).

export function json(status, body, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...headers },
  });
}

/** A JSON error the frontend can switch on: { error: { code, message } }. */
export function fail(status, code, message, extra = {}) {
  return json(status, { error: { code, message, ...extra } });
}

export function env(name, fallback = undefined) {
  const v = process.env[name];
  return v === undefined || v === "" ? fallback : v;
}

export function missingEnv(names) {
  return names.filter((n) => !env(n));
}

export function bearer(req) {
  const h = req.headers.get("authorization") || "";
  const m = h.match(/^Bearer\s+(.+)$/i);
  return m ? m[1].trim() : null;
}

export function originOf(req) {
  const configured = env("PUBLIC_SITE_URL");
  if (configured) return configured.replace(/\/$/, "");
  const url = new URL(req.url);
  return `${url.protocol}//${url.host}`;
}

export async function readJson(req, maxBytes = 200_000) {
  const text = await req.text();
  if (text.length > maxBytes) throw Object.assign(new Error("Request too large"), { status: 413 });
  try {
    return JSON.parse(text || "{}");
  } catch {
    throw Object.assign(new Error("Invalid JSON"), { status: 400 });
  }
}
