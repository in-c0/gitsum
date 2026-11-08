// The one place that decides which browser origins may call the Team API.
// Team keys are bearer tokens (no cookies), so CORS here is about letting the extension read
// responses, not about protecting a session. Non-browser callers (CI, MCP, CLI) send no Origin.

import { env } from "./http.js";

const EXTENSION_ORIGIN = /^chrome-extension:\/\/[a-z0-9]{1,64}$/;

function extraOrigins() {
  const list = [env("PUBLIC_SITE_URL"), ...(env("GITSUM_CORS_ORIGINS", "") || "").split(",")];
  return list.map((o) => (o || "").trim().replace(/\/$/, "")).filter(Boolean);
}

export function isAllowedOrigin(origin) {
  if (!origin) return false;
  return EXTENSION_ORIGIN.test(origin) || extraOrigins().includes(origin);
}

/**
 * CORS headers for a response. `anyOrigin` answers `*` (safe for read-only, cookie-less endpoints).
 * Otherwise the request's Origin is echoed only when it is on the allow-list.
 */
export function corsHeaders(req, { anyOrigin = false, methods = "GET, POST, OPTIONS" } = {}) {
  const origin = req.headers.get("origin");
  const h = {
    "Access-Control-Allow-Methods": methods,
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Max-Age": "86400",
  };
  if (anyOrigin) h["Access-Control-Allow-Origin"] = "*";
  else {
    h.Vary = "Origin";
    if (isAllowedOrigin(origin)) h["Access-Control-Allow-Origin"] = origin;
  }
  return h;
}

/** Adds CORS headers to an existing Response and returns it. */
export function withCors(res, req, opts) {
  for (const [k, v] of Object.entries(corsHeaders(req, opts))) res.headers.set(k, v);
  return res;
}

/** Answer for an OPTIONS preflight. */
export function preflight(req, opts) {
  return new Response(null, { status: 204, headers: corsHeaders(req, opts) });
}
