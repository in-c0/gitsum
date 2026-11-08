// GET /api/team/me — what a team key can do: { team: { name, seats, status }, usage: { used, limit } }.
// Authorization: Bearer gsk_…  · 401 bad_key · 402 seat_inactive · 503 not_configured
// CORS: any origin (read-only, bearer token, no cookies).

import { fail, json, missingEnv } from "../../server/http.js";
import { preflight, withCors } from "../../server/cors.js";
import { teamFromKey } from "../../server/teams.js";
import { teamUsage } from "../../server/limits.js";

const CORS = { anyOrigin: true, methods: "GET, OPTIONS" };

export function OPTIONS(req) {
  return preflight(req, CORS);
}

export async function GET(req) {
  return withCors(await handle(req), req, CORS);
}

async function handle(req) {
  if (missingEnv(["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]).length)
    return fail(503, "not_configured", "Gitsum Team isn't switched on for this site yet.");
  try {
    const found = await teamFromKey(req);
    if (found.error) return fail(...found.error);
    const { team } = found;
    return json(200, { team: { name: team.name, seats: team.seats, status: team.status }, usage: await teamUsage(team) });
  } catch (err) {
    console.error("[team/me]", err?.message);
    return fail(503, "capacity", "Couldn't check this key. Try again in a moment.");
  }
}
