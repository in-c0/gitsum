// Who is calling? Identity comes from the Supabase access token the browser sends as a Bearer token.
// Individuals are free everywhere, so there is no per-user plan here: paid state lives on the team
// (public.gitsum_teams, written only by /api/team/stripe-webhook with the service role).

import { createClient } from "@supabase/supabase-js";
import { bearer, env } from "./http.js";

let admin = null;

export function supabaseAdmin() {
  if (admin) return admin;
  const url = env("SUPABASE_URL");
  const key = env("SUPABASE_SERVICE_ROLE_KEY");
  if (!url || !key) return null;
  admin = createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } });
  return admin;
}

/** Test hook: swap the Supabase client. */
export function __setSupabaseAdmin(client) {
  admin = client;
}

/** Returns the signed-in user ({ id, email }) or null. Never throws for a bad token. */
export async function currentUser(req) {
  const token = bearer(req);
  // Team keys are not Supabase sessions; don't send them to the auth server.
  if (!token || token.startsWith("gsk_")) return null;
  const sb = supabaseAdmin();
  if (!sb) return null;
  try {
    const { data, error } = await sb.auth.getUser(token);
    if (error || !data?.user) return null;
    return { id: data.user.id, email: data.user.email || null };
  } catch {
    return null;
  }
}

const ACTIVE = new Set(["active", "trialing"]);

/** True when a subscription-backed row (team) is paid up. */
export function isActive(row, now = Date.now()) {
  if (!row || !ACTIVE.has(row.status)) return false;
  if (!row.current_period_end) return true;
  // One day of grace so a renewal webhook arriving late doesn't lock people out.
  return new Date(row.current_period_end).getTime() + 86_400_000 > now;
}
