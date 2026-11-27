// Team data: teams, owners and team keys (public.gitsum_teams, gitsum_team_members, gitsum_team_keys).
// All access goes through the service role. Team keys are stored ONLY as a SHA-256 hash + last 4 chars;
// the plaintext is returned once at mint time and never logged.

import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { bearer } from "./http.js";
import { isActive, supabaseAdmin } from "./auth.js";

export const MIN_SEATS = 2;
export const MAX_SEATS = 500;
export const KEYS_PER_SEAT = 2;

const TEAM_COLS = "id, name, owner_user_id, stripe_customer_id, stripe_subscription_id, status, seats, current_period_end, cancel_at_period_end";
const KEY_FORMAT = /^gsk_[A-Za-z0-9]{32}$/;
const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

function db() {
  const sb = supabaseAdmin();
  if (!sb) throw new Error("Supabase is not configured");
  return sb;
}

/** `gsk_` + 32 uniformly random base62 characters (rejection sampling, no modulo bias). */
export function generateKey() {
  let out = "";
  while (out.length < 32) {
    for (const b of randomBytes(48)) {
      if (b < 248 && out.length < 32) out += BASE62[b % 62]; // 248 = 4 × 62
    }
  }
  return `gsk_${out}`;
}

export const hashKey = (key) => createHash("sha256").update(key, "utf8").digest("hex");

function sameHash(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  return timingSafeEqual(Buffer.from(a, "utf8"), Buffer.from(b, "utf8"));
}

export const looksLikeTeamKey = (s) => typeof s === "string" && KEY_FORMAT.test(s);

// ---------- teams ----------

export async function createTeam({ name, ownerId, seats }) {
  const sb = db();
  const { data, error } = await sb
    .from("gitsum_teams")
    .insert({ name, owner_user_id: ownerId, seats, status: "incomplete" })
    .select("id")
    .single();
  if (error) throw error;
  const { error: mErr } = await sb.from("gitsum_team_members").insert({ team_id: data.id, user_id: ownerId, role: "owner" });
  if (mErr) throw mErr;
  return data.id;
}

export async function deleteTeam(teamId) {
  await db().from("gitsum_teams").delete().eq("id", teamId);
}

export async function teamById(teamId) {
  const { data, error } = await db().from("gitsum_teams").select(TEAM_COLS).eq("id", teamId).maybeSingle();
  if (error) throw error;
  return data || null;
}

export async function teamBySubscription(subscriptionId) {
  const { data, error } = await db().from("gitsum_teams").select(TEAM_COLS).eq("stripe_subscription_id", subscriptionId).maybeSingle();
  if (error) throw error;
  return data || null;
}

export async function teamsOwnedBy(userId) {
  const { data, error } = await db().from("gitsum_teams").select(TEAM_COLS).eq("owner_user_id", userId).order("created_at", { ascending: false });
  if (error) throw error;
  return data || [];
}

/** Writes Stripe billing state onto a team. Idempotent; returns the team id or null when no row matched. */
export async function saveTeamBilling(teamId, fields) {
  const { data, error } = await db().from("gitsum_teams").update(fields).eq("id", teamId).select("id");
  if (error) throw error;
  return data?.[0]?.id || null;
}

/**
 * Picks the team an owner is acting on. `teamId` is optional when the user owns exactly one team.
 * Returns { team } or { error: [status, code, message] }.
 */
export async function ownedTeam(user, teamId) {
  const teams = await teamsOwnedBy(user.id);
  if (teamId) {
    const team = teams.find((t) => t.id === teamId);
    return team ? { team } : { error: [403, "not_owner", "Only the team owner can manage keys."] };
  }
  if (teams.length === 1) return { team: teams[0] };
  if (!teams.length) return { error: [404, "no_team", "You don't own a Gitsum team yet."] };
  return { error: [400, "team_required", "You own several teams; pass team_id."] };
}

// ---------- keys ----------

export async function countActiveKeys(teamId) {
  const { count, error } = await db().from("gitsum_team_keys").select("id", { count: "exact", head: true }).eq("team_id", teamId).is("revoked_at", null);
  if (error) throw error;
  return count || 0;
}

export async function mintKey({ teamId, userId, label }) {
  const key = generateKey();
  const row = { team_id: teamId, key_hash: hashKey(key), last4: key.slice(-4), label, created_by: userId };
  const { data, error } = await db().from("gitsum_team_keys").insert(row).select("id, label, last4, created_at").single();
  if (error) throw error;
  return { key, ...data };
}

export async function listKeys(teamId) {
  const { data, error } = await db()
    .from("gitsum_team_keys")
    .select("id, label, last4, created_by, created_at, last_used_at, revoked_at")
    .eq("team_id", teamId)
    .order("created_at", { ascending: false });
  if (error) throw error;
  return data || [];
}

export async function keyById(id) {
  const { data, error } = await db().from("gitsum_team_keys").select("id, team_id, revoked_at").eq("id", id).maybeSingle();
  if (error) throw error;
  return data || null;
}

export async function revokeKey(id) {
  const { error } = await db().from("gitsum_team_keys").update({ revoked_at: new Date().toISOString() }).eq("id", id).is("revoked_at", null);
  if (error) throw error;
}

export function touchKey(id) {
  // Best effort; never blocks or fails a request.
  Promise.resolve()
    .then(() => db().from("gitsum_team_keys").update({ last_used_at: new Date().toISOString() }).eq("id", id))
    .catch(() => {});
}

/**
 * Resolves `Authorization: Bearer gsk_…` to { key, team } or { error: [status, code, message] }.
 * 401 bad_key for a missing/unknown/revoked key, 402 seat_inactive when the subscription lapsed.
 */
export async function teamFromKey(req) {
  const token = bearer(req);
  if (!looksLikeTeamKey(token)) return { error: [401, "bad_key", "Send a Gitsum team key as `Authorization: Bearer gsk_…`."] };
  const hash = hashKey(token);
  const { data: key, error } = await db().from("gitsum_team_keys").select("id, team_id, key_hash, revoked_at").eq("key_hash", hash).maybeSingle();
  if (error) throw error;
  if (!key || key.revoked_at || !sameHash(key.key_hash, hash)) return { error: [401, "bad_key", "This team key is unknown or was revoked."] };
  const team = await teamById(key.team_id);
  if (!team) return { error: [401, "bad_key", "This team key is unknown or was revoked."] };
  if (!isActive(team)) return { error: [402, "seat_inactive", "This team's Gitsum subscription isn't active. Ask the team owner to renew."], team };
  return { key, team };
}
