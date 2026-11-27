// /api/team/keys — team owners manage team keys (gsk_…) for CI agents, the MCP `ask` tool and the extension.
// POST   { label?, team_id? }  → { key, id, label, last4 }  (plaintext shown ONCE; only a SHA-256 hash is stored)
// GET    ?team_id=            → { team, keys: [{ id, label, last4, created_at, last_used_at, revoked_at }] }
// DELETE ?id=                 → { revoked: true }
// Max active keys per team = seats × 2.

import { fail, json, missingEnv, readJson } from "../../server/http.js";
import { currentUser, isActive } from "../../server/auth.js";
import { KEYS_PER_SEAT, countActiveKeys, keyById, listKeys, mintKey, ownedTeam, revokeKey, teamsOwnedBy } from "../../server/teams.js";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

async function owner(req) {
  if (missingEnv(["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]).length)
    return { res: fail(503, "not_configured", "Gitsum Team isn't switched on for this site yet.") };
  const user = await currentUser(req);
  if (!user) return { res: fail(401, "login_required", "Sign in as the team owner.") };
  return { user };
}

const teamSummary = (t) => ({ id: t.id, name: t.name, seats: t.seats, status: t.status, active: isActive(t), maxKeys: t.seats * KEYS_PER_SEAT });

export async function POST(req) {
  const { user, res } = await owner(req);
  if (res) return res;

  let body;
  try {
    body = await readJson(req, 2000);
  } catch (err) {
    return fail(err.status || 400, "bad_request", err.message);
  }
  const label = typeof body.label === "string" && body.label.trim() ? body.label.trim().slice(0, 60) : "Team key";
  if (body.team_id !== undefined && !UUID.test(String(body.team_id))) return fail(400, "bad_request", "Invalid: team_id");

  try {
    const picked = await ownedTeam(user, body.team_id);
    if (picked.error) return fail(...picked.error);
    const { team } = picked;
    if (!isActive(team)) return fail(402, "seat_inactive", "This team's subscription isn't active, so new keys can't be made.");
    const max = team.seats * KEYS_PER_SEAT;
    if ((await countActiveKeys(team.id)) >= max)
      return fail(409, "key_limit", `This team can have ${max} active keys (2 per seat). Revoke one first.`, { limit: max });
    const minted = await mintKey({ teamId: team.id, userId: user.id, label });
    return json(201, { key: minted.key, id: minted.id, label: minted.label, last4: minted.last4, created_at: minted.created_at, team: teamSummary(team) });
  } catch (err) {
    console.error("[team/keys] mint failed", err?.message);
    return fail(503, "capacity", "Couldn't create the key. Try again in a moment.");
  }
}

export async function GET(req) {
  const { user, res } = await owner(req);
  if (res) return res;
  const teamId = new URL(req.url).searchParams.get("team_id") || undefined;
  if (teamId && !UUID.test(teamId)) return fail(400, "bad_request", "Invalid: team_id");
  try {
    if (!teamId) {
      const teams = await teamsOwnedBy(user.id);
      if (teams.length > 1) return json(200, { teams: teams.map(teamSummary), keys: [] });
    }
    const picked = await ownedTeam(user, teamId);
    if (picked.error) return fail(...picked.error);
    const keys = await listKeys(picked.team.id);
    return json(200, { team: teamSummary(picked.team), keys });
  } catch (err) {
    console.error("[team/keys] list failed", err?.message);
    return fail(503, "capacity", "Couldn't load keys. Try again in a moment.");
  }
}

export async function DELETE(req) {
  const { user, res } = await owner(req);
  if (res) return res;
  const id = new URL(req.url).searchParams.get("id") || "";
  if (!UUID.test(id)) return fail(400, "bad_request", "Invalid: id");
  try {
    const key = await keyById(id);
    if (!key) return fail(404, "not_found", "No such key.");
    const owned = await teamsOwnedBy(user.id);
    if (!owned.some((t) => t.id === key.team_id)) return fail(403, "not_owner", "Only the team owner can revoke keys.");
    if (!key.revoked_at) await revokeKey(id);
    return json(200, { revoked: true, id });
  } catch (err) {
    console.error("[team/keys] revoke failed", err?.message);
    return fail(503, "capacity", "Couldn't revoke the key. Try again in a moment.");
  }
}
