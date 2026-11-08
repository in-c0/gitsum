// Team usage limits that hold across every serverless instance (Upstash Redis).
// - burst: per-key sliding window, stops runaway CI jobs and scripts
// - monthly: per-team quota = seats × GITSUM_TEAM_MONTHLY_PER_SEAT
// - global daily cap: a hard ceiling on total AI calls per day. It shares its Redis key with the
//   Gitsum web app, so one Anthropic bill has one cost guard across both apps.
// Without Upstash configured, limits fall back to per-instance memory ONLY for local dev.
// Deployed instances fail closed rather than silently losing the global bill cap.

import { Redis } from "@upstash/redis";
import { Ratelimit } from "@upstash/ratelimit";
import { env } from "./http.js";

let redis;
let burst;
let burstPerMinute;
const memory = new Map();

function client() {
  if (redis !== undefined) return redis;
  const url = env("UPSTASH_REDIS_REST_URL") || env("KV_REST_API_URL");
  const token = env("UPSTASH_REDIS_REST_TOKEN") || env("KV_REST_API_TOKEN");
  redis = url && token ? new Redis({ url, token }) : null;
  return redis;
}

/** Test hook. */
export function __setRedis(r) {
  redis = r;
  burst = undefined;
  memory.clear();
}

export function limitsConfig(seats = 0) {
  const perSeat = Number(env("GITSUM_TEAM_MONTHLY_PER_SEAT", 1500));
  return {
    perMinute: Number(env("GITSUM_RATE_PER_MINUTE", 12)),
    monthly: Math.max(0, Number(seats) || 0) * perSeat,
    globalDaily: Number(env("GITSUM_GLOBAL_DAILY_LIMIT", 20000)),
  };
}

const monthKey = (d = new Date()) => `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
const dayKey = (d = new Date()) => d.toISOString().slice(0, 10);
const teamUsageKey = (teamId) => `gitsum:team-usage:${teamId}:${monthKey()}`;

async function incr(key, ttlSeconds) {
  const r = client();
  if (r) {
    const n = await r.incr(key);
    if (n === 1) await r.expire(key, ttlSeconds);
    return n;
  }
  const n = (memory.get(key) || 0) + 1;
  memory.set(key, n);
  return n;
}

async function get(key) {
  const r = client();
  if (r) return Number((await r.get(key)) || 0);
  return memory.get(key) || 0;
}

async function decr(key) {
  const r = client();
  if (r) return r.decr(key);
  memory.set(key, Math.max(0, (memory.get(key) || 1) - 1));
  return memory.get(key);
}

export async function teamUsage(team) {
  const { monthly } = limitsConfig(team.seats);
  return { used: await get(teamUsageKey(team.id)), limit: monthly };
}

/**
 * Reserves one AI answer for this team, charged to `keyId` for burst limiting.
 * Returns { ok: true, used, limit, release } or { ok: false, code, retryAfter?, limit? }.
 * Call release() if the model call fails before producing an answer, so the team isn't charged.
 */
export async function reserveTeamMessage(team, keyId) {
  const cfg = limitsConfig(team.seats);
  const r = client();

  // A per-instance Map is not a rate limit for serverless production: each cold start
  // gets a new counter. Refuse billable requests until shared Redis is configured.
  if (!r && (env("VERCEL") === "1" || env("NODE_ENV") === "production")) {
    return { ok: false, code: "capacity" };
  }

  if (r) {
    if (!burst || burstPerMinute !== cfg.perMinute) {
      burst = new Ratelimit({ redis: r, limiter: Ratelimit.slidingWindow(cfg.perMinute, "60 s"), prefix: "gitsum:team-burst" });
      burstPerMinute = cfg.perMinute;
    }
    const b = await burst.limit(keyId);
    if (!b.success) return { ok: false, code: "rate_limited", retryAfter: Math.max(1, Math.ceil((b.reset - Date.now()) / 1000)) };
  } else {
    const k = `team-burst:${keyId}:${Math.floor(Date.now() / 60000)}`;
    if ((await incr(k, 120)) > cfg.perMinute) return { ok: false, code: "rate_limited", retryAfter: 60 };
  }

  const gKey = `gitsum:global:${dayKey()}`;
  const g = await incr(gKey, 2 * 86400);
  if (g > cfg.globalDaily) {
    await decr(gKey);
    return { ok: false, code: "capacity" };
  }

  const uKey = teamUsageKey(team.id);
  const u = await incr(uKey, 40 * 86400);
  if (u > cfg.monthly) {
    await decr(uKey);
    await decr(gKey);
    return { ok: false, code: "quota_exceeded", limit: cfg.monthly };
  }

  return {
    ok: true,
    used: u,
    limit: cfg.monthly,
    release: async () => {
      await decr(uKey);
      await decr(gKey);
    },
  };
}
