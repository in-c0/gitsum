// Gitsum Team client: hosted AI answers for paid team seats (key: gsk_…).
// Individuals never need this — every other feature is free and runs locally.

export const TEAM_API_DEFAULT = ""; // set once the hosted Team API is deployed

export function teamConfig(env = globalThis.process?.env || {}) {
  const key = env.GITSUM_TEAM_KEY || "";
  const base = (env.GITSUM_API || TEAM_API_DEFAULT).replace(/\/$/, "");
  return { key, base, enabled: Boolean(key && base) };
}

/** Builds the request body the Team API expects from an opened session + evidence citations. */
export function teamBody(session, question, citations = [], history = []) {
  const { repoData: r, overview: o } = session;
  return {
    owner: r.owner,
    repo: r.repo,
    branch: r.branch,
    question,
    overview: {
      description: o.oneLiner,
      stack: o.stack,
      languages: o.languages,
      keyFiles: o.keyFiles,
      dirs: o.structure.dirs,
      readme: o.readme ? o.readme.slice(0, 9000) : "",
      readmePath: o.readmePath,
    },
    snippets: citations.filter((c) => c.snippet).map((c) => ({ path: c.path, startLine: c.startLine, endLine: c.endLine, snippet: c.snippet })),
    history,
  };
}

/** Calls POST /api/team/ask and collects the streamed answer. Returns { text, used, limit } or throws { code, message }. */
export async function teamAsk(cfg, body, { fetchImpl = globalThis.fetch, onText } = {}) {
  const res = await fetchImpl(`${cfg.base}/api/team/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.key}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let err = { code: "team_error", message: `Team API returned ${res.status}` };
    try {
      err = (await res.json()).error || err;
    } catch {
      /* not JSON */
    }
    throw Object.assign(new Error(err.message), err);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let text = "";
  let meta = {};
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      if (/^event: error/m.test(frame)) {
        const data = frame.match(/^data: (.*)$/m)?.[1];
        throw Object.assign(new Error(data ? JSON.parse(data) : "Team answer failed"), { code: "model_error" });
      }
      const data = frame.match(/^data: (.*)$/m)?.[1];
      if (!data || data === "[DONE]") continue;
      const obj = JSON.parse(data);
      if (obj.text) {
        text += obj.text;
        onText?.(obj.text);
      }
      if (obj.done) meta = obj;
    }
  }
  return { text, used: meta.used, limit: meta.limit };
}
