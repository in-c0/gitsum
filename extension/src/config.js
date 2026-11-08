// The one place to point the extension at the hosted Team API (e.g. "https://gitsum.vercel.app").
// Empty = Team plan not deployed yet: the Team key field shows "early access" and stays disabled.
// build.mjs adds this origin to host_permissions automatically when it is set.
export const TEAM_API = "";

export const TEAM_REQUEST_URL = "https://github.com/in-c0/gitsum/issues/new?template=team.yml&title=Team+plan+request";
export const ANTHROPIC_MODEL = "claude-haiku-5-5";
export const GRAIN_CAP = 5000; // beyond this, the strata show the top N files by size (labelled)
export const MAX_LAYERS = 11; // top-level folders shown as their own stratum; the rest merge into "other"
