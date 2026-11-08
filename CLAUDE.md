# gitsum

Rules live in one place: [in-c0/Project-harness](https://github.com/in-c0/Project-harness) —
`POLICY.md` (what a session may do), `INFRA.md` (what is already provisioned), `NEEDS_JI.md`
(the only hand-back queue). Read `POLICY.md` before acting. This file carries no policy of its
own; only the facts below.

## This repo
- `src/core/` is the one engine; every surface (`src/cli.js`, `src/mcp.js`, `extension/`, `api/team/`, `site/`) imports it — never fork its logic. Runtime deps stay tiny (MCP SDK + zod only).
- `skills/gitsum/SKILL.md`, `.claude-plugin/`, `.mcp.json` — Agent Skill / plugin. `extension/`: Chrome MV3 HUD (`npm run build:extension`, `npm run judge:hud`). `api/team/`, `server/`, `supabase/`: paid Team plan on Vercel. `site/`: GitHub Pages.
- `bench/run.mjs` is the outside judge (hit@3 on real repos): run after touching `src/core/find.js` or `analyze.js`; don't merge a drop below 0.9.

Earlier rule files for this repo are archived under `docs/archive/rules/2026-11-08/` and expire
on 2027-02-06 (POLICY.md §8). Open one only if something in this repo stops making sense
without it, and if you did, say which file in your PR description in one line.

Reply to the owner in Korean. Code, commit messages and PR text stay in English.
