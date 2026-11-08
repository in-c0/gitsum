## Merging pull requests

The repository owner (@in-c0) has pre-approved merges: when Claude opens a pull request in this repository, merge it as soon as its CI checks pass. Do not wait for a review and do not ask the owner for permission.

- Merge with `gh api -X PUT repos/in-c0/<repo>/pulls/<number>/merge -f merge_method=squash`. In cloud sessions `gh pr merge` fails because the GitHub proxy blocks GraphQL; locally `gh pr merge <number> --squash` also works. Both forms are allow-listed in `.claude/settings.json`.
- If the merge is still blocked (a permission check, branch protection or failing CI), tell the owner once, in one line, with the reason and the PR link. Do not repeat that message on later wake-ups.
- Reply to the owner in Korean. Code, commit messages and PR text can stay in English.

## Layout

- `src/core/` — the one engine (GitHub access, overview, search, definition hunt, formatting). Every surface imports it; never fork its logic.
- `src/cli.js`, `src/mcp.js` — CLI and MCP server (`npx -y github:in-c0/gitsum#main …`). Runtime deps stay tiny: only the MCP SDK and zod are `dependencies`; server/build deps are `devDependencies` so the npx install stays fast.
- `skills/gitsum/SKILL.md`, `.claude-plugin/`, `.mcp.json` — Agent Skill and Claude Code plugin.
- `extension/` — Chrome MV3 side-panel HUD (`npm run build:extension`, judged by `npm run judge:hud`).
- `api/team/`, `server/`, `supabase/` — paid Team plan (Vercel functions).
- `site/` — landing + pricing (GitHub Pages).
- `bench/run.mjs` — outside judge for answer quality (hit@3 on real repos). Run it after any change to `src/core/find.js` or `analyze.js`; don't merge a drop below 0.9.
