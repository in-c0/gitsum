---
name: gitsum
description: Understand an unfamiliar GitHub repository fast — what it is, how to run it, where things are defined — with exact path:line citations, without cloning or crawling the tree. Use whenever a task involves a GitHub repo you haven't read yet (evaluating a library, onboarding, answering "where is X", "how does Y work", "how do I run this").
version: 0.1.0
---

# Gitsum

Gitsum turns any GitHub repo into a compact briefing and answers questions with evidence (`path:line` + snippet). It is free for individuals and needs no account. It runs with Node 18+ via `npx`, nothing to install.

## Commands

Run these in a terminal. `<repo>` is `owner/repo` or any `github.com` URL.

```sh
npx -y github:in-c0/gitsum overview <repo>                  # start here: purpose, stack, health, run command, folders, files to read first
npx -y github:in-c0/gitsum ask <repo> "<question>"          # evidence for a question, with path:line citations
npx -y github:in-c0/gitsum find <repo> <Symbol>             # where a function/class/type is defined (searches the whole tree)
npx -y github:in-c0/gitsum read <repo> <path> [start] [end] # file with line numbers (220 lines per call)
npx -y github:in-c0/gitsum tree <repo> [folder] --match <s> # list files
```

Add `--json` to any command for machine-readable output.

## How to work with it

1. Always run `overview` first. It replaces reading the README and listing the tree, in about 150–700 tokens.
2. For "where is X" questions use `find`; for anything else use `ask`. Treat the citations as the evidence: open more lines with `read` before claiming behaviour you haven't seen.
3. Cite locations back to the user as `path:line` and include the GitHub link Gitsum prints.
4. If a command fails, read its one-line hint: `not_found` → private or misspelled (set `GITHUB_TOKEN`); `rate_limited` → set `GITHUB_TOKEN` (60 → 5,000 requests/hour) or wait the minutes shown.

## Environment (optional)

- `GITHUB_TOKEN` — private repos you can access, and higher rate limits.
- `GITSUM_TEAM_KEY` + `GITSUM_API` — Gitsum Team plan: `ask` also returns a written AI answer from the hosted service.

## One-line use from any agent

```text
Use this Skill: https://raw.githubusercontent.com/in-c0/gitsum/main/skills/gitsum/SKILL.md
Explain how <owner/repo> handles authentication, with file:line citations.
```
