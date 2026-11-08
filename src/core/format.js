// Compact Markdown for agents and terminals. Every response stays under ~4k tokens (≈16k chars).

const MAX_CHARS = 15_000;

const clamp = (s, n = MAX_CHARS) => (s.length > n ? `${s.slice(0, n)}\n…[truncated — ask a narrower question or read a line range]` : s);

const ago = (iso) => {
  if (!iso) return "unknown";
  const d = Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000);
  return d <= 1 ? "today" : d < 60 ? `${d}d ago` : d < 730 ? `${Math.round(d / 30)}mo ago` : `${Math.round(d / 365)}y ago`;
};

export function formatOverview({ repoData: r, overview: o }) {
  const health = o.health.map((h) => `${h.ok ? "✓" : "·"} ${h.label}: ${h.value}`).join(" | ");
  const parts = [
    `# ${r.fullName}${r.archived ? " (archived)" : ""}`,
    o.oneLiner ? `> ${o.oneLiner}` : "",
    `★ ${r.stars.toLocaleString()} · forks ${r.forks.toLocaleString()} · ${r.license || "no license"} · last push ${ago(r.pushedAt)} · branch \`${r.branch}\` · ${o.fileCount.toLocaleString()} files${r.truncated ? " (tree truncated by GitHub)" : ""}`,
    o.intro ? o.intro : "",
    o.stack.length ? `**Stack:** ${o.stack.join(", ")}` : "",
    o.languages.length ? `**Languages:** ${o.languages.map((l) => `${l.name} ${Math.round(l.pct)}%`).join(", ")}` : "",
    `**Health:** ${health}`,
    o.run ? `**Run** (from ${o.run.source}):\n\`\`\`sh\n${o.run.code}\n\`\`\`` : "",
    o.structure.dirs.length ? `**Structure:**\n${o.structure.dirs.map((d) => `- \`${d.name}/\` ${d.meaning || ""} (${d.count} files)`).join("\n")}${o.structure.rootFiles ? `\n- ${o.structure.rootFiles} files at the root` : ""}` : "",
    o.keyFiles.length ? `**Read first:**\n${o.keyFiles.map((f) => `- \`${f.path}\` — ${f.reason}`).join("\n")}` : "",
  ];
  return clamp(parts.filter(Boolean).join("\n\n"));
}

export function formatAnswer(question, answer) {
  const lines = [`**Q:** ${question}`, answer.text];
  if (answer.code) lines.push(`\`\`\`sh\n${answer.code}\n\`\`\``);
  for (const c of answer.citations || []) {
    const loc = c.startLine ? `${c.path}:${c.startLine}-${c.endLine}` : c.path;
    lines.push(`\n\`${loc}\`${c.note ? ` — ${c.note}` : ""}  \n${c.url}`);
    if (c.snippet) lines.push(`\`\`\`\n${c.snippet}\n\`\`\``);
  }
  return clamp(lines.join("\n"));
}

export function formatFile(f) {
  const width = String(f.endLine).length;
  const numbered = f.text
    .split("\n")
    .map((l, i) => `${String(f.startLine + i).padStart(width, " ")}  ${l}`)
    .join("\n");
  return clamp(`\`${f.path}\` lines ${f.startLine}-${f.endLine} of ${f.totalLines}  \n${f.url}\n\`\`\`\n${numbered}\n\`\`\``);
}

export function formatTree(t, under) {
  const head = `${t.total} files${under ? ` under \`${under}/\`` : ""}${t.truncated ? " (list truncated)" : ""}`;
  return clamp(`${head}\n${t.files.map((f) => `${f.path}  (${f.size < 1024 ? `${f.size} B` : `${Math.round(f.size / 1024)} KB`})`).join("\n")}`);
}
