// Recreates the HUD judge's fixture clones at the exact commits recorded in extension/fixtures/*.json,
// so the judge reads the same file contents the fixtures were built from (CI and fresh machines).
// Usage: node scripts/fetch-hud-clones.mjs [targetDir]   (default: each fixture's cloneDir; or $HUD_CLONES)
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = process.argv[2] || process.env.HUD_CLONES || "";
for (const name of ["repo.json", "large.json"]) {
  const fx = JSON.parse(readFileSync(join(root, "extension", "fixtures", name), "utf8"));
  if (!fx.cloneUrl || !fx.sha) throw new Error(`${name} has no cloneUrl/sha — rebuild it with scripts/make-hud-fixtures.mjs`);
  const dir = target ? join(target, basename(fx.cloneDir)) : fx.cloneDir;
  const git = (...a) => execFileSync("git", ["-C", dir, ...a], { stdio: "inherit" });
  if (existsSync(join(dir, ".git"))) {
    try {
      execFileSync("git", ["-C", dir, "cat-file", "-e", `${fx.sha}^{commit}`]);
      git("checkout", "-q", "--detach", fx.sha);
      console.log(`${name}: ${dir} @ ${fx.sha.slice(0, 7)} (cached)`);
      continue;
    } catch {}
  } else {
    mkdirSync(dir, { recursive: true });
    git("init", "-q");
    git("remote", "add", "origin", fx.cloneUrl);
  }
  git("fetch", "-q", "--depth", "1", "origin", fx.sha);
  git("checkout", "-q", "--detach", "FETCH_HEAD");
  console.log(`${name}: ${dir} @ ${fx.sha.slice(0, 7)}`);
}
