// GitHub linguist language colours (github-linguist/linguist languages.yml) for the grains.
import { LANG_BY_EXT } from "../../src/core/analyze.js";

export const LINGUIST = {
  JavaScript: "#f1e05a", TypeScript: "#3178c6", Python: "#3572A5", Java: "#b07219", Go: "#00ADD8",
  Rust: "#dea584", Ruby: "#701516", PHP: "#4F5D95", C: "#555555", "C++": "#f34b7d", "C#": "#178600",
  Swift: "#F05138", Kotlin: "#A97BFF", "Objective-C": "#438eff", Shell: "#89e051", HTML: "#e34c26",
  CSS: "#663399", SCSS: "#c6538c", Less: "#1d365d", Vue: "#41b883", Svelte: "#ff3e00", Astro: "#ff5a03",
  Dart: "#00B4AB", Scala: "#c22d40", Elixir: "#6e4a7e", Erlang: "#B83998", Haskell: "#5e5086",
  Clojure: "#db5855", Lua: "#000080", R: "#198CE7", Julia: "#a270ba", Zig: "#ec915c", Nim: "#ffc200",
  Solidity: "#AA6746", PowerShell: "#012456", SQL: "#e38c00", GraphQL: "#e10098", HCL: "#844FBA",
  Nix: "#7e7eff", OCaml: "#ef7a08", Elm: "#60B5CC", Perl: "#0298c3", Groovy: "#4298b8", "F#": "#b845fc",
  "Jupyter Notebook": "#DA5B0B", Markdown: "#083fa1", MDX: "#fcb32c", JSON: "#292929", YAML: "#cb171e",
  TOML: "#9c4221", Dockerfile: "#384d54", Makefile: "#427819", SVG: "#ff9900", "Protocol Buffers": "#a0a0a0",
};

const EXTRA_EXT = { md: "Markdown", markdown: "Markdown", mdx: "MDX", json: "JSON", jsonc: "JSON", yml: "YAML", yaml: "YAML", toml: "TOML", svg: "SVG" };

/** Language of a path (linguist name) or "" when unknown. */
export function langOf(path) {
  const base = path.slice(path.lastIndexOf("/") + 1);
  if (/^Dockerfile/i.test(base)) return "Dockerfile";
  if (/^(GNU)?makefile$/i.test(base)) return "Makefile";
  const ext = (base.match(/\.([^.]+)$/)?.[1] || "").toLowerCase();
  return LANG_BY_EXT[ext] || EXTRA_EXT[ext] || "";
}

// Data and prose files are sediment, not source: GitHub's language bar ignores them too.
const SEDIMENT = new Set(["JSON", "YAML", "TOML", "Markdown", "MDX", "SVG"]);

/** Grain colour: linguist colour for code, the theme's neutral sediment colour for data/prose/unknown. */
export function langColor(lang, neutral) {
  return SEDIMENT.has(lang) ? neutral : LINGUIST[lang] || neutral;
}
