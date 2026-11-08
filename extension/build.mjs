// Builds the Chrome extension into extension/dist/ and zips it to extension/gitsum-extension.zip.
//   node extension/build.mjs            → production build
//   node extension/build.mjs --fixtures → also copies extension/fixtures (judge only; never shipped in the zip)
import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "dist");
const withFixtures = process.argv.includes("--fixtures");

rmSync(dist, { recursive: true, force: true });
mkdirSync(join(dist, "icons"), { recursive: true });

await build({
  entryPoints: { panel: join(here, "src/panel.js") },
  outdir: dist,
  bundle: true,
  format: "esm",
  target: "chrome116",
  minify: true,
  sourcemap: false,
  legalComments: "none",
  logLevel: "warning",
});
// the service worker and content script run as classic scripts, not modules
for (const name of ["background", "content"])
  await build({ entryPoints: [join(here, `src/${name}.js`)], outfile: join(dist, `${name}.js`), bundle: true, format: "iife", target: "chrome116", minify: true, logLevel: "warning" });

copyFileSync(join(here, "sidepanel.html"), join(dist, "sidepanel.html"));
copyFileSync(join(here, "src/panel.css"), join(dist, "panel.css"));

// manifest: add the Team API origin when config.js sets one
const manifest = JSON.parse(readFileSync(join(here, "manifest.json"), "utf8"));
const teamApi = readFileSync(join(here, "src/config.js"), "utf8").match(/export const TEAM_API = "([^"]*)"/)?.[1] || "";
if (teamApi) manifest.host_permissions.push(`${new URL(teamApi).origin}/*`);
writeFileSync(join(dist, "manifest.json"), JSON.stringify(manifest, null, 2));

for (const size of [16, 32, 48, 128]) writeFileSync(join(dist, "icons", `icon${size}.png`), icon(size));

if (!withFixtures) {
  const zip = join(here, "gitsum-extension.zip");
  rmSync(zip, { force: true });
  execFileSync("zip", ["-q", "-r", "-X", zip, "."], { cwd: dist });
  console.log(`built extension/dist and ${zip}`);
} else {
  mkdirSync(join(dist, "fixtures"), { recursive: true });
  for (const f of ["repo.json", "large.json"]) if (existsSync(join(here, "fixtures", f))) copyFileSync(join(here, "fixtures", f), join(dist, "fixtures", f));
  console.log("built extension/dist with fixtures (judge build, not for release)");
}

// ---------- icon: three stacked strata on slate, top layer in ochre (original mark) ----------
function icon(size) {
  const S = 4; // supersampling
  const N = size * S;
  const px = new Float32Array(size * size * 4);
  const slate = [24, 29, 36];
  const layers = [
    { y: 0.24, c: [227, 154, 82] },
    { y: 0.45, c: [228, 226, 220] },
    { y: 0.66, c: [143, 151, 160] },
  ];
  const r = 0.2 * N; // corner radius
  for (let sy = 0; sy < N; sy++) {
    for (let sx = 0; sx < N; sx++) {
      const x = sx + 0.5;
      const y = sy + 0.5;
      // rounded square mask
      const dx = Math.max(r - x, 0, x - (N - r));
      const dy = Math.max(r - y, 0, y - (N - r));
      if (dx * dx + dy * dy > r * r) continue;
      let col = slate;
      const u = x / N;
      const v = y / N;
      for (const L of layers) {
        // a sheared slab: parallelogram leaning right, thickness 0.13
        const t = (v - L.y) / 0.13;
        if (t < 0 || t > 1) continue;
        const left = 0.2 + (1 - t) * 0.08;
        const right = 0.72 + (1 - t) * 0.08;
        if (u >= left && u <= right) col = L.c;
      }
      const o = (Math.floor(sy / S) * size + Math.floor(sx / S)) * 4;
      px[o] += col[0];
      px[o + 1] += col[1];
      px[o + 2] += col[2];
      px[o + 3] += 255;
    }
  }
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    for (let x = 0; x < size; x++) {
      const o = (y * size + x) * 4;
      const a = px[o + 3];
      const d = y * (size * 4 + 1) + 1 + x * 4;
      raw[d] = a ? Math.round(px[o] / (a / 255)) : 0;
      raw[d + 1] = a ? Math.round(px[o + 1] / (a / 255)) : 0;
      raw[d + 2] = a ? Math.round(px[o + 2] / (a / 255)) : 0;
      raw[d + 3] = Math.round(a / (S * S));
    }
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}

function crc32(buf) {
  let c = ~0;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}
