// Visual check: render each page in local Chrome, wait for content, save screenshots, report console errors.
//   node scripts/snap.mjs <outdir> "name:/path[#clickText]...[@selector=value]..." [theme] [probeSelector]
// Clicks run in order (`#~text` matches a substring); `@selector=value` then fills an input
// (e.g. a range slider); `probeSelector`
// prints the text of the first matching elements, so a run can show that a control changed the page.
import { chromium } from "playwright-core";
const out = process.argv[2];
const pages = (process.argv[3] || "overview:/").split(",");
const theme = process.argv[4] || "dark";
const probe = process.argv[5];
const browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", args: ["--no-proxy-server"] });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
await ctx.addInitScript((t) => localStorage.setItem("defaco.theme", t), theme);
for (const spec of pages) {
  const [name, rest] = spec.split(/:(.*)/s);
  const [target, ...fills] = rest.split("@");
  const [path, ...clicks] = target.split("#");
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("http://127.0.0.1:5173" + path, { waitUntil: "networkidle" });
  await page.waitForTimeout(1800);
  for (const c of clicks) {
    const loose = c.startsWith("~");
    await page.getByText(loose ? c.slice(1) : c, { exact: !loose }).first().click();
    await page.waitForTimeout(2000);
  }
  for (const f of fills) {
    const [sel, value] = f.split(/=(?=[^=]*$)/);
    await page.locator(sel).first().fill(value);
    await page.waitForTimeout(1500);
  }
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: true });
  console.log(name, errors.length ? "ERRORS: " + errors.slice(0, 3).join(" | ") : "ok");
  if (probe) {
    const texts = await page.locator(probe).allInnerTexts();
    for (const t of texts.slice(0, 6)) console.log("  " + t.replace(/\s+/g, " ").slice(0, 160));
  }
  await page.close();
}
await browser.close();
