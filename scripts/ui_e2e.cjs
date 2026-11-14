// Browser end-to-end: real Chromium, fake microphone that plays a real singing recording.
//   node scripts/ui_e2e.cjs <baseUrl> <queryWav> <expectedSingerSubstring> <shotDir>
const { chromium } = require("playwright");
const [base, wav, expect, shotDir] = process.argv.slice(2);
(async () => {
  const browser = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`, "--no-sandbox"] });
  const ctx = await browser.newContext({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: 1, permissions: ["microphone"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(base);
  await page.waitForFunction(() => document.getElementById("dbline").textContent.includes("voices"));
  console.log("DB line:", await page.textContent("#dbline"));

  // ---- record with the fake mic
  await page.click("#rec-btn");
  await page.waitForTimeout(9000);
  await page.click("#rec-btn");
  await page.waitForFunction(() => !document.getElementById("go-btn").disabled);
  console.log("recording captured, recording time label:", await page.textContent("#rec-time"));
  await page.click("#go-btn");
  await page.waitForSelector(".match", { timeout: 120000 });
  const top = await page.$eval(".match", (m) => m.innerText);
  console.log("TOP MATCH CARD:\n" + top.split("\n").slice(0, 8).join("\n"));
  const attr = await page.$eval(".match .attr", (e) => e.innerText);
  console.log("attribution line:", attr);
  const playable = await page.$eval(".match video", (v) => new Promise((res) => {
    if (v.readyState >= 1) return res(v.duration);
    v.addEventListener("loadedmetadata", () => res(v.duration)); v.addEventListener("error", () => res(-1)); setTimeout(() => res(-2), 15000); v.load();
  }));
  console.log("top clip video duration (s):", playable);
  await page.screenshot({ path: `${shotDir}/results.png`, fullPage: true });
  const ok = attr.includes(expect) && playable > 1;
  console.log(ok ? "PASS: top match is the expected singer and the clip plays" : `FAIL: expected '${expect}' in attribution`);
  // ---- other tabs render
  await page.click("#tab-db"); await page.waitForSelector(".vcard video"); await page.screenshot({ path: `${shotDir}/database.png` });
  await page.click("#tab-crawl"); await page.waitForSelector("#crawl-kv dd"); await page.screenshot({ path: `${shotDir}/crawler.png` });
  if (errors.length) console.log("BROWSER ERRORS:", errors);
  await browser.close();
  process.exit(ok && !errors.length ? 0 : 1);
})().catch((e) => { console.error(e); process.exit(2); });
