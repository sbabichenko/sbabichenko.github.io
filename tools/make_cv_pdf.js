// Print the CV page to static/Resume_Samuel_Babichenko.pdf, so the download always matches the page.
// Needs playwright and the site served locally (zola serve, or a build behind any static server):
//     node tools/make_cv_pdf.js [http://127.0.0.1:8770]
// Links in the PDF point at https://sbabichenko.com whatever server it was printed from.
const { chromium } = require("playwright");
const path = require("path");
const BASE = (process.argv[2] || "http://127.0.0.1:8770").replace(/\/$/, "");
const OUT = path.join(__dirname, "..", "static", "Resume_Samuel_Babichenko.pdf");
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  // the build's stylesheets and fonts are linked as https://sbabichenko.com/...; serve those from the local
  // build too, or the PDF is printed with the live site's CSS rather than the one being edited
  await p.route("https://sbabichenko.com/**", async (r) => {
    const u = new URL(r.request().url());
    r.fulfill({ response: await p.request.get(BASE + u.pathname + u.search) });
  });
  await p.goto(BASE + "/cv/", { waitUntil: "networkidle" });
  await p.evaluate(async (base) => {
    for (const a of document.querySelectorAll("a[href]")) {
      const u = new URL(a.getAttribute("href"), location.href);
      if (u.origin === location.origin) a.href = "https://sbabichenko.com" + u.pathname + u.hash;
    }
    await document.fonts.ready;
  }, BASE);
  await p.emulateMedia({ media: "print" });
  await p.pdf({ path: OUT, format: "Letter", printBackground: false, preferCSSPageSize: true });
  await b.close();
  console.log("wrote", OUT);
})();
