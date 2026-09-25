// Render the social cards (static/images/og*.png). The art is drawn, not stock: the mesh cards run
// the page's own mesh engine in a headless browser; the explorer card uses the kernel family already
// drawn for the home page. Needs playwright and the site served locally:
//     node tools/make_og.js [http://127.0.0.1:8770]
const { chromium } = require("playwright");
const path = require("path");

const BASE = process.argv[2] || "http://127.0.0.1:8770";
const OUT = path.join(__dirname, "..", "static", "images");

const CARDS = [
  { out: "og.png", art: "mesh", surface: "hills", seed: 20260924,
    eyebrow: "Quantitative research &middot; PhD, UC Santa Barbara",
    title: "Sam Babichenko",
    sub: "Game theory, empirical Bayes, machine learning, and quantitative finance." },
  { out: "og-noisestate.png", art: "image", src: "/images/card-noisestate-light.webp", titleWidth: 440,   // clear of the curves
    eyebrow: "sbabichenko.com/noisestate",
    title: "Noise-State Game Explorer",
    sub: "Solve a linear-quadratic-Gaussian game in your browser: shock responses, sample paths, sweeps." },
  { out: "og-mesh.png", art: "mesh", surface: "cliff", seed: 77,
    eyebrow: "sbabichenko.com/mesh",
    title: "Decision Mesh",
    sub: "Freeform triangles, right triangles, rectangles and a regression tree fit the same noisy surface." },
  { out: "og-gate.png", art: "image", src: "/images/card-gate-light.webp",
    eyebrow: "sbabichenko.com/gate",
    title: "Decision Mesh, Live",
    sub: "Coin flips fitted in your browser by an estimator that cuts only where a false-discovery gate allows." },
  { out: "og-idea.png", art: "image", fit: "contain", src: "/images/card-idea.webp",
    eyebrow: "sbabichenko.com/dissertation/idea",
    title: "The Noise-State, Explained",
    sub: "Why knowing what others know changes everything, drawn as you read." },
  { out: "og-wedge.png", art: "image", fit: "contain", src: "/images/card-wedge.webp",
    eyebrow: "sbabichenko.com/dissertation/wedge",
    title: "The Price of Changing Someone's Mind",
    sub: "The information wedge, solved live: what moving an opponent's beliefs is worth." },
  { out: "og-gate-how.png", art: "image", fit: "contain", src: "/images/card-gate-how.webp",
    eyebrow: "sbabichenko.com/gate/how",
    title: "The False-Discovery Gate, Step by Step",
    sub: "How the decision mesh decides where the data justify more detail, and when to stop." },
  { out: "og-dissertation.png", art: "shocks", seed: 5, titleWidth: 1000,   // one line: the title is the book's first sentence
    eyebrow: "sbabichenko.com/dissertation",
    title: "No one knows much.",
    sub: "Noise-State Calculus for Dynamic Games with Strategic Information. The whole dissertation, to read in the browser." },
  { out: "og-partial-pooling.png", art: "image", fit: "contain", src: "/images/card-partial-pooling.svg",
    eyebrow: "sbabichenko.com/writing",
    title: "Partial Pooling for Interaction Effects",
    sub: "Every additional filter halves the sample again. Each one has a cost." },
  { out: "og-cv.png", art: "mesh", surface: "cliff", seed: 2021,
    eyebrow: "sbabichenko.com/cv",
    title: "Curriculum Vitae",
    sub: "Sam Babichenko. Quantitative researcher; PhD in Statistics and Applied Probability, UC Santa Barbara, 2026." },
  { out: "og-writing.png", art: "mesh", surface: "hills", seed: 1108,
    eyebrow: "sbabichenko.com/writing",
    title: "Writing",
    sub: "Longer pieces on statistics and markets." },
  // one card per dissertation page, from the same file the page head reads its description from; a long
  // description is cut to its first sentence so the card keeps clear of the footer
  ...Object.entries(require("../data/dissertation_previews.json")).filter(([k]) => !k.startsWith("_")).map(([slug, d], i) => ({
    out: `og-diss-${slug}.png`, art: "shocks", seed: 101 + 17 * i, fade: true,   // the walks kept off the text
    titleWidth: 1040, titleSize: d.title.length > 40 ? 58 : d.title.length > 22 ? 68 : 84,
    eyebrow: `${d.eyebrow} &middot; Noise-State Calculus`,
    title: d.title,
    sub: d.card || (d.description.length > 150 ? d.description.slice(0, d.description.indexOf(". ") + 1) : d.description) })),
];

const page = (c) => `<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="${BASE}/fonts/newsreader.css">
<style>
 html,body{margin:0;width:1200px;height:630px;overflow:hidden}
 body{background:#FFFFF0;position:relative;font-family:Newsreader,Georgia,serif;color:#17171b}
 canvas,img.art{position:absolute;inset:0;width:1200px;height:630px;object-fit:cover}
 canvas.fade{-webkit-mask-image:linear-gradient(100deg,transparent 42%,#000 75%)}
 img.art{opacity:.9;object-position:right center;-webkit-mask-image:linear-gradient(100deg,transparent 40%,#000 66%)}
 img.art.contain{object-fit:contain;left:auto;right:28px;width:500px;opacity:1;-webkit-mask-image:none}
 body.narrow h1{font-size:62px;max-width:600px}
 body.narrow p{font-size:26px;max-width:580px}
 .e{position:absolute;left:78px;top:128px;font-size:19px;letter-spacing:.14em;text-transform:uppercase;color:#85858f;z-index:2;font-family:ui-sans-serif,system-ui,sans-serif}
 .t{position:absolute;left:78px;top:190px;z-index:2}
 h1{font-size:${c.titleSize || (c.title.length > 22 ? 72 : 88)}px;font-weight:400;letter-spacing:-0.02em;margin:0 0 20px;max-width:760px;line-height:1.02}
 p{font-size:29px;color:#4a4a55;margin:0;max-width:660px;line-height:1.35}
 .u{position:absolute;left:78px;bottom:70px;font-size:24px;color:#1f3fd0;z-index:2;font-family:ui-sans-serif,system-ui,sans-serif}
</style></head><body class="${c.fit === "contain" ? "narrow" : ""}">
${c.art === "mesh" || c.art === "shocks" ? `<canvas id="c"${c.fade ? ' class="fade"' : ""} width="2400" height="1260"></canvas>` : `<img class="art${c.fit === "contain" ? " contain" : ""}" src="${BASE}${c.src}">`}
<div class="e">${c.eyebrow}</div>
<div class="t"><h1${c.titleWidth ? ` style="max-width:${c.titleWidth}px"` : ""}>${c.title}</h1><p>${c.sub}</p></div>
<div class="u">sbabichenko.com</div>
<script src="${BASE}/mesh/decision-mesh.js"></script>
<script>
const SURFACES = {
  hills: (x,y) => 2.4*Math.exp(-((x-1.5)**2+(y-1)**2)/2.2) - 2.1*Math.exp(-((x+1.5)**2+(y+1.3)**2)/1.6),
  cliff: (x,y) => 2.2*Math.tanh(1.3*(x-0.7*y)),
};
function mul(a){return function(){a|=0;a=(a+0x6d2b79f5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
function gs(r){let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());}
const c = document.getElementById('c');
if (c && ${JSON.stringify(c.art)} === "shocks") {
  const g = c.getContext('2d'), W = c.width, H = c.height, r = mul(${c.seed || 1});
  g.lineCap = 'round';
  for (let k = 0; k < 9; ++k) {
    let y = k === 3 ? H * 0.8 : H * (0.15 + 0.7 * r());
    g.beginPath(); g.moveTo(0, y);
    for (let i = 1; i <= 220; ++i) { y += gs(r) * H * 0.01; y = Math.max(H * 0.05, Math.min(H * 0.95, y)); g.lineTo(W * i / 220, y); }
    g.strokeStyle = k === 3 ? 'rgba(31,63,208,0.55)' : 'rgba(20,22,40,0.11)'; g.lineWidth = k === 3 ? 4 : 2.4; g.stroke();
  }
} else if (c) {
  const f = SURFACES[${JSON.stringify(c.surface || "hills")}];
  const N=6000, X=new Float64Array(2*N), Y=new Float64Array(N), r=mul(${c.seed || 1});
  for(let i=0;i<N;i++){const x=-4+8*r(),y=-4+8*r();X[2*i]=x;X[2*i+1]=y;Y[i]=f(x,y)+0.45*gs(r);}
  DM.reset();
  const m=new DM.DecisionMesh(X,Y,{maxAspectRatio:5,minPoints:4,refresh:true,rng:mul(7)});
  for(let i=0;i<2600&&m.activeFaces.size<900;i++) if(m.step(0.08)==='none') break;
  const g=c.getContext('2d'), W=c.width, H=c.height;
  const side=H*1.85, ox=W-side*0.86, oy=(H-side)/2;
  const px=x=>ox+((x+4)/8)*side, py=y=>oy+((4-y)/8)*side;
  const fx=W*0.80, fy=H*0.48, R=0.95*Math.max(W*0.52,H);
  g.lineWidth=1.5; g.lineCap='round';
  for(const e of m.activeEdges){
    const x0=px(e.v0.x),y0=py(e.v0.y),x1=px(e.v1.x),y1=py(e.v1.y);
    const mx=(x0+x1)/2,my=(y0+y1)/2;
    const d=Math.hypot(mx-fx,(my-fy)*0.85)/R; if(d>=1) continue;
    const band=Math.min(1,Math.min(my,H-my)/(0.28*H));
    const v=(1-d)*(1-d)*band*band;
    g.strokeStyle='rgba(20,22,40,'+(0.34*v)+')';
    g.beginPath(); g.moveTo(x0,y0); g.lineTo(x1,y1); g.stroke();
  }
}
document.title='ready';
</script></body></html>`;

(async () => {
  const b = await chromium.launch();
  const only = process.argv[3];
  for (const c of CARDS) {
    if (only && c.out !== only) continue;
    // served from the site's own origin (a file:// page may not load the site's fonts: they are cross-origin
    // there, and a static server sends no CORS header), then held until Newsreader has loaded
    const url = BASE.replace(/\/$/, "") + "/__og_card.html";
    const p = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 2 });
    await p.route(url, (r) => r.fulfill({ contentType: "text/html; charset=utf-8", body: page(c) }));
    await p.goto(url);
    await p.waitForFunction(() => document.title === "ready", null, { timeout: 60000 });
    await p.evaluate(() => document.fonts.ready);
    if (!(await p.evaluate(() => document.fonts.check("88px Newsreader"))))
      throw new Error(`${c.out}: Newsreader did not load from ${BASE}`);
    await p.waitForTimeout(900);
    await p.screenshot({ path: path.join(OUT, c.out) });
    await p.close();
    console.log("wrote", c.out);
  }
  await b.close();
})();
