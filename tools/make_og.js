// Render the social cards (static/images/og*.png). The art is drawn, not stock: the mesh cards run
// the page's own mesh engine in a headless browser; the explorer card uses the kernel family already
// drawn for the home page. Needs playwright and the site served locally:
//     node tools/make_og.js [http://127.0.0.1:8770]
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const BASE = process.argv[2] || "http://127.0.0.1:8770";
const OUT = path.join(__dirname, "..", "static", "images");

const CARDS = [
  { out: "og.png", art: "mesh", surface: "hills", seed: 20260924,
    eyebrow: "Statistics &amp; Applied Probability &middot; UC Santa Barbara",
    title: "Samuel Babichenko",
    sub: "Games played through noise, and regressions that decide where to look." },
  { out: "og-noisestate.png", art: "image", src: "/images/card-noisestate-light.webp",
    eyebrow: "sbabichenko.com/noisestate",
    title: "Noise-state game explorer",
    sub: "Solve a linear-quadratic-Gaussian game in your browser: shock responses, sample paths, sweeps." },
  { out: "og-mesh.png", art: "mesh", surface: "cliff", seed: 77,
    eyebrow: "sbabichenko.com/mesh",
    title: "Decision Mesh",
    sub: "Freeform triangles, right triangles, rectangles and a regression tree fit the same noisy surface." },
  { out: "og-gate.png", art: "image", src: "/images/card-gate-light.webp",
    eyebrow: "sbabichenko.com/gate",
    title: "A mesh that knows when to stop",
    sub: "Coin flips fitted in your browser by an estimator that cuts only where a false-discovery gate allows." },
  { out: "og-dissertation.png", art: "shocks", seed: 5,
    eyebrow: "sbabichenko.com/dissertation",
    title: "No one knows much.",
    sub: "Noise-State Calculus for Dynamic Games with Strategic Information. The whole dissertation, to read in the browser." },
];

const page = (c) => `<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300..600&display=swap">
<style>
 html,body{margin:0;width:1200px;height:630px;overflow:hidden}
 body{background:#FFFFF0;position:relative;font-family:Newsreader,Georgia,serif;color:#17171b}
 canvas,img.art{position:absolute;inset:0;width:1200px;height:630px;object-fit:cover}
 img.art{opacity:.9;object-position:right center;-webkit-mask-image:linear-gradient(100deg,transparent 40%,#000 66%)}
 .e{position:absolute;left:78px;top:128px;font-size:19px;letter-spacing:.14em;text-transform:uppercase;color:#85858f;z-index:2;font-family:ui-sans-serif,system-ui,sans-serif}
 .t{position:absolute;left:78px;top:190px;z-index:2}
 h1{font-size:${c.title.length > 22 ? 72 : 88}px;font-weight:400;letter-spacing:-0.02em;margin:0 0 20px;max-width:760px;line-height:1.02}
 p{font-size:29px;color:#4a4a55;margin:0;max-width:660px;line-height:1.35}
 .u{position:absolute;left:78px;bottom:70px;font-size:24px;color:#1f3fd0;z-index:2;font-family:ui-sans-serif,system-ui,sans-serif}
</style></head><body>
${c.art === "mesh" || c.art === "shocks" ? '<canvas id="c" width="2400" height="1260"></canvas>' : `<img class="art" src="${BASE}${c.src}">`}
<div class="e">${c.eyebrow}</div>
<div class="t"><h1>${c.title}</h1><p>${c.sub}</p></div>
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
    const file = "/tmp/og_card.html";
    fs.writeFileSync(file, page(c));
    const p = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 2 });
    await p.goto("file://" + file);
    await p.waitForFunction(() => document.title === "ready", null, { timeout: 60000 });
    await p.waitForTimeout(900);
    await p.screenshot({ path: path.join(OUT, c.out) });
    await p.close();
    console.log("wrote", c.out);
  }
  await b.close();
})();
