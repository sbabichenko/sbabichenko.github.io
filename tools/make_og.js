// Render the social card: the mesh engine drawing a figure, with the name set over it.
const { chromium } = require('playwright');
const fs = require('fs');
const html = `<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300..600&display=swap">
<style>
 html,body{margin:0;width:1200px;height:630px;overflow:hidden}
 body{background:#FFFFF0;position:relative;font-family:Newsreader,Georgia,serif;color:#17171b}
 canvas{position:absolute;inset:0;width:1200px;height:630px}
 .t{position:absolute;left:78px;top:196px;z-index:2}
 h1{font-size:88px;font-weight:400;letter-spacing:-0.02em;margin:0 0 18px}
 p{font-size:30px;color:#4a4a55;margin:0;max-width:640px;line-height:1.35}
 .e{position:absolute;left:78px;top:130px;font-size:19px;letter-spacing:.14em;text-transform:uppercase;color:#85858f;z-index:2;font-family:ui-sans-serif,system-ui,sans-serif}
 .u{position:absolute;left:78px;bottom:72px;font-size:24px;color:#1f3fd0;z-index:2;font-family:ui-sans-serif,system-ui,sans-serif}
</style></head><body>
<canvas id="c" width="2400" height="1260"></canvas>
<div class="e">Statistics &amp; Applied Probability &middot; UC Santa Barbara</div>
<div class="t"><h1>Samuel Babichenko</h1><p>Games played through noise, and regressions that decide where to look.</p></div>
<div class="u">sbabichenko.com</div>
<script src="http://127.0.0.1:8770/mesh/decision-mesh.js"></script>
<script>
function mul(a){return function(){a|=0;a=(a+0x6d2b79f5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
function gs(r){let u=0;while(u===0)u=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*r());}
const f=(x,y)=>2.4*Math.exp(-((x-1.5)**2+(y-1)**2)/2.2)-2.1*Math.exp(-((x+1.5)**2+(y+1.3)**2)/1.6);
const N=6000,X=new Float64Array(2*N),Y=new Float64Array(N),r=mul(20260924);
for(let i=0;i<N;i++){const x=-4+8*r(),y=-4+8*r();X[2*i]=x;X[2*i+1]=y;Y[i]=f(x,y)+0.45*gs(r);}
DM.reset();
const m=new DM.DecisionMesh(X,Y,{maxAspectRatio:5,minPoints:4,refresh:true,rng:mul(7)});
for(let i=0;i<2600&&m.activeFaces.size<900;i++) if(m.step(0.08)==='none') break;
const c=document.getElementById('c'),g=c.getContext('2d'),W=c.width,H=c.height;
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
document.title='ready';
</script></body></html>`;
fs.writeFileSync('/tmp/og.html', html);
(async()=>{ const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1200,height:630}, deviceScaleFactor:2});
 await p.goto('file:///tmp/og.html'); await p.waitForFunction(()=>document.title==='ready',null,{timeout:60000}); await p.waitForTimeout(900);
 await p.screenshot({path:'/home/claude/site/static/images/og.png'});
 await b.close(); })();
