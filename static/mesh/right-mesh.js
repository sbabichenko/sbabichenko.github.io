// A right-triangle decision mesh: the geometry of Sam Babichenko's triangular-decision-mesh, grown by
// this page's greedy rule.
//
// Every triangle is right isosceles and is only ever cut one way: through the midpoint of its
// hypotenuse (newest-vertex bisection), which yields two right isosceles halves. When the triangle on the
// other side of that hypotenuse would be left with a hanging vertex, it is cut first, recursively (the
// completion step), so the mesh is always conforming. The shapes can never degenerate, whatever the
// data do, which is the point of the geometry; the price is that a cut cannot follow an arbitrary
// direction, as the freeform mesh's can.
//
// Heights are hierarchical, as in the rectangular mesh: a vertex is born bisecting an edge, its height
// is the average of that edge's endpoints plus its own surplus, and only a free vertex has a surplus.
// A vertex made by completion is structural: it starts with no surplus, so the cut leaves the surface
// unchanged until the greedy rule frees it.
(function (root) {
  "use strict";
  const S = 1 << 20;                                 // the dyadic lattice over the square

  let SEQ = 0;
  class TVertex {
    constructor(ix, iy, p0, p1) {
      this.id = SEQ++; this.ix = ix; this.iy = iy; this.p0 = p0; this.p1 = p1;
      this.free = false; this.surplus = 0;
    }
    get height() { return (this.p0 ? 0.5 * (this.p0.height + this.p1.height) : 0) + this.surplus; }
    ancestors() {
      if (this.anc) return this.anc;
      const m = new Map([[this, 1]]);
      if (this.p0) for (const p of [this.p0, this.p1])
        for (const [v, f] of p.ancestors()) m.set(v, (m.get(v) || 0) + f / 2);
      return (this.anc = m);
    }
  }

  // a is the right-angle vertex, b-c the hypotenuse (the refinement edge)
  class TFace {
    constructor(a, b, c) { this.a = a; this.b = b; this.c = c; this.idx = []; this.kids = null; }
    get v() { return [this.a, this.b, this.c]; }
  }
  const ekey = (u, v) => (u.id < v.id ? u.id * 1e7 + v.id : v.id * 1e7 + u.id);

  class RightMesh {
    constructor(X, Y, { depth = 4, minPoints = 0 } = {}) {
      this.X = X; this.Y = Y; this.n = Y.length; this.minPoints = minPoints;
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (let i = 0; i < this.n; ++i) {
        const x = X[2 * i], y = X[2 * i + 1];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x; if (y < ymin) ymin = y; if (y > ymax) ymax = y;
      }
      Object.assign(this, { xmin, xmax, ymin, ymax });
      this.sx = S / (xmax - xmin || 1); this.sy = S / (ymax - ymin || 1);
      this.px = new Float64Array(this.n); this.py = new Float64Array(this.n);
      for (let i = 0; i < this.n; ++i) { this.px[i] = (X[2 * i] - xmin) * this.sx; this.py[i] = (X[2 * i + 1] - ymin) * this.sy; }
      this.verts = new Map(); this.leaves = new Set(); this.byEdge = new Map();
      this.pred = new Float64Array(this.n); this.W = new Float64Array(3 * this.n);
      this.face = new Array(this.n);
      this.steps = 0;

      // the square cut along its diagonal, bottom-left to top-right, as the engine starts
      const corner = (ix, iy) => { const v = new TVertex(ix, iy, null, null); v.free = true; this.verts.set(this.key(ix, iy), v); return v; };
      const bl = corner(0, 0), br = corner(S, 0), tl = corner(0, S), tr = corner(S, S);
      const top = new TFace(tl, bl, tr), bot = new TFace(br, tr, bl);
      this.roots = [top, bot];
      for (const f of this.roots) this.addLeaf(f);
      for (let i = 0; i < this.n; ++i) (this.side(top, i) ? top : bot).idx.push(i);
      // then uniform refinement to a coarse starting mesh (2 * 2^depth triangles), all structural
      for (let d = 0; d < depth; ++d) for (const f of [...this.leaves]) if (this.leaves.has(f)) this.bisect(f);
      this.refresh();
    }
    key(ix, iy) { return ix * (S + 1) + iy; }
    // is point i in the top-left root (on or above the diagonal)?
    side(top, i) { return this.py[i] >= this.px[i]; }

    addLeaf(f) {
      this.leaves.add(f);
      for (const [u, v] of [[f.a, f.b], [f.b, f.c], [f.c, f.a]]) {
        const k = ekey(u, v); let s = this.byEdge.get(k);
        if (!s) { s = new Set(); this.byEdge.set(k, s); }
        s.add(f);
      }
    }
    dropLeaf(f) {
      this.leaves.delete(f);
      for (const [u, v] of [[f.a, f.b], [f.b, f.c], [f.c, f.a]]) {
        const k = ekey(u, v), s = this.byEdge.get(k);
        if (s) { s.delete(f); if (!s.size) this.byEdge.delete(k); }
      }
    }
    across(f) {                                       // the leaf on the other side of f's hypotenuse
      const s = this.byEdge.get(ekey(f.b, f.c));
      if (s) for (const g of s) if (g !== f) return g;
      return null;
    }
    born(ix, iy, p, q) {
      const k = this.key(ix, iy);
      let v = this.verts.get(k);
      if (!v) { v = new TVertex(ix, iy, p, q); this.verts.set(k, v); }
      return v;
    }

    // cut f through its hypotenuse's midpoint, completing the neighbour first when it needs it
    bisect(f) {
      for (let guard = 0; guard < 64; ++guard) {
        const g = this.across(f);
        if (!g || (ekey(g.b, g.c) === ekey(f.b, f.c))) {
          this.halve(f);
          if (g) this.halve(g);
          return;
        }
        this.bisect(g);                               // completion: g must be cut first
      }
    }
    halve(f) {
      const m = this.born((f.b.ix + f.c.ix) / 2, (f.b.iy + f.c.iy) / 2, f.b, f.c);
      const k0 = new TFace(m, f.a, f.b), k1 = new TFace(m, f.c, f.a);
      // the chord a-m separates the halves
      const ax = f.a.ix, ay = f.a.iy, dx = m.ix - ax, dy = m.iy - ay;
      const bs = Math.sign(dx * (f.b.iy - ay) - dy * (f.b.ix - ax));
      for (const i of f.idx) (Math.sign(dx * (this.py[i] - ay) - dy * (this.px[i] - ax)) === bs ? k0 : k1).idx.push(i);
      f.kids = [k0, k1]; f.chord = bs;
      this.dropLeaf(f); this.addLeaf(k0); this.addLeaf(k1);
      f.idx = [];
    }

    // --------------------------------------------------------------- the surface
    bary(f, x, y) {
      const { a, b, c } = f;
      const det = (b.iy - c.iy) * (a.ix - c.ix) + (c.ix - b.ix) * (a.iy - c.iy);
      const l0 = ((b.iy - c.iy) * (x - c.ix) + (c.ix - b.ix) * (y - c.iy)) / det;
      const l1 = ((c.iy - a.iy) * (x - c.ix) + (a.ix - c.ix) * (y - c.iy)) / det;
      return [l0, l1, 1 - l0 - l1];
    }
    refresh() {
      const W = this.W;
      for (const f of this.leaves) {
        const ha = f.a.height, hb = f.b.height, hc = f.c.height;
        for (const i of f.idx) {
          const [w0, w1, w2] = this.bary(f, this.px[i], this.py[i]);
          W[3 * i] = w0; W[3 * i + 1] = w1; W[3 * i + 2] = w2;
          this.pred[i] = w0 * ha + w1 * hb + w2 * hc;
          this.face[i] = f;
        }
      }
    }
    leafAt(ix, iy) {
      let f = iy >= ix ? this.roots[0] : this.roots[1];
      while (f.kids) {
        const ax = f.a.ix, ay = f.a.iy, m = f.kids[0].a, dx = m.ix - ax, dy = m.iy - ay;
        f = Math.sign(dx * (iy - ay) - dy * (ix - ax)) === f.chord ? f.kids[0] : f.kids[1];
      }
      return f;
    }
    at(x, y) {
      const ix = Math.min(S, Math.max(0, (x - this.xmin) * this.sx)), iy = Math.min(S, Math.max(0, (y - this.ymin) * this.sy));
      const f = this.leafAt(ix, iy), [w0, w1, w2] = this.bary(f, ix, iy);
      return w0 * f.a.height + w1 * f.b.height + w2 * f.c.height;
    }

    // --------------------------------------------------------------- one greedy step
    // Candidates: every vertex's surplus (its column reaches through its descendants, one half per
    // generation), and every leaf's hypotenuse midpoint (the hat the cut would give it on the triangles
    // that have that hypotenuse). The winner is cut in, with completion, and its surplus is then fitted
    // against the exact column of the resulting mesh, so the squared error cannot rise.
    buildIndex() {
      const byAnc = new Map(), where = new Map(), byMid = new Map();
      for (const f of this.leaves) {
        if (!f.idx.length) continue;
        const cs = f.v;
        for (let k = 0; k < 3; ++k) for (const [v, w] of cs[k].ancestors()) {
          let a = byAnc.get(v.id);
          if (!a) { a = []; byAnc.set(v.id, a); where.set(v.id, v); }
          a.push(f, k, w);
        }
        const mx = (f.b.ix + f.c.ix) / 2, my = (f.b.iy + f.c.iy) / 2, key = this.key(mx, my);
        let m = byMid.get(key);
        if (!m) { m = { ix: mx, iy: my, faces: [] }; byMid.set(key, m); }
        m.faces.push(f);
      }
      this.ix = { byAnc, where, byMid };
    }
    column(v, mid, col) {
      const val = col.val, hit = col.hit;
      for (let t = 0; t < col.k; ++t) val[hit[t]] = 0;
      col.k = 0;
      if (v) {
        const a = this.ix.byAnc.get(v.id);
        if (a) for (let t = 0; t < a.length; t += 3) {
          const f = a[t], k = a[t + 1], w = a[t + 2];
          for (const i of f.idx) {
            const x = this.W[3 * i + k] * w;
            if (x === 0) continue;
            if (val[i] === 0) hit[col.k++] = i;
            val[i] += x;
          }
        }
      }
      if (mid) for (const f of mid.faces) {
        // the hat of the new vertex m on f's two halves: in the half (m, a, b) its weight is its
        // barycentric coordinate there, and likewise in (m, c, a)
        const m = { ix: mid.ix, iy: mid.iy };
        const h0 = new TFace(m, f.a, f.b), h1 = new TFace(m, f.c, f.a);
        for (const i of f.idx) {
          const q = this.bary(h0, this.px[i], this.py[i]);
          const w = q[0] > -1e-12 && q[1] > -1e-12 && q[2] > -1e-12 ? q[0] : this.bary(h1, this.px[i], this.py[i])[0];
          if (!(w > 0)) continue;
          if (val[i] === 0) hit[col.k++] = i;
          val[i] += w;
        }
      }
      return col;
    }
    step() {
      this.steps++;
      this.buildIndex();
      const col = this.scratch || (this.scratch = { val: new Float64Array(this.n), hit: new Int32Array(this.n), k: 0 });
      const floor = Math.max(3, 2 * this.minPoints);
      let best = null, bestGain = 0;
      const consider = (v, mid) => {
        this.column(v, mid, col);
        if (col.k < floor) return;
        let num = 0, den = 0;
        for (let t = 0; t < col.k; ++t) { const i = col.hit[t], x = col.val[i]; num += x * (this.Y[i] - this.pred[i]); den += x * x; }
        if (!(den > 1e-12)) return;
        const gain = (num * num) / den;
        if (gain > bestGain) { bestGain = gain; best = { v, mid }; }
      };
      for (const v of this.ix.where.values()) consider(v, null);
      for (const mid of this.ix.byMid.values()) if (!this.verts.has(this.key(mid.ix, mid.iy))) consider(null, mid);
      if (!best) return "none";
      let v = best.v;
      const split = !v;
      if (split) {
        // cut every leaf whose hypotenuse carries this midpoint (completion handles the rest)
        for (const f of best.mid.faces) if (this.leaves.has(f)) this.bisect(f);
        v = this.verts.get(this.key(best.mid.ix, best.mid.iy));
        this.refresh();
        this.buildIndex();
      }
      if (!v) return "none";
      this.column(v, null, col);
      let num = 0, den = 0;
      for (let t = 0; t < col.k; ++t) { const i = col.hit[t], x = col.val[i]; num += x * (this.Y[i] - this.pred[i]); den += x * x; }
      if (!(den > 1e-12)) return "none";
      v.free = true;
      v.surplus += num / den;
      this.refresh();
      return split ? "split" : "refit";
    }

    // --------------------------------------------------------------- reporting
    get faces() { return this.leaves.size; }
    freeCount() { let k = 0; for (const v of this.verts.values()) if (v.free) ++k; return k; }
    sse() { let s = 0; for (let i = 0; i < this.n; ++i) { const r = this.Y[i] - this.pred[i]; s += r * r; } return s; }
    edges() {
      const out = [];
      for (const k of this.byEdge.keys()) {
        const s = this.byEdge.get(k), f = s.values().next().value;
        for (const [u, v] of [[f.a, f.b], [f.b, f.c], [f.c, f.a]]) if (ekey(u, v) === k) { out.push([u.ix, u.iy, v.ix, v.iy]); break; }
      }
      return out;
    }
  }

  const api = { RightMesh, reset: () => { SEQ = 0; }, S };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TM = api;
})(typeof self !== "undefined" ? self : this);
