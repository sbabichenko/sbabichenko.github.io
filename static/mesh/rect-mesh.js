// A rectangular decision mesh: the same greedy rule as the triangular one, on rectangles cut in
// half along either axis, with bilinear interpolation inside each cell.
//
// The geometry follows Sam Babichenko's rectangular-decision-mesh. Every vertex but the four
// corners of the square is born bisecting one edge and has exactly two parents, that edge's
// endpoints; its height is their average plus its own surplus, and only a free vertex has a
// surplus. That one rule buys three things:
//   * continuity for free. A vertex left hanging in the middle of a coarser cell's edge is the
//     midpoint of a segment of that edge, so its height is exactly the edge's linear trace and the
//     surface matches across it. No constraint table, no balance rule.
//   * invisible geometry. Cutting a cell whose new vertices are not free reproduces the old surface
//     exactly (bilinear refinement is nested), so a cut can be made wherever a candidate wants one.
//   * a coefficient is freed only once it is a corner of every cell that touches it, which is what
//     keeps a surplus from tearing the surface.
//
// The fitting here is this page's greedy rule, not that repository's FDR gate: each step takes the
// candidate whose least-squares surplus removes the most squared error. The page is about the
// geometry, so both meshes are grown by the same rule.
(function (root) {
  "use strict";
  const S = 4096;                                   // the dyadic lattice over the square
  const tz = (n) => 31 - Math.clz32(n & -n);

  let SEQ = 0;
  class RVertex {
    constructor(ix, iy, p0, p1) {
      this.id = SEQ++; this.ix = ix; this.iy = iy;
      this.p0 = p0; this.p1 = p1;                   // null only for the four roots
      this.free = false; this.surplus = 0;
    }
    get height() { return (this.p0 ? 0.5 * (this.p0.height + this.p1.height) : 0) + this.surplus; }
    // which vertices' surpluses this height carries, and at what factor (one half per generation).
    // A vertex's parents are fixed at birth, so this is computed once.
    ancestors() {
      if (this.anc) return this.anc;
      const m = new Map([[this, 1]]);
      if (this.p0) for (const p of [this.p0, this.p1])
        for (const [v, f] of p.ancestors()) m.set(v, (m.get(v) || 0) + f / 2);
      return (this.anc = m);
    }
  }

  class RCell {
    constructor(x0, y0, x1, y1, c) {
      this.x0 = x0; this.y0 = y0; this.x1 = x1; this.y1 = y1;
      this.c = c;                                   // corners: (x0,y0) (x1,y0) (x0,y1) (x1,y1)
      this.idx = []; this.kids = null;
    }
  }

  class RectMesh {
    // X: Float64Array of 2n coordinates, Y: Float64Array of n values, as the triangular engine
    constructor(X, Y, { grid = 4, minPoints = 0, maxAspect = 0, rng = Math.random } = {}) {
      this.X = X; this.Y = Y; this.n = Y.length; this.minPoints = minPoints; this.rng = rng;
      this.maxAspect = maxAspect;                   // 0: no limit; else the longest a new cell may be
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (let i = 0; i < this.n; ++i) {
        const x = X[2 * i], y = X[2 * i + 1];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x; if (y < ymin) ymin = y; if (y > ymax) ymax = y;
      }
      Object.assign(this, { xmin, xmax, ymin, ymax });
      this.sx = S / (xmax - xmin || 1); this.sy = S / (ymax - ymin || 1);
      this.px = new Float64Array(this.n); this.py = new Float64Array(this.n);
      for (let i = 0; i < this.n; ++i) {
        this.px[i] = (X[2 * i] - xmin) * this.sx;
        this.py[i] = (X[2 * i + 1] - ymin) * this.sy;
      }
      this.verts = new Map();
      this.steps = 0;
      this.pred = new Float64Array(this.n);
      this.W = new Float64Array(4 * this.n);        // each point's bilinear weights in its own cell

      // the starting grid: the four corners free, every other grid vertex constrained. On this
      // grid the bisected edge is the one the canonical rule names, so the parents can be derived.
      for (const [ix, iy] of [[0, 0], [S, 0], [0, S], [S, S]]) {
        const v = new RVertex(ix, iy, null, null); v.free = true; this.verts.set(this.key(ix, iy), v);
      }
      const step = S / grid;
      this.grid = grid; this.gstep = step;
      this.leaves = []; this.roots = [];
      for (let a = 0; a < grid; ++a)
        for (let b = 0; b < grid; ++b) {
          const x0 = a * step, y0 = b * step, x1 = x0 + step, y1 = y0 + step;
          const cell = new RCell(x0, y0, x1, y1,
            [this.gridVertex(x0, y0), this.gridVertex(x1, y0), this.gridVertex(x0, y1), this.gridVertex(x1, y1)]);
          this.leaves.push(cell); this.roots.push(cell);
        }
      for (let i = 0; i < this.n; ++i) {
        const a = Math.min(grid - 1, Math.max(0, Math.floor(this.px[i] / step)));
        const b = Math.min(grid - 1, Math.max(0, Math.floor(this.py[i] / step)));
        this.roots[a * grid + b].idx.push(i);
      }
      this.refresh();
    }

    key(ix, iy) { return ix * (S + 1) + iy; }

    // the initial grid only: the vertex bisects the segment of the axis in which it is the odder
    gridVertex(ix, iy) {
      const k = this.key(ix, iy);
      let v = this.verts.get(k);
      if (v) return v;
      let axis;
      if (ix === 0 || ix === S) axis = "y";
      else if (iy === 0 || iy === S) axis = "x";
      else axis = tz(ix) < tz(iy) ? "x" : "y";
      const h = axis === "x" ? 1 << tz(ix) : 1 << tz(iy);
      const a = axis === "x" ? this.gridVertex(ix - h, iy) : this.gridVertex(ix, iy - h);
      const b = axis === "x" ? this.gridVertex(ix + h, iy) : this.gridVertex(ix, iy + h);
      v = new RVertex(ix, iy, a, b);
      this.verts.set(k, v);
      return v;
    }
    // everywhere else: the parents are the endpoints of the edge being bisected, recorded at birth
    born(ix, iy, a, b) {
      const k = this.key(ix, iy);
      let v = this.verts.get(k);
      if (v) return v;
      v = new RVertex(ix, iy, a, b);
      this.verts.set(k, v);
      return v;
    }

    // --------------------------------------------------------------- the surface
    weights(cell, i) {
      const u = (this.px[i] - cell.x0) / (cell.x1 - cell.x0);
      const v = (this.py[i] - cell.y0) / (cell.y1 - cell.y0);
      return [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v];
    }
    // the heights and each point's four weights, which the columns then only read
    refresh() {
      const W = this.W;
      for (const cell of this.leaves) {
        const h0 = cell.c[0].height, h1 = cell.c[1].height, h2 = cell.c[2].height, h3 = cell.c[3].height;
        const dx = cell.x1 - cell.x0, dy = cell.y1 - cell.y0;
        for (const i of cell.idx) {
          const u = (this.px[i] - cell.x0) / dx, v = (this.py[i] - cell.y0) / dy;
          const w0 = (1 - u) * (1 - v), w1 = u * (1 - v), w2 = (1 - u) * v, w3 = u * v;
          W[4 * i] = w0; W[4 * i + 1] = w1; W[4 * i + 2] = w2; W[4 * i + 3] = w3;
          this.pred[i] = w0 * h0 + w1 * h1 + w2 * h2 + w3 * h3;
        }
      }
    }
    leafAt(ix, iy) {
      const a = Math.min(this.grid - 1, Math.floor(ix / this.gstep));
      const b = Math.min(this.grid - 1, Math.floor(iy / this.gstep));
      let cell = this.roots[a * this.grid + b];
      while (cell.kids) {
        const k0 = cell.kids[0];
        cell = (k0.x1 === cell.x1 ? iy < k0.y1 : ix < k0.x1) ? k0 : cell.kids[1];
      }
      return cell;
    }
    at(x, y) {
      const ix = Math.min(S, Math.max(0, (x - this.xmin) * this.sx));
      const iy = Math.min(S, Math.max(0, (y - this.ymin) * this.sy));
      const cell = this.leafAt(ix, iy);
      const u = (ix - cell.x0) / (cell.x1 - cell.x0), v = (iy - cell.y0) / (cell.y1 - cell.y0);
      return (1 - u) * ((1 - v) * cell.c[0].height + v * cell.c[2].height)
           + u * ((1 - v) * cell.c[1].height + v * cell.c[3].height);
    }

    // --------------------------------------------------------------- the round's indexes
    // One pass over the cells: for each vertex, the cells whose corners its surplus moves and by
    // how much (one half per generation); and for each leaf edge midpoint, the cells a cut there
    // would give it as a corner, with that edge's half length.
    buildIndex() {
      const byAnc = new Map(), byMid = new Map(), where = new Map();
      const cells = this.leaves;
      for (let ci = 0; ci < cells.length; ++ci) {
        const cell = cells[ci];
        if (!cell.idx.length) continue;
        for (let k = 0; k < 4; ++k) {
          for (const [v, f] of cell.c[k].ancestors()) {
            if (f === 0) continue;
            let a = byAnc.get(v.id);
            if (!a) { a = []; byAnc.set(v.id, a); where.set(v.id, v); }
            a.push(ci, k, f);
          }
        }
        const mx = (cell.x0 + cell.x1) >> 1, my = (cell.y0 + cell.y1) >> 1;
        const hx = (cell.x1 - cell.x0) >> 1, hy = (cell.y1 - cell.y0) >> 1;
        //          midpoint          along   half of the bisected edge   the cell's other half width
        const mids = [[cell.x0, my, "y", hy, cell.x1 - cell.x0], [cell.x1, my, "y", hy, cell.x1 - cell.x0],
                      [mx, cell.y0, "x", hx, cell.y1 - cell.y0], [mx, cell.y1, "x", hx, cell.y1 - cell.y0]];
        for (const [ix, iy, along, half, span] of mids) {
          const key = this.key(ix, iy);
          let a = byMid.get(key);
          if (!a) { a = { ix, iy, hits: [] }; byMid.set(key, a); }
          a.hits.push({ ci, along, half, span });
        }
      }
      this.ix_byAnc = byAnc; this.ix_byMid = byMid; this.ix_where = where;
    }

    // What one unit of a candidate's surplus would add to every point: through the cells it already
    // reaches, plus, for the cells where it is only an edge midpoint, the bilinear hat the cut would
    // give it. A corner is never a midpoint, so the two sets do not overlap.
    column(v, mid, col) {
      const val = col.val, hit = col.hit;
      for (let t = 0; t < col.k; ++t) val[hit[t]] = 0;
      col.k = 0;
      if (v) {
        const a = this.ix_byAnc.get(v.id);
        if (a) for (let t = 0; t < a.length; t += 3) {
          const cell = this.leaves[a[t]], k = a[t + 1], f = a[t + 2];
          const W = this.W;
          for (const i of cell.idx) {
            const x = W[4 * i + k] * f;
            if (x === 0) continue;
            if (val[i] === 0) hit[col.k++] = i;
            val[i] += x;
          }
        }
      }
      if (mid) for (const h of mid.hits) {
        const cell = this.leaves[h.ci];
        const hx = h.along === "x" ? h.half : h.span;
        const hy = h.along === "x" ? h.span : h.half;
        for (const i of cell.idx) {
          const a2 = 1 - Math.abs(this.px[i] - mid.ix) / hx;
          if (a2 <= 0) continue;
          const b2 = 1 - Math.abs(this.py[i] - mid.iy) / hy;
          if (b2 <= 0) continue;
          if (val[i] === 0) hit[col.k++] = i;
          val[i] += a2 * b2;
        }
      }
      return col;
    }

    // --------------------------------------------------------------- cutting
    splitCell(cell, axis) {
      const c = cell.c;
      let a, b;
      if (axis === "x") {
        const m = (cell.x0 + cell.x1) >> 1;
        const bot = this.born(m, cell.y0, c[0], c[1]), top = this.born(m, cell.y1, c[2], c[3]);
        a = new RCell(cell.x0, cell.y0, m, cell.y1, [c[0], bot, c[2], top]);
        b = new RCell(m, cell.y0, cell.x1, cell.y1, [bot, c[1], top, c[3]]);
        for (const i of cell.idx) (this.px[i] < m ? a : b).idx.push(i);
      } else {
        const m = (cell.y0 + cell.y1) >> 1;
        const left = this.born(cell.x0, m, c[0], c[2]), right = this.born(cell.x1, m, c[1], c[3]);
        a = new RCell(cell.x0, cell.y0, cell.x1, m, [c[0], c[1], left, right]);
        b = new RCell(cell.x0, m, cell.x1, cell.y1, [left, right, c[2], c[3]]);
        for (const i of cell.idx) (this.py[i] < m ? a : b).idx.push(i);
      }
      cell.kids = [a, b];
      const at = this.leaves.indexOf(cell);
      this.leaves.splice(at, 1, a, b);
      return [a, b];
    }

    // cut until (ix, iy) is a corner of every leaf cell that touches it
    makeCorner(ix, iy) {
      for (let guard = 0; guard < 400; ++guard) {
        let target = null, axis = null;
        for (const cell of this.leaves) {
          if ((ix === cell.x0 || ix === cell.x1) && iy > cell.y0 && iy < cell.y1) { target = cell; axis = "y"; break; }
          if ((iy === cell.y0 || iy === cell.y1) && ix > cell.x0 && ix < cell.x1) { target = cell; axis = "x"; break; }
        }
        if (!target) return true;
        this.splitCell(target, axis);
      }
      return false;
    }

    // --------------------------------------------------------------- one greedy step
    // The ranking uses the columns above. The winner is then cut in, which does not move the
    // surface, and its surplus is fitted against the exact column of the geometry that results, so
    // the applied step is a least-squares step and the squared error cannot rise.
    step() {
      this.steps++;
      this.buildIndex();
      const col = this.scratch || (this.scratch = { val: new Float64Array(this.n), hit: new Int32Array(this.n), k: 0 });
      const floor = Math.max(3, 2 * this.minPoints);
      let best = null, bestGain = 0;
      const consider = (v, mid) => {
        // a cut that would leave a hair-thin cell is not offered: the same limit the triangles use
        if (mid && this.maxAspect > 0 && !v) {
          let ok = false;
          for (const h of mid.hits) {
            const a = h.along === "x" ? h.half / h.span : h.span / h.half;
            if (Math.max(a, 1 / a) <= this.maxAspect) { ok = true; break; }
          }
          if (!ok) return;
        }
        this.column(v, mid, col);
        if (col.k < floor) return;
        let num = 0, den = 0;
        for (let t = 0; t < col.k; ++t) { const i = col.hit[t], x = col.val[i]; num += x * (this.Y[i] - this.pred[i]); den += x * x; }
        if (!(den > 1e-12)) return;
        const gain = (num * num) / den;
        if (gain > bestGain) { bestGain = gain; best = v || mid; }
      };
      for (const v of this.ix_where.values()) {
        if (!v.p0 && !v.free) continue;                       // a root that was never freed
        consider(v, this.ix_byMid.get(this.key(v.ix, v.iy)));
      }
      for (const mid of this.ix_byMid.values()) {
        const v = this.verts.get(this.key(mid.ix, mid.iy));
        if (v && this.ix_where.has(v.id)) continue;           // already scored above
        consider(v || null, mid);
      }
      if (!best) return "none";
      const ix = best.ix, iy = best.iy;
      const had = this.verts.get(this.key(ix, iy));
      const wasFree = !!(had && had.free);
      if (!this.makeCorner(ix, iy)) return "none";
      this.refresh();                                          // the cut moved points into new cells
      this.buildIndex();
      const v = this.verts.get(this.key(ix, iy));
      if (!v) return "none";
      this.column(v, null, col);
      let num = 0, den = 0;
      for (let t = 0; t < col.k; ++t) { const i = col.hit[t], x = col.val[i]; num += x * (this.Y[i] - this.pred[i]); den += x * x; }
      if (!(den > 1e-12)) return "none";
      v.free = true;
      v.surplus += num / den;
      this.refresh();
      return wasFree ? "refit" : "split";
    }

    // --------------------------------------------------------------- reporting
    get cells() { return this.leaves.length; }
    freeCount() { let k = 0; for (const v of this.verts.values()) if (v.free) ++k; return k; }
    sse() { let s = 0; for (let i = 0; i < this.n; ++i) { const r = this.Y[i] - this.pred[i]; s += r * r; } return s; }
    edges() {                       // the leaf outlines, each shared segment drawn once
      const seen = new Set(), out = [];
      for (const c of this.leaves)
        for (const e of [[c.x0, c.y0, c.x1, c.y0], [c.x0, c.y1, c.x1, c.y1],
                         [c.x0, c.y0, c.x0, c.y1], [c.x1, c.y0, c.x1, c.y1]]) {
          const k = e.join(",");
          if (seen.has(k)) continue;
          seen.add(k); out.push(e);
        }
      return out;
    }
  }

  const api = { RectMesh, reset: () => { SEQ = 0; }, S };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RM = api;
})(typeof self !== "undefined" ? self : this);
