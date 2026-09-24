// Decision Mesh: a JavaScript port of DecisionMesh.py (github.com/sbabichenko/Decision-Mesh).
// A piecewise-linear fit on a triangular mesh that refines like a decision tree: every step either activates the
// midpoint of an edge (splitting the one or two triangles on it through the opposite vertices) or re-fits the
// height of an active vertex, whichever reduces the squared error most; with probability `random` it instead
// splits the longest edge of a triangle drawn with weight area x points.  Same objects and steps as the Python;
// the only departures are storage (index arrays and per-face sufficient statistics) and tie-breaking in the queue.
(function (root) {
  "use strict";

  let SEQ = 0;

  class Vertex {
    constructor(mesh, x, y, active = false, parentEdge = null) {
      this.id = SEQ++;
      this.x = x; this.y = y; this.mesh = mesh;
      mesh.vertices.add(this);
      this.active = active;
      this.disqualified = false;
      this.parentEdge = parentEdge;
      this.height = parentEdge ? (parentEdge.v0.height + parentEdge.v1.height) / 2 : 0.0;
      this.edges = new Set();
      this.neighbors = new Set();
      this.newHeight = this.height; this.lossReduction = 0; this.affected = new Set();
    }
    addEdge(e) { this.edges.add(e); this.neighbors.add(e.other(this)); }
    removeEdge(e) { this.edges.delete(e); this.neighbors.delete(e.other(this)); }

    activate() {
      if (this.active) return;
      this.active = true;
      this.height = this.newHeight;
      this.parentEdge.split();
      for (const v of this.affected) v.updateInfo();
      this.lossReduction = 0;
      this.mesh.heap.set(this, 0);
    }
    updateInfo() {
      const r = this.locRegress();
      this.newHeight = r[0]; this.lossReduction = r[1]; this.affected = r[2];
      if (!this.disqualified) this.mesh.heap.set(this, -this.lossReduction);
    }
    updateHeight() {
      this.height = this.newHeight;
      for (const v of this.affected) v.updateInfo();
      if (this.mesh.refresh) {
        // the candidate midpoints on this vertex's triangles were fitted against its old height: refit them
        for (const f of this.getFaces())
          for (const e of f.edges) {
            const m = e.midpoint;
            if (m && !m.active) { m.height = (e.v0.height + e.v1.height) / 2; m.updateInfo(); }
          }
      }
      this.lossReduction = 0;
      this.mesh.heap.set(this, 0);
    }
    getFaces() {
      const faces = new Set();
      for (const e of this.edges)
        for (const s of ["+", "-"]) {
          const f = e.faces[s];
          if (f && f.vertices.includes(this)) faces.add(f);
        }
      return faces;
    }
    getSimFaces() {
      const faces = new Set(), pe = this.parentEdge;
      for (const s of ["+", "-"]) {
        const f = pe.faces[s];
        if (!f) continue;
        const sd = f.subdiv[f.edges.indexOf(pe)];
        faces.add(sd["+"]); faces.add(sd["-"]);
      }
      return faces;
    }
    // the best height of this vertex with every other height fixed, and the squared error it saves
    locRegress() {
      const faces = this.active ? this.getFaces() : this.getSimFaces();
      const neighbors = new Set();
      for (const f of faces) for (const v of f.vertices) if (v !== this) neighbors.add(v);
      if (!faces.size) return [this.height, 0, neighbors];
      const Y = this.mesh.Y;
      let xx = 0, xr = 0, rr = 0;
      for (const f of faces) {
        const n = f.idx.length;
        if (!n) continue;
        const W = f.weights(), j = f.vertices.indexOf(this);
        const h0 = j === 0 ? 0 : f.vertices[0].height, h1 = j === 1 ? 0 : f.vertices[1].height, h2 = j === 2 ? 0 : f.vertices[2].height;
        for (let k = 0; k < n; ++k) {
          const w0 = W[3 * k], w1 = W[3 * k + 1], w2 = W[3 * k + 2];
          const r = Y[f.idx[k]] - (w0 * h0 + w1 * h1 + w2 * h2);
          const x = j < 0 ? 0 : W[3 * k + j];
          xx += x * x; xr += x * r; rr += r * r;
        }
      }
      const b0 = this.height;
      const orig = rr - 2 * b0 * xr + b0 * b0 * xx;
      let beta = b0, post = rr;
      if (xx > 0) { beta = xr / xx; post = rr - (xr * xr) / xx; }
      return [beta, orig - post, neighbors];
    }
  }

  class Edge {
    constructor(mesh, v0, v1, active) {
      this.id = SEQ++;
      this.mesh = mesh; this.v0 = v0; this.v1 = v1;
      this.faces = { "+": null, "-": null };
      this.opp = { "+": null, "-": null };
      this.disqualifying = new Set();
      const dx = v1.x - v0.x, dy = v1.y - v0.y;
      this.length = Math.hypot(dx, dy);
      this.nx = -dy / this.length; this.ny = dx / this.length;
      this.intercept = this.nx * v0.x + this.ny * v0.y;
      this.midpoint = null;
      this.sub = { "0": null, "1": null, "+": null, "-": null };
      this.active = false;
      if (active) this.activate();
    }
    other(v) { return v === this.v0 ? this.v1 : v === this.v1 ? this.v0 : null; }
    test(v) { return v.x * this.nx + v.y * this.ny - this.intercept; }
    activate() {
      if (this.active) return;
      this.active = true;
      this.mesh.activeEdges.add(this);
      if (!this.midpoint) this.midpoint = new Vertex(this.mesh, (this.v0.x + this.v1.x) / 2, (this.v0.y + this.v1.y) / 2, false, this);
      this.v0.addEdge(this); this.v1.addEdge(this);
      this.sub = { "0": new Edge(this.mesh, this.v0, this.midpoint, false), "1": new Edge(this.mesh, this.midpoint, this.v1, false), "+": null, "-": null };
    }
    split() {
      if (this.faces["+"]) this.faces["+"].split(this);
      if (this.faces["-"]) this.faces["-"].split(this);
      this.sub["0"].activate(); this.sub["1"].activate();
      this.deactivate();
    }
    deactivate() {
      if (!this.active) return;
      this.active = false;
      this.mesh.activeEdges.delete(this);
      this.v0.removeEdge(this); this.v1.removeEdge(this);
    }
    // attach a face and precompute its split through this edge's midpoint
    addFace(face) {
      const mesh = this.mesh;
      const opposing = face.vertices[face.edges.indexOf(this)];
      const type = this.test(opposing) > 0 ? "+" : "-";
      this.faces[type] = face; this.opp[type] = opposing;
      const chord = this.sub[type] = new Edge(mesh, this.midpoint, opposing, false);
      const edge0 = face.edges[face.vertices.indexOf(this.v1)];
      const edge1 = face.edges[face.vertices.indexOf(this.v0)];
      const X = mesh.X, idx = face.idx, n = idx.length;
      const a = new Int32Array(n), b = new Int32Array(n);
      let na = 0, nb = 0;
      for (let k = 0; k < n; ++k) {
        const i = idx[k];
        if (X[2 * i] * chord.nx + X[2 * i + 1] * chord.ny >= chord.intercept) a[na++] = i; else b[nb++] = i;
      }
      const side = a.subarray(0, na), rest = b.subarray(0, nb);
      let face0, face1;
      if (type === "+") {
        face0 = new Face(mesh, chord, edge0, this.sub["0"], side, false, face.path + "+");
        face1 = new Face(mesh, chord, edge1, this.sub["1"], rest, false, face.path + "-");
        face.subdiv[face.edges.indexOf(this)] = { e: chord, "+": face0, "-": face1 };
      } else {
        face0 = new Face(mesh, chord, edge0, this.sub["0"], rest, false, face.path + "-");
        face1 = new Face(mesh, chord, edge1, this.sub["1"], side, false, face.path + "+");
        face.subdiv[face.edges.indexOf(this)] = { e: chord, "+": face1, "-": face0 };
      }
      if (Math.max(face0.aspectRatio(), face1.aspectRatio()) >= mesh.maxAspectRatio ||
          Math.min(face0.idx.length, face1.idx.length) < mesh.minPoints) {
        this.disqualifying.add(face);
        if (!this.midpoint.disqualified) { this.midpoint.disqualified = true; mesh.heap.delete(this.midpoint); }
      }
      this.midpoint.updateInfo();
    }
    removeFace(face) {
      if (this.faces["+"] === face) this.faces["+"] = null;
      if (this.faces["-"] === face) this.faces["-"] = null;
      this.disqualifying.delete(face);
      if (this.disqualifying.size === 0) this.midpoint.disqualified = false;
    }
  }

  class Face {
    constructor(mesh, e0, e1, e2, idx, active = true, path = "") {
      this.id = SEQ++;
      this.mesh = mesh; this.path = path; this.idx = idx;
      this.edges = [e0, e1, e2];
      this.subdiv = [{ e: null, "+": null, "-": null }, { e: null, "+": null, "-": null }, { e: null, "+": null, "-": null }];
      let v1 = e0.v0, v2 = e0.v1, v0 = e2.other(v1);
      if (!v0) { [v1, v2] = [v2, v1]; v0 = e2.other(v1); }
      this.vertices = [v0, v1, v2];
      this._W = null;
      const det = (v0.x - v2.x) * (v1.y - v0.y) - (v0.x - v1.x) * (v2.y - v0.y);
      this.area = 0.5 * Math.abs(det);
      this.active = false;
      if (active) this.activate();
    }
    aspectRatio() {
      const L = this.edges.map((e) => e.length);
      return Math.max(...L) / Math.min(...L);
    }
    // barycentric weights of the face's points, (n, 3) row-major, computed when first needed
    weights() {
      if (this._W) return this._W;
      const [v0, v1, v2] = this.vertices, X = this.mesh.X, n = this.idx.length;
      const ax = v1.x - v0.x, ay = v1.y - v0.y, bx = v2.x - v0.x, by = v2.y - v0.y;
      const d00 = ax * ax + ay * ay, d01 = ax * bx + ay * by, d11 = bx * bx + by * by;
      const den = d00 * d11 - d01 * d01;
      const W = new Float64Array(3 * n);
      for (let k = 0; k < n; ++k) {
        const i = this.idx[k], px = X[2 * i] - v0.x, py = X[2 * i + 1] - v0.y;
        const d20 = px * ax + py * ay, d21 = px * bx + py * by;
        const u = (d11 * d20 - d01 * d21) / den, v = (d00 * d21 - d01 * d20) / den;
        W[3 * k] = 1 - u - v; W[3 * k + 1] = u; W[3 * k + 2] = v;
      }
      return (this._W = W);
    }
    split(edge) {
      const sd = this.subdiv[this.edges.indexOf(edge)];
      if (!sd.e) throw new Error("Edge has not been activated/split yet.");
      this.deactivate();
      sd["+"].activate(); sd["-"].activate();
    }
    deactivate() {
      if (!this.active) return;
      this.active = false;
      this.mesh.activeFaces.delete(this);
      for (const e of this.edges) e.removeFace(this);
      this._W = null;
      for (const sd of this.subdiv) for (const k of ["+", "-"]) if (sd[k] && !sd[k].active) sd[k]._W = null;
    }
    activate() {
      if (this.active) return;
      this.active = true;
      this.mesh.activeFaces.add(this);
      for (const e of this.edges) { e.activate(); e.addFace(this); }
    }
  }

  class DecisionMesh {
    // X: Float64Array of 2n coordinates (x0, y0, x1, y1, ...), Y: Float64Array of n values
    // minPoints (not in the Python; 0 keeps its behaviour): a split is disqualified, like a too-thin one, when either
    // new triangle would hold fewer data points, which keeps a vertex from being fitted to one or two points
    // refresh (not in the Python's default; false keeps its behaviour): after a vertex is refitted, refit the candidate
    // midpoints on its triangles as well, so that no step acts on a fit made against an old height
    constructor(X, Y, { maxAspectRatio = 5, minPoints = 0, refresh = false, rng = Math.random } = {}) {
      this.X = X; this.Y = Y; this.n = Y.length;
      this.maxAspectRatio = maxAspectRatio; this.minPoints = minPoints; this.refresh = refresh; this.rng = rng;
      this.vertices = new Set(); this.activeFaces = new Set(); this.activeEdges = new Set();
      this.heap = new Map();
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (let i = 0; i < this.n; ++i) {
        const x = X[2 * i], y = X[2 * i + 1];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x; if (y < ymin) ymin = y; if (y > ymax) ymax = y;
      }
      Object.assign(this, { xmin, xmax, ymin, ymax });
      const bl = new Vertex(this, xmin, ymin, true), tl = new Vertex(this, xmin, ymax, true);
      const br = new Vertex(this, xmax, ymin, true), tr = new Vertex(this, xmax, ymax, true);
      const left = new Edge(this, tl, bl, false), bottom = new Edge(this, br, bl, false);
      const right = new Edge(this, tr, br, false), top = new Edge(this, tr, tl, false);
      const diag = new Edge(this, bl, tr, true);
      const up = [], down = [];
      for (let i = 0; i < this.n; ++i) (X[2 * i] * diag.nx + X[2 * i + 1] * diag.ny >= diag.intercept ? up : down).push(i);
      this.topFace = new Face(this, diag, left, top, Int32Array.from(up), true, "+");
      this.bottomFace = new Face(this, diag, bottom, right, Int32Array.from(down), true, "-");
      for (const v of [...this.vertices]) v.updateInfo();
      this.steps = 0;
    }

    randomFace() {
      const faces = [...this.activeFaces];
      const w = faces.map((f) => Math.max(f.area, f.area > 0 ? 1e-12 : 0) * f.idx.length);
      const total = w.reduce((a, b) => a + b, 0);
      let u = this.rng() * (total > 0 ? total : faces.length);
      for (let i = 0; i < faces.length; ++i) { u -= total > 0 ? w[i] : 1; if (u < 0) return faces[i]; }
      return faces[faces.length - 1];
    }

    // one step; returns what it did
    step(random = 0) {
      let best;
      if (random > 0 && this.rng() < random) {
        const face = this.randomFace();
        const L = Math.max(...face.edges.map((e) => e.length));
        const cand = face.edges.filter((e) => Math.abs(e.length - L) <= 1e-12);
        best = cand[Math.floor(this.rng() * cand.length)].midpoint;
      } else {
        let bp = Infinity;
        for (const [v, p] of this.heap) if (p < bp) { bp = p; best = v; }
      }
      this.steps++;
      if (!best) return "none";
      if (best.active) { best.updateHeight(); return "refit"; }
      best.activate(); return "split";
    }

    activeVertices() { let k = 0; for (const v of this.vertices) if (v.active) ++k; return k; }

    sse() {
      let s = 0;
      for (const f of this.activeFaces) {
        const W = f.weights(), [a, b, c] = f.vertices.map((v) => v.height);
        for (let k = 0; k < f.idx.length; ++k) { const r = this.Y[f.idx[k]] - (W[3 * k] * a + W[3 * k + 1] * b + W[3 * k + 2] * c); s += r * r; }
      }
      return s;
    }
  }

  const api = { DecisionMesh, Vertex, Edge, Face, reset: () => { SEQ = 0; } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DM = api;
})(typeof self !== "undefined" ? self : this);
