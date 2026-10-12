// The fitted mesh, one depth at a time, for a slider that walks it from coarse to fine.
// A vertex's depth is the number of halvings of the square that reach it (level() below; the engine's own depth). At
// depth d the mesh is the one made of the fit's vertices of depth d or less: built here from the square's two triangles
// by bisecting hypotenuses until each of those vertices is a corner, with the extra cuts that keep every vertex a corner
// of every triangle touching it. Each vertex carries its fitted height, so the mesh at depth d is also a surface: the
// fit with only its corrections down to that depth (a finer vertex's own correction is nothing at a coarser vertex, so
// reading the fitted heights at the coarse vertices is the same as leaving the finer corrections out; the one exception
// is a vertex the engine also gave a finer role, whose height here includes it). The last stop is the engine's own mesh.
// Rectangular fits are not rebuilt: their lines and surface stay the final ones, and only the dots follow the slider.
window.MeshRounds = (function () {
  "use strict";
  const G = 1 << 16, key = (p) => p[0] * 131072 + p[1];
  let HT = null;   // heights of the fit's vertices by position, while a mesh is being built
  const mid = (a, b) => { const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], h = HT && HT.get(key(m)); m[2] = h !== undefined && h !== null ? h : (a[2] + b[2]) / 2; return m; };
  const split = (T) => { const m = mid(T[1], T[2]); return [[m, T[0], T[1]], [m, T[2], T[0]]]; };   // T = [right-angle corner, a, b]
  function inside(T, q) {
    const [[x1, y1], [x2, y2], [x3, y3]] = T, d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    const a = ((y2 - y3) * (q[0] - x3) + (x3 - x2) * (q[1] - y3)) / d, b = ((y3 - y1) * (q[0] - x3) + (x1 - x3) * (q[1] - y3)) / d;
    return a >= 0 && b >= 0 && a + b <= 1;
  }
  const mkey = (a, b) => (a[0] + b[0]) / 2 * 131072 + (a[1] + b[1]) / 2;
  function close(ts) {   // cut any triangle that has another's corner in the middle of one of its sides, until none has
    for (;;) {
      const vs = new Set(); for (const T of ts) for (const p of T) vs.add(key(p));
      let changed = false; const out = [];
      for (const T of ts) { if (vs.has(mkey(T[0], T[1])) || vs.has(mkey(T[0], T[2])) || vs.has(mkey(T[1], T[2]))) { out.push(...split(T)); changed = true; } else out.push(T); }
      ts = out; if (!changed) return ts;
    }
  }
  function insert(ts, v) {
    const kv = key(v);
    for (let guard = 0; guard < 40; ++guard) {
      if (ts.some((T) => key(T[0]) === kv || key(T[1]) === kv || key(T[2]) === kv)) break;
      let hit = ts.filter((T) => mkey(T[1], T[2]) === kv);
      if (!hit.length) { const T = ts.find((U) => inside(U, v)); if (!T) break; hit = [T]; }
      const set = new Set(hit); ts = ts.flatMap((T) => (set.has(T) ? split(T) : [T]));
    }
    return close(ts);
  }
  // the depth of a point of the unit square: the fewest halvings that reach it (0 the corners; larger is finer)
  function level(x, y) { for (let L = 0; L <= 16; ++L) { const s = 2 ** L, a = x * s, b = y * s; if (Math.abs(a - Math.round(a)) < 1e-6 && Math.abs(b - Math.round(b)) < 1e-6) return L; } return 16; }
  // depths(fit): [{ depth, lines, tri }], coarse to fine. lines: Float32Array, six numbers a triangle (its corners);
  // tri: Float64Array, nine a triangle (x y height for each corner), the surface at that depth. Both are null where the
  // fit's own final mesh is the one to draw (the last stop, and every stop of a rectangular fit).
  function depths(fit) {
    if (fit._depths) return fit._depths;
    const lv = fit.verts.map((v) => level(v.x, v.y)), top = Math.max(0, ...lv), out = [];
    if (fit.engine === "rect") { for (let d = 0; d <= top; ++d) out.push({ depth: d, lines: null, tri: null }); return (fit._depths = out); }
    // heights from the fit's own triangles: the surface's value at each vertex (the vertex list can hold two entries at a
    // place the engine gave two roles)
    HT = new Map(); for (let i = 0; i < fit.tri.length; i += 9) for (let k = 0; k < 3; ++k) HT.set(key([Math.round(fit.tri[i + 3 * k] * G), Math.round(fit.tri[i + 3 * k + 1] * G)]), fit.tri[i + 3 * k + 2]);
    const c = (x, y) => [x, y, HT.get(key([x, y])) || 0];
    let ts = [[c(G, 0), c(0, 0), c(G, G)], [c(0, G), c(G, G), c(0, 0)]];
    for (let d = 0; d <= top; ++d) {
      for (let i = 0; i < fit.verts.length; ++i) if (lv[i] === d) ts = insert(ts, [Math.round(fit.verts[i].x * G), Math.round(fit.verts[i].y * G)]);
      const lines = new Float32Array(ts.length * 6), tri = new Float64Array(ts.length * 9);
      ts.forEach((T, i) => { for (let k = 0; k < 3; ++k) { lines[6 * i + 2 * k] = tri[9 * i + 3 * k] = T[k][0] / G; lines[6 * i + 2 * k + 1] = tri[9 * i + 3 * k + 1] = T[k][1] / G; tri[9 * i + 3 * k + 2] = T[k][2]; } });
      out.push({ depth: d, lines, tri });
    }
    HT = null; out[out.length - 1].lines = out[out.length - 1].tri = null;
    return (fit._depths = out);
  }
  return { depths, level };
})();
