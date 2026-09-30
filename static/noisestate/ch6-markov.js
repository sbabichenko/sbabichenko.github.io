/* Chapter 6's finite-state equilibrium. No lag-window truncation.
 * The trader's HJB has three coefficients; the transparent maker has one,
 * the opaque maker three plus the equilibrium quote loading. See ch6-markov.md.
 * Pure module: callable from a worker and from Node's numerical checks.
 */
(function (root) {
  "use strict";
  const maxabs = (v) => Math.max(...v.map(Math.abs));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const zeros = (n, m = n) => Array.from({ length: n }, () => Array(m).fill(0));
  const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  const mul = (a, b) => a.map(row => b[0].map((_, j) => row.reduce((s, x, k) => s + x * b[k][j], 0)));
  const mv = (a, v) => a.map(row => dot(row, v));
  function linear(a, b) {
    const n = b.length, M = a.map((row, i) => [...row, b[i]]);
    for (let j = 0; j < n; ++j) {
      let p = j;
      for (let i = j + 1; i < n; ++i) if (Math.abs(M[i][j]) > Math.abs(M[p][j])) p = i;
      if (!(Math.abs(M[p][j]) > 1e-15)) throw Error("Singular equilibrium Jacobian");
      [M[p], M[j]] = [M[j], M[p]];
      for (let i = j + 1; i < n; ++i) {
        const w = M[i][j] / M[j][j];
        for (let k = j; k <= n; ++k) M[i][k] -= w * M[j][k];
      }
    }
    const x = Array(n).fill(0);
    for (let i = n - 1; i >= 0; --i) {
      let v = M[i][n];
      for (let j = i + 1; j < n; ++j) v -= M[i][j] * x[j];
      x[i] = v / M[i][i];
    }
    return x;
  }
  function newton(f, start) {
    let z = [...start];
    for (let it = 0; it < 50; ++it) {
      const e = f(z), error = maxabs(e);
      if (error < 2e-12) return { z, error, iterations: it };
      const J = zeros(z.length);
      for (let j = 0; j < z.length; ++j) {
        const h = 1e-5 * Math.max(1, Math.abs(z[j])), plus = [...z], minus = [...z];
        plus[j] += h; minus[j] -= h;
        const p = f(plus), m = f(minus);
        for (let i = 0; i < z.length; ++i) J[i][j] = (p[i] - m[i]) / (2 * h);
      }
      const step = linear(J, e.map(x => -x));
      let accepted = false;
      for (let w = 1; w >= 1 / 4096; w /= 2) {
        const next = z.map((x, i) => x + w * step[i]);
        if (maxabs(f(next)) < error) { z = next; accepted = true; break; }
      }
      if (!accepted) throw Error("The finite-state equilibrium iteration stalled");
    }
    throw Error("The finite-state equilibrium did not converge");
  }
  // Scaling and squaring with a Taylor series on a matrix whose row norm is <= 1/2.
  // Small matrices (up to 8x8) only; this also handles repeated and zero rates.
  function expm(a, t) {
    const n = a.length, norm = Math.max(...a.map(row => row.reduce((s, x) => s + Math.abs(x * t), 0)));
    const squarings = Math.max(0, Math.ceil(Math.log2(Math.max(norm * 2, 1))));
    const b = a.map(row => row.map(x => x * t / 2 ** squarings));
    let out = eye(n), term = eye(n);
    for (let k = 1; k <= 24; ++k) {
      term = mul(term, b).map(row => row.map(x => x / k));
      out = out.map((row, i) => row.map((x, j) => x + term[i][j]));
      if (maxabs(term.flat()) < 1e-17) break;
    }
    for (let k = 0; k < squarings; ++k) out = mul(out, out);
    return out;
  }
  function solve(market, params) {
    const { eps: e, gamma: g, rho: r, sigma_Z: s } = params;
    if (![e, g, r, s].every(Number.isFinite) || Math.min(e, r, s) <= 0 || g < 0 || !["transparent", "opaque"].includes(market))
      throw Error("Chapter 6 needs positive trading cost, discount and noise volatility, and nonnegative inventory cost");
    const l = 1 / s;
    function terms(z, gamma) {
      const [a, b, c] = z, gq = -(l * b + c) / (2 * e);
      const pq = market === "transparent" ? e * gq - z[3] / 2 : z[6];
      const n1 = 1 - l * a - b, n2 = -pq - l * b - c;
      const beta = n1 / (2 * e), delta = n2 / (2 * e);
      const eq = [r * a / 2 - n1 * n1 / (4 * e), r * b - n1 * n2 / (2 * e) - l * delta * a,
        r * c / 2 - n2 * n2 / (4 * e) - l * delta * b];
      if (market === "transparent") {
        const u = z[3];
        eq.push(r * u - 2 * gamma + e * gq * gq + gq * u + u * u / (4 * e));
        return { eq, beta, delta, pq, gain: [0], drift: [[-delta]], initial: [1 / (2 * e)] };
      }
      const cn = (beta * l + delta) / (l - pq), cx = beta * pq + delta;
      const kx = l * beta * pq / (l - pq), kn = l * l * beta / (l - pq) ** 2;
      const A = [[-delta, -cx], [0, kx]], B = [-cn, kn];
      const R = [[-pq * delta + gamma, -pq * cx / 2], [-pq * cx / 2, 0]];
      const N = [(delta - cn * pq) / 2, cx / 2], U = [[z[3], z[4]], [z[4], z[5]]];
      const UB = mv(U, B), v = N.map((x, i) => x + UB[i]), gain = v.map(x => -x / cn);
      const UA = mul(U, A), E = U.map((row, i) => row.map((x, j) => r * x - R[i][j] - UA[i][j] - UA[j][i] + v[i] * v[j] / cn));
      eq.push(E[0][0], E[0][1], E[1][1], gain[0]);
      return { eq, beta, delta, pq, gain, initial: [cn, -kn],
        drift: A.map((row, i) => row.map((x, j) => x + B[i] * gain[j])) };
    }
    const a0 = 1 / (l + e * r + Math.sqrt(e * r * (2 * l + e * r)));
    let z, iterations = 0;
    function traderAt(d) {
      const b = d / (r + d), h = r / (r + d);
      const c = (2 * e * d * d + 2 * l * d * b) / r;
      const a = h * h / (l * h + e * r + Math.sqrt(e * r * (2 * l * h + e * r)));
      return { a, b, c, beta: Math.sqrt(r * a / (2 * e)), p: -2 * e * d - l * b - c };
    }
    function bracket(hi, value) {
      let lo = 0;
      if (!g) return 0;
      for (; iterations < 64; ++iterations) {
        const y = (lo + hi) / 2;
        if (y === lo || y === hi) break;
        if (value(y) < 2) lo = y; else hi = y;
      }
      return g * (lo + hi) / 2;
    }
    if (market === "transparent") {
      // Eliminate (a,b,c,u) to a strictly increasing scalar equation for delta.
      // Bracket delta/g, so very small positive g retains relative accuracy.
      const d = bracket(2 / (4 * e * r + l), y => {
        const d = g * y;
        return 4 * e * r * y + 6 * e * g * y * y + l * y * (r + 2 * d) / (r + d);
      });
      const { a, b, c } = traderAt(d);
      z = [a, b, c, 4 * e * d + l * b + c];
    } else {
      const d = bracket(2 / (2 * e * r + l), y => {
        const d = g * y, { beta, p } = traderAt(d);
        const T = (l * beta * (r + 2 * d) + d * (r + d)) / ((l - p) * (r + d) - l * beta * p);
        return y * (r + 2 * d) / T + 2 * e * r * y + l * r * y / (r + d)
          + 2 * e * g * y * y + 2 * l * g * y * y / (r + d);
      });
      const { a, b, c, beta, p } = traderAt(d);
      const cn = (l * beta + d) / (l - p), cx = beta * p + d;
      const kx = l * beta * p / (l - p), kn = l * l * beta / (l - p) ** 2;
      const u11 = (g - p * d) / (r + 2 * d), u12 = -cx * (p / 2 + u11) / (r + d - kx);
      const v0 = cx / 2 - cn * u12, A = kn * kn / cn;
      const B = r - 2 * kx + 2 * v0 * kn / cn, C = 2 * cx * u12 + v0 * v0 / cn;
      const u22 = -2 * C / (B + Math.sqrt(B * B - 4 * A * C));
      z = [a, b, c, u11, u12, u22, p];
      // Keep continuation as a fallback outside the validated parameter range.
      if (!(maxabs(terms(z, g).eq) < 1e-9)) {
        z = [a0, 0, 0, 0, 0, 0, 0]; iterations = 0;
        const steps = Math.max(1, Math.min(64, Math.ceil(g / .01)));
        for (let k = 1; k <= steps; ++k) {
          const ans = newton(x => terms(x, g * k / steps).eq, z);
          z = ans.z; iterations += ans.iterations;
        }
      }
    }
    const ans = terms(z, g), residual = maxabs(ans.eq);
    const A = ans.drift, trace = A.length === 1 ? A[0][0] : A[0][0] + A[1][1];
    const determinant = A.length === 1 ? -A[0][0] : A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (!(residual < 1e-9 && ans.beta > 0) || (g > 0 && !(ans.delta > 0 && ans.pq < 0 && trace < 0 && determinant > 0)))
      throw Error("The finite-state root is not a stabilizing equilibrium");
    const gross = s - ans.pq * s * s / 2, trading = e * (ans.beta * s + ans.delta * s * s / 2);
    const inventory = g ? g * s * s / (2 * ans.delta) : 0;
    return { ...ans, root: z, params: { ...params }, market, residual, iterations,
      costs: { market_maker: gross + inventory, trader: trading - gross },
      accounts: { gross, trading, inventory, noise: (l - ans.pq) * s * s },
      rates: { information: l * ans.beta, inventory: ans.delta } };
  }
  function sample(eq, ages) {
    const { beta: b, delta: d, pq: p, params: { sigma_Z: s } } = eq, l = 1 / s;
    const F = [[0, 0, 0, 0], [1, -1, 0, 0], [0, l * b, -l * b, 0], [0, -b, b, -d]];
    const B = [[1, 0, 0], [0, 0, 1], [0, l * s, 0], [0, -s, 0]];
    const C = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, p], [0, b, -b, d]];
    const names = ["V", "Q", "P", "D"], shocks = ["wV", "wZ", "wY"];
    const kernels = Object.fromEntries(names.map(n => [n, Object.fromEntries(shocks.map(w => [w, []]))]));
    const deviation = { Q: [], P: [], D: [] };
    for (const t of ages) {
      const K = mul(mul(C, expm(F, t)), B);
      names.forEach((n, i) => shocks.forEach((w, j) => kernels[n][w].push(K[i][j])));
      const state = mv(expm(eq.drift, t), eq.initial), q = state[0];
      deviation.Q.push(q); deviation.P.push(p * q - dot(state, eq.gain)); deviation.D.push(-dot(state, eq.drift[0]));
    }
    return { kernels, deviation };
  }
  const transpose = a => a[0].map((_, j) => a.map(row => row[j]));
  function cholesky(a) {
    const n = a.length, L = zeros(n), scale = Math.max(1, maxabs(a.flat()));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let v = (a[i][j] + a[j][i]) / 2;
      for (let k = 0; k < j; k++) v -= L[i][k] * L[j][k];
      if (i === j) {
        if (v < -1e-12 * scale) throw Error("Invalid sample-path covariance");
        L[i][j] = Math.sqrt(Math.max(0, v));
      } else L[i][j] = L[j][j] ? v / L[j][j] : 0;
    }
    return L;
  }
  function paths(eq, h = .1, cells = 400) {
    // State: filtering error E=V-Vhat1, information edge X=Vhat1-Vhat0,
    // inventory Q, and the fundamental V (anchored at V(0)=0).
    const { beta: b, delta: d, pq: p, params: { sigma_Z: s, gamma: g } } = eq;
    const F = [[-1, 0, 0, 0], [1, -b/s, 0, 0], [0, -b, -d, 0], [0, 0, 0, 0]];
    const B = [[1, 0, -1], [0, -1, 1], [0, -s, 0], [1, 0, 0]], G = mul(B, transpose(B));
    // Van Loan's block exponential integrates the continuous shocks exactly
    // between displayed dates, including their correlations. No Euler step.
    const block = zeros(8);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      block[i][j] = F[i][j]; block[i][j+4] = G[i][j]; block[i+4][j+4] = -F[j][i];
    }
    const E = expm(block, h), A = E.slice(0, 4).map(row => row.slice(0, 4));
    const covariance = mul(E.slice(0, 4).map(row => row.slice(4)), transpose(A));
    const variances = [1, s/b, g ? s*s/(2*d) : 0, 0];
    return { kind: "state-space", h, cells, transition: A, noise: cholesky(covariance),
      initial: variances.map((v, i) => variances.map((_, j) => i === j ? Math.sqrt(v) : 0)),
      outputs: { V: [0, 0, 0, 1], Q: [0, 0, 1, 0], P: [-1, -1, p, 1], D: [0, b, d, 0] } };
  }
  const baseModel = {"shocks":["wV","wZ","wY"],"states":{"V":{"drift":{},"noise":{"wV":1}},"Q":{"drift":{"D":-1},"noise":{"wZ":"-sigma_Z"}}},"agents":{"market_maker":{"controls":["P"],"signals":{"flow":{"drift":{"D":1},"noise":{"wZ":"sigma_Z"}}},"loss":[[1,"V","D"],[-1,"P","D"],["gamma","Q","Q"]]},"trader":{"controls":["D"],"monitors":["market_maker"],"instant":["P"],"signals":{"y":{"drift":{"V":1,"P":-1},"noise":{"wY":1}},"flow":{"drift":{},"noise":{"wZ":"sigma_Z"}}},"loss":[[-1,"V","D"],[1,"P","D"],["eps","D","D"]]}},"horizon":{"kind":"stationary","discount":"rho"}};
  function sorted(value) {
    if (Array.isArray(value)) return value.map(sorted);
    if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(k => [k, sorted(value[k])]));
    return value;
  }
  function checkModel(model, market) {
    const actual = JSON.parse(JSON.stringify(model)), expected = JSON.parse(JSON.stringify(baseModel));
    for (const k of ["name", "params", "numerics"]) delete actual[k];
    if (actual.horizon) delete actual.horizon.window;
    if (market === "opaque") {
      const t = expected.agents.trader; delete t.monitors;
      t.signals = { y: t.signals.y, quote: { level: "P" } };
    }
    if (JSON.stringify(sorted(actual)) !== JSON.stringify(sorted(expected)))
      throw Error("This reference applies to the Chapter 6 preset. Use the general solver for modified equations.");
  }
  function payload(model, request = {}) {
    checkModel(model, "transparent");
    if (request.compare) checkModel(request.compare.model, "opaque");
    const t0 = performance.now(), params = model.params || {};
    const transparent = solve("transparent", params), opaque = request.compare ? solve("opaque", request.compare.model.params) : null;
    const rates = [transparent, opaque].filter(Boolean).map(e => e.delta).filter(d => d > 0);
    const extent = rates.length ? Math.max(20, Math.min(160, 5 / Math.min(...rates))) : 40;
    const ages = Array.from({ length: 201 }, (_, i) => 2 * Math.expm1(Math.log1p(extent / 2) * i / 200));
    function result(eq, spec) {
      const data = sample(eq, ages), seconds = (performance.now() - t0) / 1000;
      const maker = { name: "market_maker", controls: ["P"] }, trader = { name: "trader", controls: ["D"] };
      return {
        ok: true, converged: true, engine: "ch6-markov", version: "Chapter 6 finite-state reference, 2026-09-30",
        name: spec.name, model: spec, kind: "stationary", has_means: false, cost_kind: "stationary flow loss per unit time",
        costs: eq.costs, cost_parts: Object.fromEntries(Object.entries(eq.costs).map(([a, v]) => [a, { variance: v, mean: 0, constant: 0 }])),
        names: ["V", "Q", "P", "D"], states: ["V", "Q"], channels: ["wV", "wZ", "wY"], shocks: ["wV", "wZ", "wY"],
        agents: [maker, trader], definitions: [], params_used: eq.params,
        samples: { age: ages, kernels: data.kernels, foc: {} },
        paths: paths(eq),
        deviation: { continuation: "blip", origins: { market_maker: {
          privy: eq.market === "transparent" ? ["market_maker", "trader"] : ["market_maker"],
          controls: { P: { samples: data.deviation } }
        } } },
        reference: { beta: eq.beta, delta: eq.delta, price_inventory: eq.pq, equations: eq.root.length,
          accounts: eq.accounts, root: eq.root, plot_extent: extent },
        checks: [{ name: "equilibrium equations", ok: true, value: eq.residual, threshold: 1e-9,
          meaning: "Residual of the trader and market maker's coupled Hamilton–Jacobi equations." },
          { name: "stabilizing branch", ok: true, value: eq.delta, threshold: null,
            meaning: eq.params.gamma ? "Positive trading intensity, mean-reverting inventory and stabilizing market-maker feedback."
              : "The competitive equilibrium. With no inventory penalty, inventory is a random walk." }],
        flags: [], warnings: [], residual: eq.residual, evaluations: eq.iterations, seconds,
        message: `${eq.root.length} coupled equilibrium equations checked against the full system`,
      };
    }
    const out = result(transparent, model);
    if (opaque) out.compare = { ...result(opaque, request.compare.model), label: request.compare.label || "Opaque" };
    out.seconds = (performance.now() - t0) / 1000;
    return out;
  }
  const api = { solve, sample, paths, payload };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Ch6Markov = api;
})(typeof self !== "undefined" ? self : globalThis);
