"use strict";
const WORKER_URL = document.currentScript.dataset.worker || "worker.js";
// ---------------------------------------------------------------------------------------------
// Presets: a model file each, with sliders on some of its parameters.
const PRESETS = {
  ch1: {
    tab: "Tracking game",
    ch: "Ch. 1",
    title: "Two-player tracking game on a finite horizon",
    desc: `<p class="small muted" style="margin:0">Chapter 1. Two players push one shared state toward their own targets.
      Each sees the state only through its own noisy signal, so each must forecast what the other knows.
      The common shock moves the state; neither player observes it directly.</p>
      <div class="eq"><div class="tex" data-tex="dX = (D^1 + D^2)\\,dt + \\sigma\\,dW^0, \\qquad X_0 = 0">dX = (D<sup>1</sup> + D<sup>2</sup>) dt + &sigma; dW<sup>0</sup>, &nbsp; X<sub>0</sub> = 0</div>
      <div class="tex" data-tex="dY^i = \\sqrt{p_i}\\,X\\,dt + dW^i">dY<sup>i</sup> = &radic;p<sub>i</sub> X dt + dW<sup>i</sup></div>
      <div class="tex" data-tex="\\text{player } i \\text{ minimizes } \\mathbb{E}\\int_0^T \\big[(X - b_i)^2 + r_i\\,(D^i)^2\\big]\\,dt">player i minimizes E &int;<sub>0</sub><sup>T</sup> [ (X &minus; b<sub>i</sub>)<sup>2</sup> + r<sub>i</sub> (D<sup>i</sup>)<sup>2</sup> ] dt</div></div>
      <p class="small muted" style="margin:0">p<sub>i</sub> is signal precision. Reported costs include the constant b<sub>i</sub><sup>2</sup>T.</p>`,
    yaml: `name: ch1_tracking_with_targets
params: {p1: 9.0, p2: 9.0, r1: 0.1, r2: 0.1, b1: 1.0, b2: -1.0, sigma: 1.0, T: 1.0}
shocks: [w0, w1, w2]
states:
  X: {drift: {D1: 1.0, D2: 1.0}, noise: {w0: sigma}}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[1.0, X, X], ["-2*b1", X], [r1, D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
    loss: [[1.0, X, X], ["-2*b2", X], [r2, D2, D2]]
horizon: {kind: finite, T: T}
numerics: {nodes: 12}
`,
    sliders: [
      { key: "p1", label: "p₁ precision, player 1", min: 0.1, max: 100, log: true },
      { key: "p2", label: "p₂ precision, player 2", min: 0.1, max: 100, log: true },
      { key: "r1", label: "r₁ effort cost, player 1", min: 0.02, max: 2, log: true },
      { key: "r2", label: "r₂ effort cost, player 2", min: 0.02, max: 2, log: true },
      { key: "b1", label: "b₁ target, player 1", min: -2, max: 2, step: 0.1 },
      { key: "b2", label: "b₂ target, player 2", min: -2, max: 2, step: 0.1 },
      { key: "sigma", label: "σ common shock scale", min: 0.2, max: 3, step: 0.05 },
      { key: "T", label: "T horizon", min: 0.5, max: 3, step: 0.25 },
    ],
    nodes: { def: 12, options: [[8, "8: fastest, rough"], [10, "10: quick"], [12, "12: the default"], [16, "16: fine, passes the refinement check"]] },
    constCost: (p) => ({ player1: p.b1 * p.b1 * p.T, player2: p.b2 * p.b2 * p.T }),
    meansCaption: "Expected controls and state over time. With opposite targets the players pull in opposite directions; as [precision falls](#game=ch1&p1=0.1&p2=0.1) the mean controls approach the open-loop solution, and as [it rises](#game=ch1&p1=100&p2=100) they approach the full-information one.",
  },
  ch3: {
    tab: "Stationary tracking",
    ch: "Ch. 3",
    title: "Stationary two-player tracking game",
    desc: `<p class="small muted" style="margin:0">Chapter 3. The same tracking problem run forever, scored by average cost per unit time.
      Responses are kernels in shock age: how a unit shock of a given age still moves the state or a control.</p>
      <div class="eq"><div class="tex" data-tex="dX = (D^1 + D^2)\\,dt + dW^0">dX = (D<sup>1</sup> + D<sup>2</sup>) dt + dW<sup>0</sup></div>
      <div class="tex" data-tex="dY^i = \\sqrt{p_i}\\,X\\,dt + dW^i">dY<sup>i</sup> = &radic;p<sub>i</sub> X dt + dW<sup>i</sup></div>
      <div class="tex" data-tex="\\text{player } i \\text{ minimizes the average of } \\tfrac12 (X - b_i)^2 + \\tfrac12 r_i\\,(D^i)^2">player i minimizes the average of &frac12; (X &minus; b<sub>i</sub>)<sup>2</sup> + &frac12; r<sub>i</sub> (D<sup>i</sup>)<sup>2</sup></div></div>
      <p class="small muted" style="margin:0">b<sub>i</sub> is player i&rsquo;s target. Reported costs include the constant &frac12; b<sub>i</sub><sup>2</sup>.</p>`,
    yaml: `name: ch3_stationary_tracking
params: {p1: 3.0, p2: 10.0, r1: 1.0, r2: 1.0, b1: 1.0, b2: -1.0}
shocks: [w0, w1, w2]
states:
  X: {drift: {D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[0.5, X, X], ["-b1", X], ["0.5*r1", D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
    loss: [[0.5, X, X], ["-b2", X], ["0.5*r2", D2, D2]]
horizon: {kind: stationary, discount: 0.0, window: 8.0}
numerics: {nodes: 32}
`,
    sliders: [
      { key: "p1", label: "p₁ precision, player 1", min: 0.1, max: 100, log: true },
      { key: "p2", label: "p₂ precision, player 2", min: 0.1, max: 100, log: true },
      { key: "r1", label: "r₁ effort cost, player 1", min: 0.1, max: 10, log: true },
      { key: "r2", label: "r₂ effort cost, player 2", min: 0.1, max: 10, log: true },
      { key: "b1", label: "b₁ target, player 1", min: -2, max: 2, step: 0.1 },
      { key: "b2", label: "b₂ target, player 2", min: -2, max: 2, step: 0.1 },
    ],
    constCost: (p) => ({ player1: 0.5 * p.b1 * p.b1, player2: 0.5 * p.b2 * p.b2 }),
    constNote: (res) => res.cost_kind,
    nodes: { def: 32, options: [[16, "16: quick"], [32, "32: the default"], [48, "48: fine, slower"]] },
    window: { key: "window", check: "window", label: "Lag window", def: 8, options: [[8, "8: the default"], [12, "12: longer"], [16, "16: long, slower"]],
      hint: "A longer window holds slow-decaying responses; it costs time.", apply: (d, v) => { d.horizon.window = v; } },
  },
  ch4: {
    tab: "Kyle–Back market",
    ch: "Ch. 4",
    title: "Stationary Kyle–Back market",
    desc: `<p class="small muted" style="margin:0">Chapter 4. An informed trader watches a noisy signal of a drifting value V and trades against
      noise flow. A competitive market maker sets the price from the order flow alone.</p>
      <div class="eq"><div class="tex" data-tex="dV = \\sigma_V\\,dW^V, \\qquad \\text{order flow } dZ = D^1\\,dt + \\sigma_Z\\,dW^Z">dV = &sigma;<sub>V</sub> dW<sup>V</sup>, &nbsp; order flow dZ = D<sup>1</sup> dt + &sigma;<sub>Z</sub> dW<sup>Z</sup></div>
      <div class="tex" data-tex="\\text{trader sees } dY^1 = \\gamma_1 (V - P)\\,dt + dW^1 \\text{ and the flow}">trader sees dY<sup>1</sup> = &gamma;<sub>1</sub> (V &minus; P) dt + dW<sup>1</sup> and the flow</div>
      <div class="tex" data-tex="\\text{trader maximizes } \\mathbb{E}\\int e^{-\\rho t}\\big[D^1 (V - P) - \\varepsilon\\,(D^1)^2\\big]\\,dt, \\qquad P = \\mathbb{E}[V \\mid \\text{flow}]">trader maximizes E &int; e<sup>&minus;&rho;t</sup> [ D<sup>1</sup>(V &minus; P) &minus; &epsilon; (D<sup>1</sup>)<sup>2</sup> ] dt, &nbsp; P = E[V | flow]</div></div>
      <p class="small muted" style="margin:0">Costs are flow losses, so the trader's profit shows as a negative number.
      The market maker's number omits the V<sup>2</sup> term and is not a welfare measure. Small trading costs make the
      fixed point hard to reach; the status bar says when a solve did not converge.</p>`,
    yaml: `name: ch4_kyle_back
params: {eps: 0.2, rho: 0.5, gamma1: 1.0, sigma_V: 1.0, sigma_Z: 1.0}
shocks: [wV, wZ, w1]
states:
  V: {drift: {}, noise: {wV: sigma_V}}
agents:
  market_maker:
    controls: [P]
    myopic: true
    signals:
      flow: {drift: {D1: 1.0}, noise: {wZ: sigma_Z}}
    loss: [[1.0, P, P], [-2.0, P, V]]
  trader1:
    controls: [D1]
    signals:
      y1: {drift: {V: gamma1, P: "-gamma1"}, noise: {w1: 1.0}}
      flow: {drift: {}, noise: {wZ: sigma_Z}}
    loss: [[-1.0, D1, V], [1.0, D1, P], [eps, D1, D1]]
horizon: {kind: stationary, discount: rho, window: 12.0}
numerics: {nodes: 16}
`,
    sliders: [
      { key: "eps", label: "ε trading cost", min: 0.05, max: 2, log: true },
      { key: "rho", label: "ρ discount rate", min: 0.1, max: 2, step: 0.05 },
      { key: "gamma1", label: "γ₁ signal loading", min: 0.1, max: 4, log: true },
      { key: "sigma_V", label: "σ_V value volatility", min: 0.2, max: 3, step: 0.05 },
      { key: "sigma_Z", label: "σ_Z noise-flow volatility", min: 0.2, max: 3, step: 0.05 },
    ],
    defaultVar: "P", defaultCtl: "D1",
    channelNames: { wV: "value shock", wZ: "noise-trader flow shock", w1: "trader's signal noise" },
    nodes: { def: 16, options: [[12, "12: quick"], [16, "16: the default"], [24, "24: fine, slower"]] },
    window: { key: "window", check: "window", label: "Lag window", def: 12, options: [[8, "8: quick"], [12, "12: the default"], [16, "16: long, slower"]],
      hint: "A longer window holds slow-decaying responses; it costs time.", apply: (d, v) => { d.horizon.window = v; } },
  },
  ch5: {
    tab: "Supply-chain cycle",
    ch: "Ch. 5",
    title: "Three firms in a supply-chain cycle",
    desc: `<p class="small muted" style="margin:0">Chapter 5. Firm i buys from firm i - 1 and sells to firm i + 1 and to households, around a cycle of three.
      Each firm sets a price P<sub>i</sub> and an order o<sub>i</sub>, which take effect after a delay &tau;, and sees only noisy signals:
      its own sales, its supplier's price, its customer's order book and the order upstream. Demand, cost and firm-level shocks drive the market.</p>
      <div class="eq"><div class="tex" data-tex="\\text{sales}_i = q + (\\theta - 1)\\cdot\\text{price index} - \\theta P_i + \\eta_i \\qquad (\\text{all prices at lag } \\tau)">sales<sub>i</sub> = q + (&theta; &minus; 1) &middot; price index &minus; &theta; P<sub>i</sub> + &eta;<sub>i</sub> &nbsp; (all prices at lag &tau;)</div>
      <div class="tex" data-tex="\\text{firm } i \\text{ minimizes the average of } (\\text{price deviation})^2 + m\\,(\\text{inventory mismatch})^2 + r\\,o_i^2 + \\cdots - 2\\kappa\\,(\\text{revenue terms})">firm i minimizes the average of (price deviation)<sup>2</sup> + m (inventory mismatch)<sup>2</sup> + r o<sub>i</sub><sup>2</sup> + &hellip; &minus; 2&kappa; (revenue terms)</div></div>
      <p class="small muted" style="margin:0">The firms are symmetric, so the solver finds one firm's strategy and relabels it around the cycle.
      This is the heaviest game on the page: a solve takes a few seconds on the quick grid and about half a minute on the dissertation's.</p>`,
    yaml: `name: ch5_cycle_market
params: {theta: 4.0, xi: 0.15, zeta: 0.5, kappa: 0.3, m: 1.0, r: 0.2, rP: 0.0, c: 0.2, sigma_u: 1.0, theta_a: 0.5,
  sigma_a: 1.0, theta_eta: 0.5, sigma_eta: 1.0, s1: 2.5, s2: 0.3, s3: 0.3, s4: 2.0, tau: 0.5}
shocks: [w_q, w_a0, w_eta0, w_0_0, w_0_1, w_0_2, w_0_3, w_a1, w_eta1, w_1_0, w_1_1, w_1_2, w_1_3, w_a2, w_eta2,
  w_2_0, w_2_1, w_2_2, w_2_3]
states:
  q:
    drift: {}
    noise: {w_q: sigma_u}
  a0:
    drift: {a0: -theta_a}
    noise: {w_a0: sigma_a}
  eta0:
    drift: {eta0: -theta_eta}
    noise: {w_eta0: sigma_eta}
  a1:
    drift: {a1: -theta_a}
    noise: {w_a1: sigma_a}
  eta1:
    drift: {eta1: -theta_eta}
    noise: {w_eta1: sigma_eta}
  a2:
    drift: {a2: -theta_a}
    noise: {w_a2: sigma_a}
  eta2:
    drift: {eta2: -theta_eta}
    noise: {w_eta2: sigma_eta}
definitions:
  Pidx: {P0@tau: 0.3333333333333333, P1@tau: 0.3333333333333333, P2@tau: 0.3333333333333333}
  Pnext: {P0: 0.3333333333333333, P1: 0.3333333333333333, P2: 0.3333333333333333}
  pi0: {P0@tau: 1.0}
  i0: {o0@tau: 1.0}
  d0: {o1@tau: 1.0}
  yH0: {q: 1.0, Pidx: theta - 1, pi0: -theta, eta0: 1.0}
  pi1: {P1@tau: 1.0}
  i1: {o1@tau: 1.0}
  d1: {o2@tau: 1.0}
  yH1: {q: 1.0, Pidx: theta - 1, pi1: -theta, eta1: 1.0}
  pi2: {P2@tau: 1.0}
  i2: {o2@tau: 1.0}
  d2: {o0@tau: 1.0}
  yH2: {q: 1.0, Pidx: theta - 1, pi2: -theta, eta2: 1.0}
  dev0: {pi0: 1.0, Pidx: -(1 - xi - zeta), q: -xi, pi2: -zeta, a0@tau: zeta}
  mis0: {a0@tau: 1.0, i0: 1.0, d0: -1.0, yH0: -1.0}
  bill0: {P2: 1.0, Pnext: -1.0}
  quote_gap0: {P0: 1.0, Pnext: -1.0}
  dev1: {pi1: 1.0, Pidx: -(1 - xi - zeta), q: -xi, pi0: -zeta, a1@tau: zeta}
  mis1: {a1@tau: 1.0, i1: 1.0, d1: -1.0, yH1: -1.0}
  bill1: {P0: 1.0, Pnext: -1.0}
  quote_gap1: {P1: 1.0, Pnext: -1.0}
  dev2: {pi2: 1.0, Pidx: -(1 - xi - zeta), q: -xi, pi1: -zeta, a2@tau: zeta}
  mis2: {a2@tau: 1.0, i2: 1.0, d2: -1.0, yH2: -1.0}
  bill2: {P1: 1.0, Pnext: -1.0}
  quote_gap2: {P2: 1.0, Pnext: -1.0}
agents:
  firm0:
    controls: [P0, o0]
    signals:
      sales:
        drift: {yH0: 1.0}
        noise: {w_0_0: s1}
        delay: 0.0
      trans_price:
        drift: {P2: 1.0}
        noise: {w_0_1: s2}
        delay: 0.0
      order_book:
        drift: {o1: 1.0}
        noise: {w_0_2: s3}
        delay: 0.0
      upstream_order:
        drift: {o2: 1.0}
        noise: {w_0_3: s4}
        delay: 0.0
      own_prod:
        drift: {}
        noise: {w_a0: 1.0}
        delay: 0.0
    loss:
    - [1.0, dev0, dev0]
    - [-2*kappa, d0]
    - [-2*kappa, yH0]
    - [m, mis0, mis0]
    - [r, o0, o0]
    - [rP, quote_gap0, quote_gap0]
    - [2*c, o0, bill0]
    myopic: false
  firm1:
    controls: [P1, o1]
    signals:
      sales:
        drift: {yH1: 1.0}
        noise: {w_1_0: s1}
        delay: 0.0
      trans_price:
        drift: {P0: 1.0}
        noise: {w_1_1: s2}
        delay: 0.0
      order_book:
        drift: {o2: 1.0}
        noise: {w_1_2: s3}
        delay: 0.0
      upstream_order:
        drift: {o0: 1.0}
        noise: {w_1_3: s4}
        delay: 0.0
      own_prod:
        drift: {}
        noise: {w_a1: 1.0}
        delay: 0.0
    loss:
    - [1.0, dev1, dev1]
    - [-2*kappa, d1]
    - [-2*kappa, yH1]
    - [m, mis1, mis1]
    - [r, o1, o1]
    - [rP, quote_gap1, quote_gap1]
    - [2*c, o1, bill1]
    myopic: false
  firm2:
    controls: [P2, o2]
    signals:
      sales:
        drift: {yH2: 1.0}
        noise: {w_2_0: s1}
        delay: 0.0
      trans_price:
        drift: {P1: 1.0}
        noise: {w_2_1: s2}
        delay: 0.0
      order_book:
        drift: {o0: 1.0}
        noise: {w_2_2: s3}
        delay: 0.0
      upstream_order:
        drift: {o1: 1.0}
        noise: {w_2_3: s4}
        delay: 0.0
      own_prod:
        drift: {}
        noise: {w_a2: 1.0}
        delay: 0.0
    loss:
    - [1.0, dev2, dev2]
    - [-2*kappa, d2]
    - [-2*kappa, yH2]
    - [m, mis2, mis2]
    - [r, o2, o2]
    - [rP, quote_gap2, quote_gap2]
    - [2*c, o2, bill2]
    myopic: false
ties:
- [firm0, firm1, firm2]
horizon: {kind: stationary, discount: 0.0, window: 10.0}
numerics: {nodes: 8, unit: 0.5, unit_range: 4.0}
`,
    sliders: [
      { key: "theta", label: "θ price elasticity of demand", min: 1.5, max: 8, step: 0.1 },
      { key: "kappa", label: "κ weight on revenue", min: 0, max: 1, step: 0.05 },
      { key: "m", label: "m inventory-mismatch cost", min: 0.2, max: 5, log: true },
      { key: "s1", label: "s₁ sales-signal noise", min: 0.3, max: 8, log: true },
      { key: "s4", label: "s₄ upstream-order noise", min: 0.3, max: 8, log: true },
    ],
    nodes: { def: 8, max: 12, options: [[8, "8: the default"], [12, "12: fine, slow"]] },
    grid: { key: "window", check: "window", label: "Lag window", options: [[10, "10: quick, a few seconds"], [24, "24: the dissertation's, about 30 s"]],
      apply: (d, v) => { d.horizon.window = v; d.numerics.unit_range = v >= 16 ? 8 : 4; } },
    defaultVar: "P0", defaultCtl: "P0",
    approx: "Expected on this tab: none of the grids it offers reaches the solver's strict 1e-6 target.",
    // nineteen shocks: the response plot shows one group at a time, each firm's shocks in that firm's color
    channelGroups: [["demand and cost shocks", (c) => !/^w_\d_\d$/.test(c)], ["signal noise, firm 0", (c) => /^w_0_\d$/.test(c)],
      ["signal noise, firm 1", (c) => /^w_1_\d$/.test(c)], ["signal noise, firm 2", (c) => /^w_2_\d$/.test(c)], ["all nineteen", () => true]],
    channelStyle: (res, c) => {
      const m = /^w_(?:a|eta)(\d)$/.exec(c) || /^w_(\d)_(\d)$/.exec(c);
      if (!m) return { color: css("--c3"), dash: "solid" };
      return { color: agentColor(res, "firm" + m[1]), dash: m[2] !== undefined ? ["solid", "dash", "dot", "dashdot"][+m[2]] : c.startsWith("w_eta") ? "dash" : "solid" };
    },
    channelNames: {"w_q": "aggregate demand shock", "w_a0": "cost shock, firm 0", "w_eta0": "demand shock, firm 0", "w_0_0": "sales-signal noise, firm 0", "w_0_1": "price-signal noise, firm 0", "w_0_2": "order-book noise, firm 0", "w_0_3": "upstream-order noise, firm 0", "w_a1": "cost shock, firm 1", "w_eta1": "demand shock, firm 1", "w_1_0": "sales-signal noise, firm 1", "w_1_1": "price-signal noise, firm 1", "w_1_2": "order-book noise, firm 1", "w_1_3": "upstream-order noise, firm 1", "w_a2": "cost shock, firm 2", "w_eta2": "demand shock, firm 2", "w_2_0": "sales-signal noise, firm 2", "w_2_1": "price-signal noise, firm 2", "w_2_2": "order-book noise, firm 2", "w_2_3": "upstream-order noise, firm 2"},
  },
  ch6: {
    tab: "Transparent or opaque",
    ch: "Ch. 6",
    title: "A strategic market maker: transparent and opaque markets",
    desc: `<p class="small muted" style="margin:0">Chapter 6. Chapter 4's market with a market maker that is a player: it quotes a price P,
      absorbs the order flow into an inventory Q and pays &gamma;Q<sup>2</sup> for holding it. The trader sees the quote and trades on it
      within the instant. The two markets differ only in what the trader sees. In the <b>transparent</b> market the order flow is published,
      so a quote off the market maker's rule is a deviation the trader can see: it is privy to the market maker. In the <b>opaque</b> market
      it sees the quote alone and reads a quote off the rule as an unusual run of noise trades: it is naive.</p>
      <div class="eq"><div class="tex" data-tex="dV = dW^V, \\qquad dQ = -D\\,dt - \\sigma_Z\\,dW^Z, \\qquad \\text{order flow } dZ = D\\,dt + \\sigma_Z\\,dW^Z">dV = dW<sup>V</sup>, &nbsp; dQ = &minus;D dt &minus; &sigma;<sub>Z</sub> dW<sup>Z</sup>, &nbsp; order flow dZ = D dt + &sigma;<sub>Z</sub> dW<sup>Z</sup></div>
      <div class="tex" data-tex="\\text{trader sees } dY = (V - P)\\,dt + dW^Y \\text{ and } P;\\ \\text{transparent: also the flow}">trader sees dY = (V &minus; P) dt + dW<sup>Y</sup> and P; transparent: also the flow</div>
      <div class="tex" data-tex="\\text{trader minimizes } \\mathbb{E}\\int e^{-\\rho t}\\big[-(V - P)D + \\varepsilon D^2\\big]\\,dt">trader minimizes E &int; e<sup>&minus;&rho;t</sup> [ &minus;(V &minus; P) D + &epsilon; D<sup>2</sup> ] dt</div>
      <div class="tex" data-tex="\\text{market maker sees the flow and minimizes } \\mathbb{E}\\int e^{-\\rho t}\\big[(V - P)D + \\gamma Q^2\\big]\\,dt">market maker sees the flow and minimizes E &int; e<sup>&minus;&rho;t</sup> [ (V &minus; P) D + &gamma; Q<sup>2</sup> ] dt</div></div>
      <p class="small muted" style="margin:0">The page solves both markets on every change. At &gamma; = 0 both are Chapter 4's competitive market;
      with an inventory cost the market maker quotes against its inventory, and how far depends on what the trader can see.
      Costs are flow losses, so the trader's profit shows as a negative number.</p>`,
    yaml: `name: ch6_transparent_market
params: {eps: 0.2, gamma: 0.1, rho: 0.5, sigma_Z: 1.0}
shocks: [wV, wZ, wY]
states:
  V: {drift: {}, noise: {wV: 1.0}}
  Q: {drift: {D: -1.0}, noise: {wZ: "-sigma_Z"}}
agents:
  market_maker:
    controls: [P]
    signals:
      flow: {drift: {D: 1.0}, noise: {wZ: sigma_Z}}
    loss: [[1.0, V, D], [-1.0, P, D], [gamma, Q, Q]]
  trader:
    controls: [D]
    monitors: [market_maker]
    instant: [P]
    signals:
      y: {drift: {V: 1.0, P: -1.0}, noise: {wY: 1.0}}
      flow: {drift: {}, noise: {wZ: sigma_Z}}
    loss: [[-1.0, V, D], [1.0, P, D], [eps, D, D]]
horizon: {kind: stationary, discount: rho, window: 8.0}
numerics: {nodes: 24}
`,
    // the opaque market beside it: the trader sees the quote alone, a level it filters, and is naive to the market maker
    compare: {
      label: "Opaque", mainLabel: "Transparent",
      make: (d) => {
        const m = JSON.parse(JSON.stringify(d)), t = m.agents.trader;
        m.name = "ch6_opaque_market";
        delete t.monitors;
        t.signals = { y: t.signals.y, quote: { level: "P" } };
        return m;
      },
      // the chapter's comparison: the price's response to noise flow, and the trader's answer to a quote off the rule
      rows: (res, C) => {
        const blip = (R) => { const W = R.deviation && R.deviation.origins.market_maker; return W ? W.controls.P.samples : null; };
        const half = (R) => {
          const q = blip(R); if (!q || !(Math.abs(q.Q[0]) > 0)) return null;
          const k = q.Q.findIndex((v) => Math.abs(v) <= 0.5 * Math.abs(q.Q[0]));
          if (k < 0) return null;
          if (k === 0) return R.samples.age[0];
          const y0 = Math.abs(q.Q[k - 1]), y1 = Math.abs(q.Q[k]), target = .5 * Math.abs(q.Q[0]);
          return R.samples.age[k - 1] + (R.samples.age[k] - R.samples.age[k - 1]) * (y0 - target) / (y0 - y1);
        };
        const at0 = (R, f) => { const q = blip(R); return q ? f(q) : null; };
        return [
          ["Price response to a unit noise-trade shock, at once", res.samples.kernels.P.wZ[0], C.samples.kernels.P.wZ[0]],
          ["Trader's order when the quote is a unit above its rule: the block at once", at0(res, (q) => -q.Q[0]), at0(C, (q) => -q.Q[0])],
          ["Quote just after that blip, against its rule", at0(res, (q) => q.P[0]), at0(C, (q) => q.P[0])],
          ["Half-life of the inventory it leaves", half(res), half(C), 2],
          ["Trader's profit per unit time", -res.costs.trader, -C.costs.trader],
        ];
      },
      caption: "Costs are flow losses (the trader's profit is negative). ",
    },
    deviation: { origin: "market_maker", control: "P", show: ["Q", "D", "P"],
      caption: (res) => {
        const eps = (res.model && res.model.params && res.model.params.eps) || 0.2;
        return `Here the market maker quotes one unit above its rule. The transparent market's trader knows the quote for what it is and sells at once its loss's own reaction, a block of &minus;1/(2&epsilon;) = ${fmt(-1 / (2 * eps), 3)}, which moves the inventory Q at once; the opaque market's trader reads part of the quote as noise traders' flow and sells less. The inventory then unwinds while the market maker quotes against it.`;
      } },
    sliders: [
      { key: "gamma", label: "γ inventory cost", min: 0, max: 0.2, step: 0.01 },
      { key: "eps", label: "ε trading cost", min: 0.1, max: 1, log: true },
      { key: "rho", label: "ρ discount rate", min: 0.2, max: 2, step: 0.05 },
      { key: "sigma_Z", label: "σ_Z noise-flow volatility", min: 0.3, max: 3, step: 0.05 },
    ],
    defaultVar: "P", defaultCtl: "D",
    channelNames: { wV: "value shock", wZ: "noise-trader flow shock", wY: "trader's signal noise" },
    nodes: { def: 24, options: [[16, "16: quick"], [24, "24: the default"], [32, "32: fine, slower"]] },
    window: { key: "window", check: "window", label: "Lag window", def: 8, options: [[8, "8: the default"], [12, "12: longer"]],
      hint: "The chapter's window is 8. A longer one holds the slow unwinding of inventory; from far away the fixed point may not converge.",
      apply: (d, v) => { d.horizon.window = v; } },
  },
  tr: {
    tab: "Regime change",
    ch: "Ch. 3",
    title: "A change of regime in the tracking game",
    desc: `<p class="small muted" style="margin:0">Chapter 3's tracking game, with mean reversion, has run in a stationary equilibrium for ever.
      At time 0 player 1's signal precision jumps. The shocks born before 0 still drive the state and both players' forecasts,
      so the players move from the old equilibrium toward the new one. After T the new stationary equilibrium takes over.</p>
      <div class="eq"><div class="tex" data-tex="dX = (-aX + D^1 + D^2)\\,dt + dW^0">dX = (&minus;a X + D<sup>1</sup> + D<sup>2</sup>) dt + dW<sup>0</sup></div>
      <div class="tex" data-tex="dY^i = \\sqrt{p_i(t)}\\,X\\,dt + dW^i, \\qquad p_1(t) = p_1^{\\text{before}} \\text{ for } t &lt; 0,\\ p_1 \\text{ after}">dY<sup>i</sup> = &radic;p<sub>i</sub>(t) X dt + dW<sup>i</sup>, &nbsp; p<sub>1</sub>(t) = p<sub>1</sub><sup>before</sup> for t &lt; 0, p<sub>1</sub> after</div>
      <div class="tex" data-tex="\\text{player } i \\text{ minimizes } \\mathbb{E}\\int_0^T \\big[\\tfrac12 (X - b_i)^2 + \\tfrac12 r_i\\,(D^i)^2\\big]\\,dt, \\text{ then the new stationary flow}">player i minimizes E &int;<sub>0</sub><sup>T</sup> [ &frac12; (X &minus; b<sub>i</sub>)<sup>2</sup> + &frac12; r<sub>i</sub> (D<sup>i</sup>)<sup>2</sup> ] dt, then the new stationary flow</div></div>
      <p class="small muted" style="margin:0">b<sub>i</sub> is player i&rsquo;s target, the same before and after 0. Reported costs leave out the constant &frac12; b<sub>i</sub><sup>2</sup> per unit time, which no strategy changes.</p>
      <p class="small muted" style="margin:0">The strip carries the old shocks on a band of depth L = 3 below s = 0.
      "Until settled" lets the solver pick T: it marches T = 0, 3, 6, &hellip; until the best-response rules on the last window are within 2% of the new stationary ones.</p>`,
    yaml: `name: regime_change
params: {p1: 6.0, p2: 3.0, r1: 1.0, r2: 1.0, a: 1.0, T: 9.0, b1: 1.0, b2: -1.0}
shocks: [w0, w1, w2]
states:
  X: {drift: {X: "-a", D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[0.5, X, X], ["-b1", X], ["0.5*r1", D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
    loss: [[0.5, X, X], ["-b2", X], ["0.5*r2", D2, D2]]
horizon:
  kind: transition
  T: T
  past:
    model:
      name: before
      params: {p1: 1.0, p2: 3.0, r1: 1.0, r2: 1.0, a: 1.0, b1: 1.0, b2: -1.0}
      shocks: [w0, w1, w2]
      states:
        X: {drift: {X: "-a", D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
      agents:
        player1:
          controls: [D1]
          signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
          loss: [[0.5, X, X], ["-b1", X], ["0.5*r1", D1, D1]]
        player2:
          controls: [D2]
          signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
          loss: [[0.5, X, X], ["-b2", X], ["0.5*r2", D2, D2]]
      horizon: {kind: stationary, window: 3.0}
      numerics: {nodes: 12}
  continuation: stationary
numerics: {nodes: 8, continuation_nodes: 12}
`,
    sliders: [
      { key: "p1_before", param: "p1", where: "past", label: "p₁ before 0, player 1", min: 0.1, max: 100, log: true },
      { key: "p1", label: "p₁ after 0, player 1", min: 0.1, max: 100, log: true },
      { key: "p2", where: "both", label: "p₂ precision, player 2", min: 0.1, max: 100, log: true },
      { key: "r1", where: "both", label: "r₁ effort cost, player 1", min: 0.1, max: 10, log: true },
      { key: "r2", where: "both", label: "r₂ effort cost, player 2", min: 0.1, max: 10, log: true },
      { key: "a", where: "both", label: "a mean reversion", min: 0.3, max: 3, step: 0.05 },
      { key: "b1", where: "both", label: "b₁ target, player 1", min: -2, max: 2, step: 0.1 },
      { key: "b2", where: "both", label: "b₂ target, player 2", min: -2, max: 2, step: 0.1 },
      { key: "T", label: "T end of the transition", min: 3, max: 12, step: 0.5, march: false },
    ],
    nodes: { def: 8, max: 18, options: [[5, "5: fastest, rough"], [6, "6: quick"], [8, "8: about a second"], [10, "10: fine, a few seconds"]] },
    window: { key: "pastwindow", check: "past window", label: "Past window (band depth L)", def: 3, options: [[3, "3: the default"], [4.5, "4.5: longer"], [6, "6: long"]],
      hint: "How far back the old regime's shocks are carried. A longer band leaves less of [0, T] past T - L.",
      apply: (d, v) => { d.horizon.past.model.horizon.window = v; } },
    march: true,
    defaultVar: "D1", defaultCtl: "D1",
    // the surface opens on the response that the regime change moves most: player 1's reaction to its own signal
    // noise, which doubles at s = 0 when the precision jumps (peak 0.13 on old shocks, 0.26 on new ones)
    surfaceDefault: ["D1", "w1"],
    approx: "Expected on this tab: none of the grids it offers reaches the solver's strict 1e-6 target.",
    meansCaption: "",
  },
  custom: {
    tab: "Your model",
    title: "Your model",
    desc: `<p class="small muted" style="margin:0">Write or paste a model file, or start from one of the examples of the noisestate package.
      The solver takes the package's model files, written as equations or in the grammar: stationary, finite and transition
      horizons, with the past model written inline under horizon.past.model and an optional horizon.settle for the march in T.</p>`,
  },
};
const EXAMPLES = {
  "tracking game with targets (Ch. 1)": PRESETS.ch1.yaml,
  "tracking game written as equations (Ch. 1)": `# The tracking game of the first tab, written as the package's equations: a loss (X - b)^2
# keeps its constant b^2, so these costs include b^2 T.
name: ch1_tracking_equations
params: {p1: 9.0, p2: 9.0, r1: 0.1, r2: 0.1, b1: 1.0, b2: -1.0, sigma: 1.0, T: 1.0}
shocks: [w0, w1, w2]
states:
  X: (D1 + D2) dt + sigma dw0
agents:
  player1:
    controls: D1
    observes: {y1: sqrt(p1) X dt + dw1}
    loss: (X - b1)^2 + r1 D1^2
  player2:
    controls: D2
    observes: {y2: sqrt(p2) X dt + dw2}
    loss: (X - b2)^2 + r2 D2^2
horizon: {T: T}
numerics: {nodes: 12}
`,
  "stationary tracking (Ch. 3)": PRESETS.ch3.yaml,
  "Kyle–Back market (Ch. 4)": PRESETS.ch4.yaml,
  "transparent market with a strategic market maker (Ch. 6)": PRESETS.ch6.yaml,
  "tracking game, both players privy (Ch. 6)": `# Chapter 6's all-privy corner: each player monitors the other, so a deviation is answered for what it is,
# with the feedback-Nash gain; without monitors: the all-naive corner, where it is filtered as noise.
# The panel "A deviation and who sees it" draws the answers.
name: ch6_privy_tracking
params: {p1: 3.0, p2: 10.0, r1: 1.0, r2: 1.0}
shocks: [w0, w1, w2]
states:
  X: (D1 + D2) dt + dw0
agents:
  player1:
    controls: D1
    monitors: player2
    observes: {y1: sqrt(p1) X dt + dw1}
    loss: 0.5 X^2 + 0.5 r1 D1^2
  player2:
    controls: D2
    monitors: player1
    observes: {y2: sqrt(p2) X dt + dw2}
    loss: 0.5 X^2 + 0.5 r2 D2^2
horizon: {window: 8.0}
numerics: {nodes: 24}
`,
  "change of regime (Ch. 3 transition)": PRESETS.tr.yaml,
  "transition shorter than the past's window, initial shock": `# T = 1 < L = 2: the old shocks stay alive on the buffer; an initial shock xi
# loads on the state and player 2 sees it at once; player 1 is myopic.
name: tr_short
params: {p1: 10.0, p2: 3.0, r1: 1.0, r2: 1.0}
shocks: [w0, w1, w2]
states:
  X: {drift: {X: -0.2, D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
agents:
  player1:
    controls: [D1]
    myopic: true
    signals:
      y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}
    loss: [[0.5, X, X], ["0.5*r1", D1, D1]]
  player2:
    controls: [D2]
    signals:
      y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}
    loss: [[0.5, X, X], ["0.5*r2", D2, D2]]
horizon:
  kind: transition
  T: 1.0
  past:
    model:
      name: tr_short_past
      params: {p1: 3.0, p2: 3.0, r1: 1.0, r2: 1.0}
      shocks: [w0, w1, w2]
      states:
        X: {drift: {X: -0.2, D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
      agents:
        player1:
          controls: [D1]
          signals:
            y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}
          loss: [[0.5, X, X], ["0.5*r1", D1, D1]]
        player2:
          controls: [D2]
          signals:
            y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}
          loss: [[0.5, X, X], ["0.5*r2", D2, D2]]
      horizon: {kind: stationary, window: 2.0}
      numerics: {nodes: 10}
    initial:
      - {name: xi, loads: {X: 0.5}, rows: {player2.y2: 1.0}}
  continuation: stationary
numerics: {nodes: 5, continuation_nodes: 10}
`,
  "Kyle–Back from a prior (Back 1992): the value is drawn at time 0": `# The insider sees V at once; the market maker learns it from the order flow.
# Draw sample paths: each draw's price P walks toward its own V by T.
name: kyle_back_prior
params: {eps: 0.2, Sigma0: 1.0, sigma_Z: 1.0}
shocks: [wZ]
states:
  V: {drift: {}, noise: {}}
agents:
  market_maker:
    controls: [P]
    myopic: true
    signals:
      flow: {drift: {D1: 1.0}, noise: {wZ: sigma_Z}}
    loss: [[1.0, P, P], [-2.0, P, V]]
  trader1:
    controls: [D1]
    signals:
      flow: {drift: {}, noise: {wZ: sigma_Z}}
    loss: [[-1.0, D1, V], [1.0, D1, P], [eps, D1, D1]]
numerics: {nodes: 10}
horizon:
  kind: transition
  T: 2.0
  discount: 0.0
  past:
    initial:
      - {name: v0, loads: {V: "sqrt(Sigma0)"}, rows: {trader1.flow: 1.0}}
  continuation: end
`,
  "delayed control and observation (Ch. 1), a few seconds": `# Finite-horizon two-player game with a control delay and a delayed observation.
name: ch1_delayed_finite
params: {p1: 3.0, p2: 3.0, r1: 0.1, r2: 0.1, sigma: 1.0, tau: 0.25}
shocks: [w0, w1, w2]
states:
  X: {drift: {"D1@tau": 1.0, "D2@tau": 1.0}, noise: {w0: sigma}}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[1.0, X, X], [r1, D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}, delay: tau}}
    loss: [[1.0, X, X], [r2, D2, D2]]
horizon: {kind: finite, T: 1.0}
numerics: {nodes: 8}
`,
  "mean reversion, constant drift and initial state": `name: finite_means
params: {p1: 4.0, p2: 1.0, r1: 0.2, r2: 0.5, b1: 1.0, b2: -0.5, x0: 0.7, k: 0.4}
shocks: [w0, w1, w2]
states:
  X: {drift: {X: -0.3, D1: 1.0, D2: 1.0, const: k}, noise: {w0: 0.8}, initial: x0}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[1.0, X, X], ["-2*b1", X], [r1, D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
    loss: [[1.0, X, X], ["-2*b2", X], [r2, D2, D2]]
horizon: {kind: finite, T: 1.5, discount: 0.3}
numerics: {nodes: 10}
`,
};

const NAME_LABEL = { X: "state X", V: "value V", P: "price P", Q: "inventory Q" };
const CHANNEL_LABEL = { w0: "common shock W⁰", w1: "signal noise W¹", w2: "signal noise W²", w3: "signal noise W³" };
const AGENT_LABEL = { player1: "Player 1", player2: "Player 2", market_maker: "Market maker", trader1: "Informed trader", trader: "Trader", firm0: "Firm 0", firm1: "Firm 1", firm2: "Firm 2" };

// ---------------------------------------------------------------------------------------------
// State
let game = "ch1";
const values = {};              // preset -> {slider key: value, nodes}
const opts = { refine: false, stability: false, march: false, reference: true };
const MAX_NODES = 96, MAX_WINDOW = 96;   // how far a "solve again with" button, or a link, may take the grid and the window
const lastStart = {};
const lastStartCompare = {};       // preset -> the raw maps of its compared model's last equilibrium
let solverThreads = 1;
let prevResult = null;            // the result before the last change, drawn faintly for comparison
let pathSeed = 1;
let keepGroup = 0;            // the group of shocks the Supply-chain response plot shows, kept across solves             // preset -> the raw maps of its last equilibrium
let customYaml = "";
let worker = null, workerReady = false;
let reqId = 0, inFlight = null, pending = false, debounce = null, lastResult = null;
let solveStart = 0, timerHandle = null, progress = null;

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => { if (x === null || x === undefined || !isFinite(x)) return "–"; const s = Number(x).toFixed(d); return /^-0(\.0*)?$/.test(s) ? s.slice(1) : s; };   // no "-0.0000"
const fmtE = (x) => (x === null || x === undefined || !isFinite(x)) ? "–" : Number(x).toExponential(1);
// the explorer's colors live on its root element (they follow the site's light and dark themes)
const css = (v) => getComputedStyle(document.querySelector(".explorer") || document.documentElement).getPropertyValue(v).trim();
const isDark = () => document.documentElement.classList.contains("dark");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// a caption may carry [text](#hash) links that set the explorer's own state
const links = (s) => esc(s).replace(/\[([^\]]+)\]\((#[^)\s]+)\)/g, '<a href="$2">$1</a>');

function presetModel(g) { return jsyaml.load(PRESETS[g].yaml); }

// the solver calls every finite-horizon cost a "discounted integral"; most games here have no discounting
function costKind(res) {
  const kind = res.cost_kind || "";
  if (!kind.startsWith("discounted ")) return kind;
  let m; try { m = currentModel(); } catch (e) { return kind; }
  let d = m && m.horizon ? m.horizon.discount : undefined;
  if (typeof d === "string" && m.params && d in m.params) d = m.params[d];
  return d === undefined || Number(d) === 0 ? kind.slice("discounted ".length) : kind;
}

// ---------------------------------------------------------------------------------------------
// Each game in Python, in noisestate's equations form (the README's "As equations"): the structure written
// out, the parameter values, horizon and numerics read from the model being solved, so the code is the
// solve on screen. tools/check_explorer_python.js builds every one and checks it is the YAML's model.
const py = {
  num: (v) => (typeof v === "number" ? (Number.isInteger(v) ? v.toFixed(1) : String(v)) : String(v)),
  params(p, names) {
    const k = names || Object.keys(p);
    return `${k.join(", ")}${k.length === 1 ? "," : ""} = ns.params(${k.map((n) => `${n}=${py.num(p[n])}`).join(", ")})`;
  },
  horizon(h, past) {
    const d = h.discount !== undefined && h.discount !== 0 && h.discount !== "0" ? `, discount=${py.num(h.discount)}` : "";
    if (h.kind === "finite") return `ns.Finite(T=${py.num(h.T)}${d})`;
    if (h.kind === "stationary") return `ns.Stationary(window=${py.num(h.window)}${d})`;
    const T = h.T !== undefined ? h.T : 6.0;
    const cont = (h.continuation || "stationary") === "stationary" ? "" : `, continuation="${h.continuation}"`;
    return `ns.Transition(T=${py.num(T)}, past=${past}${cont}${d})`;
  },
  numerics(n) {
    // node counts are whole numbers (nodes=12, as the package's examples write them); the rest as the model has them
    return `ns.Numerics(${Object.entries(n).map(([k, v]) => `${k}=${k === "engine" ? JSON.stringify(v) : /nodes$/.test(k) ? String(v) : py.num(v)}`).join(", ")})`;
  },
  head: (extra = "") => `import noisestate as ns\nfrom noisestate import dt, sqrt${extra}\n`,
};
// the tracking games: one state X pushed by both players, each seeing X through its own noise
const tracking = (loss1, loss2, drift = "D1 + D2", noise = "sigma * dw0") => `dw0, dw1, dw2 = ns.shocks("w0", "w1", "w2")
X = ns.State("X")
D1, D2 = ns.Control("D1"), ns.Control("D2")
X.d = (${drift}) * dt + ${noise}
player1 = ns.Agent("player1", controls=D1, observes={"y1": sqrt(p1) * X * dt + dw1},
                   loss=${loss1})
player2 = ns.Agent("player2", controls=D2, observes={"y2": sqrt(p2) * X * dt + dw2},
                   loss=${loss2})`;
const kyleBack = `dwV, dwZ, dw1 = ns.shocks("wV", "wZ", "w1")
V = ns.State("V")
P, D1 = ns.Control("P"), ns.Control("D1")
V.d = sigma_V * dwV
market_maker = ns.Agent("market_maker", controls=P, myopic=True,
                        observes={"flow": D1 * dt + sigma_Z * dwZ},
                        loss=P**2 - 2 * P * V)
trader1 = ns.Agent("trader1", controls=D1,
                   observes={"y1": (gamma1 * V - gamma1 * P) * dt + dw1, "flow": sigma_Z * dwZ},
                   loss=-D1 * V + D1 * P + eps * D1**2)`;
// the model panel under the results: the model file and the same model in Python, one shown at a time
let codeTab = "yaml", modelOpen = false;
function codeTabs(on) {
  return `<div class="tabs codetabs" role="tablist">${[["yaml", "YAML"], ["python", "Python"]].map(([k, l]) =>
    `<button type="button" class="tab" role="tab" data-pane="${k}" aria-selected="${k === on}">${l}</button>`).join("")}</div>`;
}
function modelPanel(d) {
  const yaml = jsyaml.dump(d, { flowLevel: 3 }), code = PYTHON[game] ? PYTHON[game](d) : "";
  const pane = (k, text) => `<div class="pane" data-pane="${k}"${codeTab === k ? "" : " hidden"}>
    <button type="button" class="secondary small copycode" data-copy="${esc(text)}" title="Copy">Copy</button><pre class="describe">${esc(text)}</pre></div>`;
  return `<details class="modelcode"${modelOpen ? " open" : ""}><summary>The model</summary>${codeTabs(codeTab)}${pane("yaml", yaml)}${pane("python", code)}</details>`;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest(".codetabs [data-pane]");
  if (!b) return;
  const box = b.closest(".codetabs").parentElement;
  box.querySelectorAll(".codetabs [data-pane]").forEach((x) => x.setAttribute("aria-selected", x === b));
  box.querySelectorAll(":scope > .pane").forEach((p) => { p.hidden = p.dataset.pane !== b.dataset.pane; });
  if (b.closest("details.modelcode")) codeTab = b.dataset.pane;
});
document.addEventListener("toggle", (e) => { if (e.target.classList && e.target.classList.contains("modelcode")) modelOpen = e.target.open; }, true);

const PYTHON = {
  ch1: (d) => `${py.head()}
${py.params(d.params)}
${tracking("X**2 - 2 * b1 * X + r1 * D1**2", "X**2 - 2 * b2 * X + r2 * D2**2")}
game = ns.Game(X, [player1, player2], horizon=${py.horizon(d.horizon)},
               name="${d.name}", numerics=${py.numerics(d.numerics)})
eq = game.solve()`,
  ch3: (d) => `${py.head()}
${py.params(d.params)}
${tracking("0.5 * X**2 + 0.5 * r1 * D1**2", "0.5 * X**2 + 0.5 * r2 * D2**2", "D1 + D2", "dw0")}
game = ns.Game(X, [player1, player2], horizon=${py.horizon(d.horizon)},
               name="${d.name}", numerics=${py.numerics(d.numerics)})
eq = game.solve()`,
  ch4: (d) => `${py.head()}
${py.params(d.params)}
${kyleBack}
game = ns.Game(V, [market_maker, trader1], horizon=${py.horizon(d.horizon)},
               name="${d.name}", numerics=${py.numerics(d.numerics)})
eq = game.solve()`,
  ch5: (d) => `${py.head(", define")}
N = 3                                                    # firms on the cycle
theta, xi, zeta, kappa, m, r, rP, c = ns.params(${["theta", "xi", "zeta", "kappa", "m", "r", "rP", "c"].map((n) => `${n}=${py.num(d.params[n])}`).join(", ")})
sigma_u, theta_a, sigma_a, theta_eta, sigma_eta = ns.params(${["sigma_u", "theta_a", "sigma_a", "theta_eta", "sigma_eta"].map((n) => `${n}=${py.num(d.params[n])}`).join(", ")})
s1, s2, s3, s4, tau = ns.params(${["s1", "s2", "s3", "s4", "tau"].map((n) => `${n}=${py.num(d.params[n])}`).join(", ")})
dw = ns.shocks("w_q", *[n for v in range(N) for n in (f"w_a{v}", f"w_eta{v}", *[f"w_{v}_{k}" for k in range(4)])])
q = ns.State("q"); q.d = sigma_u * dw.w_q
a, eta, P, o = [], [], [], []
for v in range(N):
    a.append(ns.State(f"a{v}")); a[v].d = -theta_a * a[v] * dt + sigma_a * dw[f"w_a{v}"]
    eta.append(ns.State(f"eta{v}")); eta[v].d = -theta_eta * eta[v] * dt + sigma_eta * dw[f"w_eta{v}"]
    P.append(ns.Control(f"P{v}")); o.append(ns.Control(f"o{v}"))
Pidx = define("Pidx", sum(P[v].lag(tau) for v in range(N)) / N)          # the price index in force
Pnext = define("Pnext", sum(P[v] for v in range(N)) / N)                 # the quotes' mean
defs = [Pidx, Pnext]; pi, i, d, yH = [], [], [], []
for v in range(N):
    cus = (v + 1) % N
    pi.append(define(f"pi{v}", P[v].lag(tau)))                           # the price in force
    i.append(define(f"i{v}", o[v].lag(tau)))                             # the input arriving
    d.append(define(f"d{v}", o[cus].lag(tau)))                           # the deliveries owed
    yH.append(define(f"yH{v}", q + (theta - 1) * Pidx - theta * pi[v] + eta[v]))     # household demand
    defs += [pi[v], i[v], d[v], yH[v]]
firms = []
for v in range(N):
    sup, cus = (v - 1) % N, (v + 1) % N
    dev = define(f"dev{v}", pi[v] - (1 - xi - zeta) * Pidx - xi * q - zeta * pi[sup] + zeta * a[v].lag(tau))
    mis = define(f"mis{v}", a[v].lag(tau) + i[v] - d[v] - yH[v])
    bill = define(f"bill{v}", P[sup] - Pnext)
    quote_gap = define(f"quote_gap{v}", P[v] - Pnext)
    defs += [dev, mis, bill, quote_gap]
    loss = (dev**2 - 2 * kappa * d[v] - 2 * kappa * yH[v] + m * mis**2 + r * o[v]**2 + rP * quote_gap**2
            + 2 * c * o[v] * bill)
    firms.append(ns.Agent(f"firm{v}", controls=[P[v], o[v]], loss=loss, observes={
        "sales": yH[v] * dt + s1 * dw[f"w_{v}_0"], "trans_price": P[sup] * dt + s2 * dw[f"w_{v}_1"],
        "order_book": o[cus] * dt + s3 * dw[f"w_{v}_2"], "upstream_order": o[sup] * dt + s4 * dw[f"w_{v}_3"],
        "own_prod": dw[f"w_a{v}"]}))
game = ns.Game([q] + [x for v in range(N) for x in (a[v], eta[v])], firms, definitions=defs, ties=[firms],
               horizon=${py.horizon(d.horizon)}, name="${d.name}", numerics=${py.numerics(d.numerics)})
eq = game.solve()`,
  ch6: (d) => `${py.head()}
${py.params(d.params)}
dwV, dwZ, dwY = ns.shocks("wV", "wZ", "wY")
V, Q = ns.State("V"), ns.State("Q")
P, D = ns.Control("P"), ns.Control("D")
V.d = dwV
Q.d = -D * dt - sigma_Z * dwZ                            # the market maker absorbs the order flow
market_maker = ns.Agent("market_maker", controls=P, observes={"flow": D * dt + sigma_Z * dwZ},
                        loss=V * D - P * D + gamma * Q**2)
# the transparent market: the trader sees the quote at once and the order flow, so it is privy to the market maker
trader = ns.Agent("trader", controls=D, monitors=market_maker,
                  observes={"y": (V - P) * dt + dwY, "flow": sigma_Z * dwZ, "quote": ns.level(P)},
                  loss=-V * D + P * D + eps * D**2)
# the opaque market: the quote alone, a level it filters, and naive to the market maker
opaque_trader = ns.Agent("trader", controls=D, observes={"y": (V - P) * dt + dwY, "quote": ns.level(P, filter=True)},
                         loss=-V * D + P * D + eps * D**2)
horizon, numerics = ${py.horizon(d.horizon)}, ${py.numerics(d.numerics)}
opaque = ns.Game([V, Q], [market_maker, opaque_trader], horizon=horizon, name="ch6_opaque_market", numerics=numerics)
game = ns.Game([V, Q], [market_maker, trader], horizon=horizon, name="${d.name}", numerics=numerics)
eq = game.solve()
# eq_opaque = opaque.solve()
# the market maker's quote blip, seed age on the x-axis: eq.deviation_response(market_maker, [Q, D, P]).over(ages)`,
  tr: (d) => {
    const b = d.horizon.past.model, march = d.horizon.settle !== undefined;
    const now = Object.keys(d.params).filter((k) => k !== "T");
    return `${py.head()}
def tracking(name, horizon, numerics, **values):
    """The stationary tracking game with a mean-reverting state, at the given parameter values."""
    ${now.join(", ")} = ns.params(**{k: values[k] for k in (${now.map((k) => `"${k}"`).join(", ")})})
    ${tracking("0.5 * X**2 + 0.5 * r1 * D1**2", "0.5 * X**2 + 0.5 * r2 * D2**2", "-a * X + D1 + D2", "dw0").replace(/\n/g, "\n    ")}
    return ns.Game(X, [player1, player2], horizon=horizon, name=name, numerics=numerics)

# before the change: the stationary game at the old values
before = tracking("${b.name}", ${py.horizon(b.horizon)}, ${py.numerics(b.numerics)},
                  ${Object.keys(b.params).map((k) => `${k}=${py.num(b.params[k])}`).join(", ")})
${march ? `# The explorer is marching in T (settle: ${d.horizon.settle} in the model file); the Python here solves to the
# fixed T below.
` : ""}T = ${march ? "6.0" : `ns.params(T=${py.num(d.params.T)})[0]`}
game = tracking("${d.name}", ${py.horizon({ ...d.horizon, T: "T" }, "before")}, ${py.numerics(d.numerics)},
                ${now.map((k) => `${k}=${py.num(d.params[k])}`).join(", ")})
eq = game.solve()`;
  },
};
// a slider's parameter lives in the model's params ("now"), the past model's ("past"), or both
function sliderParams(d, s) {
  const where = s.where || "now", out = [];
  if (where !== "past") out.push(d.params);
  if (where !== "now" && d.horizon.past && d.horizon.past.model) out.push(d.horizon.past.model.params);
  return out;
}
function isControl(name) { return !!(lastResult && lastResult.agents && lastResult.agents.some((a) => a.controls.includes(name))); }
function label(name) { if (typeof name !== "string") return "–"; return NAME_LABEL[name] || (isControl(name) || /^[A-Z]\d*$/.test(name) ? `control ${name}` : name); }
function chLabel(res, c) { return (PRESETS[game] && PRESETS[game].channelNames && PRESETS[game].channelNames[c]) || CHANNEL_LABEL[c] || c; }

// ---------------------------------------------------------------------------------------------
// URL state
function b64encode(s) { return btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""); }
function b64decode(s) { return decodeURIComponent(escape(atob(s.replace(/-/g, "+").replace(/_/g, "/")))); }
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("game") && PRESETS[h.get("game")]) game = h.get("game");
  for (const g of Object.keys(PRESETS)) {
    if (!PRESETS[g].sliders) continue;
    const m = presetModel(g);
    values[g] = { nodes: m.numerics.nodes };
    if (PRESETS[g].grid) values[g][PRESETS[g].grid.key] = PRESETS[g].grid.options[0][0];
    if (PRESETS[g].window) values[g][PRESETS[g].window.key] = PRESETS[g].window.def;
    for (const s of PRESETS[g].sliders) values[g][s.key] = sliderParams(m, s)[0][s.param || s.key];
  }
  opts.refine = h.get("refine") === "1"; opts.stability = h.get("stability") === "1";
  opts.march = h.get("march") === "1"; opts.reference = h.get("method") !== "solver";
  customYaml = PRESETS.ch1.yaml;
  if (game === "custom" && h.get("model")) {
    try { customYaml = b64decode(h.get("model")); } catch (e) { /* keep the default */ }
  } else if (game !== "custom") {
    for (const s of PRESETS[game].sliders) {
      const v = parseFloat(h.get(s.key));
      if (isFinite(v)) values[game][s.key] = Math.min(s.max, Math.max(s.min, v));
    }
    const n = parseInt(h.get("nodes"));
    if (PRESETS[game].nodes && n >= 2 && n <= (PRESETS[game].nodes.max || MAX_NODES)) values[game].nodes = n;
    for (const G of [PRESETS[game].grid, PRESETS[game].window]) {
      if (!G) continue;
      const w = parseFloat(h.get(G.key));
      if (w > 0 && w <= MAX_WINDOW) values[game][G.key] = w;
    }
  }
}
function writeHash() {
  let h;
  if (game === "custom") {
    const enc = b64encode(customYaml);
    h = new URLSearchParams(enc.length < 6000 ? { game, model: enc } : { game });
  } else {
    const v = {}; for (const s of PRESETS[game].sliders) v[s.key] = values[game][s.key];
    if (PRESETS[game].nodes) v.nodes = values[game].nodes;
    if (PRESETS[game].grid) v[PRESETS[game].grid.key] = values[game][PRESETS[game].grid.key];
    if (PRESETS[game].window) v[PRESETS[game].window.key] = values[game][PRESETS[game].window.key];
    h = new URLSearchParams({ game, ...v });
  }
  if (game === "ch6" && !opts.reference) h.set("method", "solver");
  if (opts.refine) h.set("refine", "1");
  if (opts.stability) h.set("stability", "1");
  if (opts.march && PRESETS[game].march) h.set("march", "1");
  history.replaceState(null, "", "#" + h.toString());
}

// ---------------------------------------------------------------------------------------------
// Controls
const sliderToValue = (p, s) => p.log ? Math.exp(Math.log(p.min) + (Math.log(p.max) - Math.log(p.min)) * s / 1000) : p.min + (p.max - p.min) * s / 1000;
const valueToSlider = (p, v) => p.log ? 1000 * (Math.log(v) - Math.log(p.min)) / (Math.log(p.max) - Math.log(p.min)) : 1000 * (v - p.min) / (p.max - p.min);
function roundParam(p, v) {
  if (p.step) return Math.round(v / p.step) * p.step;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(v))) - 1);
  return Math.round(v / mag) * mag;
}
// a readout to the slider's own precision: a stepped slider to its step's decimals, a log slider to the two significant
// figures roundParam keeps
function showVal(p, v) {
  const d = p.step ? (String(p.step).split(".")[1] || "").length : v ? Math.max(0, 1 - Math.floor(Math.log10(Math.abs(v)))) : 0;
  return Number(v).toFixed(d);
}

function renderTabs() {
  const tabs = $("tabs"); tabs.innerHTML = "";
  const order = ["ch1", "ch3", "tr", "ch4", "ch5", "ch6", "custom"];
  const keys = [...order.filter((g) => PRESETS[g]), ...Object.keys(PRESETS).filter((g) => !order.includes(g))];
  for (const g of keys) {
    const def = PRESETS[g];
    const b = document.createElement("button");
    b.className = "tab"; b.setAttribute("role", "tab");
    b.innerHTML = `${esc(def.tab)}${def.ch ? `<span class="ch">${esc(def.ch)}</span>` : ""}`;
    b.setAttribute("aria-selected", g === game ? "true" : "false");
    b.onclick = () => {
      if (g === game) return;
      game = g; renderAll(); writeHash(); lastResult = null; clearPlots($("results")); $("savebtn").disabled = true;
      // "Your model" waits for Solve: a solve still running for the tab left behind is stopped, not left reporting
      if (g !== "custom") requestSolve(0); else { if (inFlight) stopSolve(); clearTimeout(debounce); customReady(); }
    };
    tabs.appendChild(b);
  }
  // keep the selected game in view when the row scrolls sideways (phones)
  const sel = tabs.querySelector('[aria-selected="true"]');
  if (sel && tabs.scrollWidth > tabs.clientWidth) tabs.scrollLeft = Math.max(0, sel.offsetLeft - tabs.offsetLeft - 24);
}

function renderControls() {
  const def = PRESETS[game];
  $("gametitle").textContent = def.title;
  $("gamedesc").innerHTML = def.desc;
  typesetEquations($("gamedesc"));
  const box = $("controls"); box.innerHTML = "";
  $("editor").hidden = game !== "custom";
  if (game === "custom") { renderEditor(); renderOptions($("paramopts")); return; }
  if (game === "ch6") {
    const view = document.createElement("div"); view.className = "ctl ch6-view";
    view.innerHTML = `<label for="ch6-view">View</label><select id="ch6-view"><option value="reference">Full equilibrium</option><option value="solver">General solver</option></select>
      <span class="hint">Follow the full inventory unwind, or inspect the general solver's finite-window approximation and first-order conditions.</span>`;
    box.appendChild(view);
    const select = view.querySelector("select"); select.value = opts.reference ? "reference" : "solver";
    select.onchange = () => { opts.reference = select.value === "reference"; renderControls(); writeHash(); requestSolve(0); };
  }
  for (const p of def.sliders) {
    if (p.march === false && opts.march) continue;
    const wrap = document.createElement("div"); wrap.className = "ctl";
    const id = "ctl-" + p.key;
    wrap.innerHTML = `<label for="${id}"><span>${p.label}</span><span class="val"></span></label><input type="range" id="${id}" min="0" max="1000" step="1">`;
    box.appendChild(wrap);
    const input = wrap.querySelector("input"), out = wrap.querySelector(".val");
    input.value = valueToSlider(p, values[game][p.key]);
    out.textContent = showVal(p, values[game][p.key]);
    input.addEventListener("input", () => {
      const v = roundParam(p, sliderToValue(p, +input.value));
      values[game][p.key] = v; out.textContent = showVal(p, v);
      writeHash(); requestSolve(500);
    });
  }
  // a numerics choice: its listed options, and the current value as one more when a "solve again" button went past them
  const numSelect = (id, label, hint, options, key) => {
    const v = values[game][key], opts_ = options.some((o) => o[0] === v) ? options : options.concat([[v, `${v}: as the solver asked`]]).sort((a, b) => a[0] - b[0]);
    const wrap = document.createElement("div"); wrap.className = "ctl";
    wrap.innerHTML = `<label for="${id}"><span>${label}</span></label><select id="${id}">${opts_.map((o) => `<option value="${o[0]}">${o[1]}</option>`).join("")}</select>
      <div class="hint">${hint}</div>`;
    box.appendChild(wrap);
    const sel = wrap.querySelector("select"); sel.value = v;
    sel.addEventListener("change", () => { values[game][key] = +sel.value; writeHash(); requestSolve(0); });
  };
  if (def.grid) numSelect("ctl-grid", def.grid.label, "A longer window holds the slow-decaying responses; it costs time.", def.grid.options, def.grid.key);
  if (def.window && !(game === "ch6" && opts.reference)) numSelect("ctl-window", def.window.label, def.window.hint, def.window.options, def.window.key);
  if (def.nodes && !(game === "ch6" && opts.reference)) numSelect("ctl-nodes", "Grid (nodes per side)", "More nodes are more accurate and slower.", def.nodes.options, "nodes");
  renderOptions(box);
}

// solver options: the end of a transition and the after-solve checks
function renderOptions(box) {
  const def = PRESETS[game];
  if (game === "ch6" && opts.reference) {
    return;
  }
  const wrap = document.createElement("div"); wrap.className = "ctl opts";
  let html = "";
  if (def.march) html += `<label class="pick"><span>End of the transition</span><select id="opt-march">
      <option value="0">Fixed T (the slider)</option><option value="1">Until settled (march in T)</option></select></label>`;
  html += `<label class="check" title="Re-solve on a grid 1.5 times finer and report how much the costs and the kernels (the shock-response curves) move"><input type="checkbox" id="opt-refine"> Refinement check</label>
      <label class="check" title="Whether players who kept best-responding to each other, starting near this equilibrium, would settle back into it: the spectral radius of the best-response map, below 1 if they would"><input type="checkbox" id="opt-stability"> Stability</label>
      <span class="hint">Checks run after the solve and add time.</span>`;
  wrap.innerHTML = html; box.appendChild(wrap);
  const on = (id, f) => { const e = wrap.querySelector("#" + id); if (e) e.addEventListener("change", () => { f(e); writeHash(); requestSolve(0); }); return e; };
  const m = on("opt-march", (e) => { opts.march = e.value === "1"; renderControls(); });
  if (m) m.value = opts.march ? "1" : "0";
  on("opt-refine", (e) => { opts.refine = e.checked; }).checked = opts.refine;
  on("opt-stability", (e) => { opts.stability = e.checked; }).checked = opts.stability;
}

function renderEditor() {
  const ex = $("example");
  if (!ex.options.length) ex.innerHTML = Object.keys(EXAMPLES).map((k) => `<option>${esc(k)}</option>`).join("");
  const ta = $("yaml");
  if (ta.value !== customYaml) ta.value = customYaml;
  renderParamControls();
}

function customModel() {
  let d;
  try { d = jsyaml.load($("yaml").value); }
  catch (e) { throw new Error("The model file is not valid YAML: " + e.message.split("\n")[0]); }
  if (!d || typeof d !== "object") throw new Error("The model file must be a mapping (name, shocks, states, agents, horizon).");
  const has = (k) => d[k] && typeof d[k] === "object" && Object.keys(d[k]).length;
  if (!has("agents")) throw new Error("The model file needs at least one agent under agents, each with its controls and loss.");
  // a file written as equations may leave its shocks to be read off the dW terms of its equations
  const equations = Object.values(d.states || {}).some((v) => typeof v === "string" || (v && typeof v === "object" && "d" in v))
    || Object.values(d.agents).some((a) => a && typeof a === "object" && ("observes" in a || typeof a.loss === "string"));
  if (!equations && !has("shocks") && !has("channels")) throw new Error("The model file needs a list of shocks, the noise that drives the game.");
  return d;
}

function renderParamControls() {
  const box = $("paramcontrols"); box.innerHTML = ""; $("yamlerror").textContent = "";
  let d;
  try { d = customModel(); } catch (e) { $("yamlerror").textContent = e.message; return; }
  const params = d.params || {};
  for (const [k, v] of Object.entries(params)) {
    if (typeof v !== "number") continue;
    const wrap = document.createElement("div"); wrap.className = "ctl";
    wrap.innerHTML = `<label for="pp-${esc(k)}"><span>${esc(k)}</span></label><input type="number" step="any" id="pp-${esc(k)}" value="${v}">`;
    box.appendChild(wrap);
    wrap.querySelector("input").addEventListener("change", (ev) => {
      const x = parseFloat(ev.target.value);
      if (!isFinite(x)) return;
      const dd = customModel(); dd.params[k] = x;
      customYaml = setParamInYaml($("yaml").value, k, x); $("yaml").value = customYaml;
      writeHash(); requestSolve(0);
    });
  }
}

// Replace one parameter's value in the YAML text, keeping its layout; falls back to re-dumping the file.
function setParamInYaml(text, key, value) {
  const re = new RegExp(`(\\bparams\\s*:\\s*\\{[^}]*?\\b${key}\\s*:\\s*)([-+0-9.eE]+)`);
  if (re.test(text)) return text.replace(re, `$1${value}`);
  const re2 = new RegExp(`(^\\s+${key}\\s*:\\s*)([-+0-9.eE]+)\\s*$`, "m");
  if (re2.test(text)) return text.replace(re2, `$1${value}`);
  const d = jsyaml.load(text); d.params[key] = value; return jsyaml.dump(d, { flowLevel: 3 });
}

function renderAll() { renderTabs(); renderControls(); if (typeof updateSweepPanel === "function") updateSweepPanel(); }

// ---------------------------------------------------------------------------------------------
// Status
function setStatus(kind, chipText, text) {
  const chip = $("chip"); chip.className = "chip " + kind; chip.textContent = chipText;
  $("statustext").textContent = text;
}
function startTimer(reset = true) {
  if (reset || !timerHandle) { solveStart = performance.now(); progress = null; rallyStart(); settleReset(); }
  clearInterval(timerHandle);
  const tick = () => {
    const s = (performance.now() - solveStart) / 1000;
    let t = `Solving in your browser, ${s.toFixed(1)} s so far`;
    if (progress) t += `, ${progress.evaluation} best-response rounds, residual ${fmtE(progress.residual)}`;
    t += ".";
    if (pending) t += " Your latest change is queued.";
    if (s > 8) t += " Bigger grids, delays and many agents take longer; Stop cancels.";
    setStatus("busy", "Solving", t);
  };
  tick(); timerHandle = setInterval(tick, 200);
}
function stopTimer() { clearInterval(timerHandle); timerHandle = null; }

// the rally beside the chip: one hit for each best-response round the solve reports (at most a dozen), then the
// ball settles in the middle: the fixed point. A solve that runs past a second starts the rally while it works,
// a hit per progress report, and tops it up to the count when it ends.
const rally = { hits: 0, played: 0, side: 1, busy: false, done: false, live: false, timer: 0 };
// under the status bar, the fixed point settling: the residual the solver reports after each best-response round, on
// a log scale, falling to the dashed line at the solve's own tolerance (1e-10 stationary, 1e-8 finite: the result's
// "converged" check carries it). A request can run more than one fixed-point solve (a start-up solve, then the model's):
// the round count starting over marks a new one, drawn as its own line from round 1, the earlier ones faint. It stays
// after the solve, a record of how this one converged.
const settle = { runs: [], lastEval: 0, tol: {}, main: -1, cmp: -1 };
function settleTol() { return settle.tol[game] || 1e-10; }
function settleReset() { Object.assign(settle, { runs: [], lastEval: 0, main: -1, cmp: -1 }); settleDraw(); }
function settleAdd(r, ev) {
  if (!(r > 0 && isFinite(r))) return;
  if (!settle.runs.length || (ev !== undefined && ev <= settle.lastEval)) settle.runs.push([]);
  settle.runs[settle.runs.length - 1].push(Math.log10(r));
  if (ev !== undefined) settle.lastEval = ev;
  settleDraw();
}
function settleDone(res) {
  const c = (res.checks || []).find((d) => d.name === "converged");
  if (c && c.threshold > 0) settle.tol[game] = c.threshold;
  if (!settle.runs.length) settleAdd(res.residual);
  // which lines are the model's solve and the compared one (a start on a coarse grid runs before each): the last run
  // with the result's round count, and for the comparison the last with its count after it
  const find = (n, from) => { for (let j = settle.runs.length - 1; j >= from; j--) if (settle.runs[j].length === n) return j; return -1; };
  settle.main = find(res.evaluations, 0);
  const C = res.compare;
  settle.cmp = C && C.ok !== false && C.evaluations ? find(C.evaluations, settle.main + 1) : -1;
  if (settle.cmp === settle.main) settle.cmp = -1;
  settleDraw();
}
function settleDraw() {
  const svg = document.getElementById("conv"), panel = document.getElementById("convpanel");
  if (!svg) return;
  const runs = settle.runs.filter((r) => r.length);
  if (panel) panel.hidden = !runs.length;
  if (!runs.length) { svg.innerHTML = ""; return; }
  const W = Math.max(240, svg.clientWidth || 600), H = W < 480 ? 150 : 190;
  const L = 42, R = 14, T = 10, B = 28;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("height", H);
  const all = runs.flat(), lt = Math.log10(settleTol());
  const top = Math.ceil(Math.max(...all, lt + 1)), bot = Math.floor(Math.min(...all, lt) - 0.5);
  const n = Math.max(5, ...runs.map((r) => r.length));
  const x = (i) => L + (W - L - R) * (n === 1 ? 0 : i / (n - 1)), y = (v) => T + (H - T - B) * (top - v) / Math.max(1e-9, top - bot);
  const step = top - bot > 8 ? 4 : 2;
  let g = "";
  for (let d = top; d >= bot; d--) {
    if (d % step) continue;
    g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(d).toFixed(1)}" y2="${y(d).toFixed(1)}"/>`;
    g += `<text class="tick" x="${L - 6}" y="${(y(d) + 4).toFixed(1)}" text-anchor="end">${d === 0 ? "1" : `10<tspan dy="-6" font-size="0.72em">${d < 0 ? "\u2212" + -d : d}</tspan>`}</text>`;
  }
  const xs = n <= 10 ? 1 : n <= 25 ? 5 : n <= 60 ? 10 : 25;
  for (let k = 1; k <= n; k++) if (k === 1 || k % xs === 0) g += `<text class="tick" x="${x(k - 1).toFixed(1)}" y="${H - 10}" text-anchor="middle">${k}</text>`;
  g += `<text class="axis" x="${W - R}" y="${H - 10}" text-anchor="end" dx="0">round</text>`;
  g += `<line class="tol" x1="${L}" x2="${W - R}" y1="${y(lt).toFixed(1)}" y2="${y(lt).toFixed(1)}"/>`;
  g += `<text class="axis" x="${W - R}" y="${(y(lt) - 5).toFixed(1)}" text-anchor="end">tolerance</text>`;
  // while solving, the latest line is the live one; after, the model's solve (solid, with its rounds marked) and the
  // compared one (dashed, as in the plots below), the start-up solves faint
  const done = settle.main >= 0;
  const role = (j) => done ? (j === settle.main ? "run" : j === settle.cmp ? "run cmp" : "run early") : (j === runs.length - 1 ? "run" : "run early");
  const rank = { "run early": 0, "run cmp": 1, "run": 2 };        // faint underneath, the model's solve on top
  const order = runs.map((r, j) => j).sort((a, b) => rank[role(a)] - rank[role(b)] || a - b);
  for (const j of order) {
    const r = runs[j], c = role(j);
    g += `<polyline class="${c}" points="${r.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}"/>`;
    if (c === "run" && (W - L - R) / n >= 7) g += r.map((v, i) => `<circle class="pt" r="2.6" cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}"/>`).join("");
  }
  svg.innerHTML = g;
}
addEventListener("resize", () => { if (settle.runs.length) settleDraw(); });
// the wide tables' fades: data-more says on which sides a table has more to scroll to ("l", "r", "lr", or none)
function tableFades(w) {
  const l = w.scrollLeft > 1, r = w.scrollLeft + w.clientWidth < w.scrollWidth - 1;
  const v = (l ? "l" : "") + (r ? "r" : "");
  if (v) w.dataset.more = v; else delete w.dataset.more;
}
function allTableFades() { for (const w of document.querySelectorAll(".explorer .tablewrap")) tableFades(w); }
document.addEventListener("scroll", (e) => { if (e.target.classList && e.target.classList.contains("tablewrap")) tableFades(e.target); }, true);
addEventListener("resize", allTableFades);
{ let queued = false;
  new MutationObserver(() => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; allTableFades(); }); } })
    .observe(document.body, { childList: true, subtree: true }); }
function rallyStart() {
  clearTimeout(rally.timer);
  Object.assign(rally, { hits: 0, played: 0, done: false, live: false });
  rally.timer = setTimeout(() => { rally.live = true; }, 1000);
}
function rallyProgress() { if (rally.live && rally.hits - rally.played < 2) { rally.hits++; rallyPlay(); } }
function rallyEnd(n) {
  clearTimeout(rally.timer);
  rally.hits = Math.max(rally.hits, Math.min(n, 12));
  rally.done = true; rallyPlay();
}
function rallyPlay() {
  const ball = document.getElementById("rallyball");
  if (!ball || rally.busy || !ball.animate || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const from = +ball.getAttribute("cx");
  let to, ms;
  if (rally.played < rally.hits) { rally.side = -rally.side; to = rally.side > 0 ? 34 : 6; ms = 150; rally.played++; }
  else if (rally.done && from !== 20) { to = 20; ms = 260; }
  else return;
  rally.busy = true;
  const a = ball.animate([{ transform: "translateX(0)" }, { transform: `translateX(${to - from}px)` }], { duration: ms, easing: to === 20 ? "ease-out" : "linear" });
  a.onfinish = () => { ball.setAttribute("cx", to); rally.busy = false; rallyPlay(); };
}

// ---------------------------------------------------------------------------------------------
// Worker
// the model tab waits for Solve rather than solving on load, so the button has to be pressable
function customReady() { setStatus("idle", "Ready", "Edit the model and press Solve."); if (!inFlight) $("solvebtn").disabled = false; }
function failWorker(message) {
  if (worker) worker.terminate();
  worker = null; workerReady = false; inFlight = null; pending = false;
  clearTimeout(debounce); stopTimer();
  $("stopbtn").disabled = true; $("solvebtn").disabled = false;
  setStatus("bad", "Failed", message + ". Press Solve again to retry.");
}
function startWorker(autoSolve = true) {
  workerReady = false;
  try { const url = new URL(WORKER_URL, location.href); if (game === "ch6" && opts.reference) url.searchParams.set("lazy", "1"); worker = new Worker(url); }
  catch (e) { failWorker("This browser could not start a Web Worker: " + e.message); return; }
  const activeWorker = worker;
  worker.onmessage = (ev) => {
    if (activeWorker !== worker) return;
    const m = ev.data;
    if (m.type === "ready") {
      workerReady = true; solverThreads = m.threads || 1;
      const tf = document.getElementById("threadfact");
      if (tf) tf.textContent = solverThreads > 1 ? `${solverThreads} threads in this browser` : "single-threaded in this browser";
      if (pending || (autoSolve && game !== "custom")) requestSolve(0);
      else if (game === "custom" && !inFlight && !lastResult) customReady();
    } else if (m.type === "progress") {
      if (inFlight && m.id === inFlight.id) { progress = m; rallyProgress(); settleAdd(m.residual, m.evaluation); }
    } else if (m.type === "result") {
      onSolved(m);
    } else if (m.type === "fatal") {
      failWorker("The solver could not load: " + m.message);
    }
  };
  worker.onerror = (e) => { if (activeWorker === worker) failWorker("The solver stopped: " + (e.message || "unknown error")); };
}
function stopSolve() {
  if (!inFlight) return;
  worker.terminate(); worker = null; workerReady = false; inFlight = null; pending = false;
  clearTimeout(debounce); stopTimer();
  $("stopbtn").disabled = true; $("solvebtn").disabled = false;
  setStatus("warn", "Stopped", "The solve was stopped. Change a parameter or press Solve again.");
}

// The equations in each game's description: typeset with KaTeX (served from the site) when it has loaded, otherwise
// the plain HTML already in the block stays.
function typesetEquations(root) {
  if (!window.katex) { window.addEventListener("load", () => window.katex && typesetEquations(root), { once: true }); return; }
  for (const d of root.querySelectorAll(".tex[data-tex]")) {
    try { katex.render(d.dataset.tex, d, { displayMode: false, throwOnError: true }); d.classList.add("typeset"); } catch (e) { /* keep the HTML fallback */ }
  }
}

// The in-browser solver is a C++ port of noisestate 2 (noisestate-cpp), which reads the package's model files as they
// are: shocks, written as equations or in the grammar.  An older file that names its shocks "channels" passes too.
function forSolver(d) { return d; }
function currentModel() {
  if (game === "custom") return customModel();
  const def = PRESETS[game], d = presetModel(game);
  for (const s of def.sliders) for (const P of sliderParams(d, s)) P[s.param || s.key] = values[game][s.key];
  if (def.nodes) d.numerics.nodes = values[game].nodes;
  // a transition's continuation on the same grid: unequal grids leave a floor under the settled check that no T removes
  if (def.nodes && d.numerics.continuation_nodes !== undefined) d.numerics.continuation_nodes = values[game].nodes;
  if (def.grid) def.grid.apply(d, values[game][def.grid.key]);
  if (def.window) def.window.apply(d, values[game][def.window.key]);
  if (def.march && opts.march) { delete d.horizon.T; delete d.params.T; d.horizon.settle = 0.02; }
  return d;
}

function currentRequest() { return game === "ch6" && opts.reference ? { method: "ch6-markov" } : { refine: opts.refine, stability: opts.stability }; }
function requestSolve(delay) {
  clearTimeout(debounce);
  if (lastResult) $("results").classList.add("stale");
  debounce = setTimeout(() => {
    if (!workerReady) { pending = true; if (!worker) startWorker(false); return; }
    if (inFlight) { pending = true; startTimer(false); return; }
    sendSolve();
  }, delay);
}
function sendSolve() {
  let model;
  try { model = currentModel(); }
  catch (e) { $("yamlerror").textContent = e.message; setStatus("bad", "Error", e.message); return; }
  $("yamlerror").textContent = "";
  if ($("statusfix")) $("statusfix").innerHTML = "";
  inFlight = { id: ++reqId, game, key: JSON.stringify([model, currentRequest()]) };
  pending = false;
  $("solvebtn").disabled = true; $("stopbtn").disabled = false;
  startTimer();
  // a warm start: the last equilibrium of this tab, which the solver uses when the shapes match (a parameter moved)
  // and ignores otherwise (a new grid or a new model)
  const request = { ...currentRequest(), return_start: true };
  addCompare(request, model, game, lastStartCompare[game]);
  // naive_observers was removed in noisestate 2 (it computed neither of Chapter 6's corners); an old file's key is dropped
  if (game === "custom" && model.naive_observers) { model = { ...model }; delete model.naive_observers; }
  const hk = model.horizon && model.horizon.kind;
  request.path_grid = hk === "stationary" ? 150 : hk === "transition" ? 48 : 60;
  if (lastStart[game]) request.start = lastStart[game];
  else request.start_policy = "coarse";   // a cold solve starts from the same model on a coarser grid
  worker.postMessage({ type: "solve", id: inFlight.id, model: forSolver(model), request });
}
// Chapter 6: a tab with a compared model (the opaque market beside the transparent one) sends it along, from its own last
// equilibrium; a model with monitors or instant observations asks for its deviation worlds (a stationary one: the solver
// computes them there)
function addCompare(request, model, g, start) {
  const def = PRESETS[g];
  if (g !== "custom" && def.compare) request.compare = { model: forSolver(def.compare.make(model)), label: def.compare.label, ...(start ? { start } : {}) };
  const ch6 = Object.values(model.agents || {}).some((a) => a && (a.monitors || a.instant || (a.observes && JSON.stringify(a.observes).includes('"level"'))
    || Object.values(a.signals || {}).some((r) => r && r.level)));
  const stat = !model.horizon || !model.horizon.kind || model.horizon.kind === "stationary";
  if ((g !== "custom" && def.deviation) || (g === "custom" && ch6 && stat)) request.deviation = { continuation: "blip" };
}
function onSolved(m) {
  if (!inFlight || m.id !== inFlight.id) return;
  const req = inFlight; inFlight = null; stopTimer();
  $("solvebtn").disabled = false; $("stopbtn").disabled = true;
  let key = null;
  try { key = JSON.stringify([currentModel(), currentRequest()]); } catch (e) { /* the editor holds an unfinished edit */ }
  const stale = pending || !req || req.game !== game || (key !== null && key !== req.key);
  const res = JSON.parse(m.result);
  rallyEnd(res.ok && res.engine !== "ch6-markov" ? res.evaluations || 0 : 0);
  if (res.ok && res.engine !== "ch6-markov") settleDone(res);
  if (res.ok && window.siteTally)
    window.siteTally("solve", res.compare ? 2 : 1, `${(PRESETS[game] && PRESETS[game].tab) || "your model"}, ${(m.wall || 0).toFixed(1)} s`);
  if (!res.ok) {
    if (req && req.game === game) {
      setStatus("bad", "Error", res.error);
      if (game === "custom") $("yamlerror").textContent = res.error;
      $("results").classList.remove("stale");
    }
    if (stale && req && req.game === game) sendSolve();
    return;
  }
  if (res.start && req) { if (res.converged) lastStart[req.game] = res.start; delete res.start; }
  if (res.compare && res.compare.start && req) { if (res.compare.converged) lastStartCompare[req.game] = res.compare.start; delete res.compare.start; }
  if (req && req.game === game) {
    const tf = document.getElementById("threadfact");
    if (tf) tf.textContent = res.engine === "ch6-markov" ? "finite-state equilibrium" : solverThreads > 1 ? `${solverThreads} threads in this browser` : "single-threaded in this browser";
    prevResult = lastResult && lastResult.name === res.name && lastResult.kind === res.kind ? lastResult : null;
    lastResult = res;
    $("savebtn").disabled = false;
    try { renderResults(res); plotSnapshot = null; updateSweepPanel(); drawSweep(); }
    catch (e) { console.error(e); setStatus("warn", "Solved, drawing failed", "The solve finished but a plot could not be drawn: " + e.message); $("results").classList.remove("stale"); return; }
  }
  if (stale) { sendSolve(); return; }
  $("results").classList.remove("stale");
  const failed = res.checks.filter((d) => d.ok === false && d.name !== "converged");
  const t = m.wall < 0.1 ? "under 0.1" : m.wall.toFixed(1), how = res.warm_start ? " from the last equilibrium" : "";
  const accuracy = failed.filter((d) => ACCURACY_CHECKS.has(d.name)), other = failed.filter((d) => !ACCURACY_CHECKS.has(d.name));
  if (res.engine === "ch6-markov") setStatus("ok", "Equilibrium", `Solved in ${t} s. Costs include the full inventory tail; the curves follow the finite-state equilibrium.`);
  else if (!res.converged) setStatus("bad", "Not converged", `The fixed point did not converge (residual ${fmtE(res.residual)} after ${res.evaluations} rounds). Try a finer grid or less extreme parameters.`);
  else if (failed.length && !other.length)
    setStatus("warn", "Solved, approximate", `Solved in ${t} s${how}. ${cap(accuracy.map(accuracyNote).join("; "))}.`
      + (game !== "custom" && PRESETS[game].approx && accuracy.every((d) => d.name === "resolution" || d.name === "settled") ? " " + PRESETS[game].approx : ""));
  else if (failed.length) setStatus("warn", "Converged, with warnings", `Solved in ${t} s${how}. Failed: ${failed.map((d) => checkName(d.name)).join(", ")}; the Diagnostics table says what each means.`);
  else setStatus("ok", "Solved", `Solved in ${t} s${how}, ${res.evaluations} best-response rounds, residual ${fmtE(res.residual)}. All checks passed.`);
  showFixes(res.converged ? accuracy : failed.filter((d) => d.name === "resolution"));
}

// Checks that measure numerical accuracy (the grid, the windows, whether a transition has settled), as against
// findings about the equilibrium (a saddle, unstable best responses). Only these grade a solve "approximate".
const ACCURACY_CHECKS = new Set(["resolution", "window", "past window", "continuation window", "settled"]);
const cap = (x) => x.charAt(0).toUpperCase() + x.slice(1);
function accuracyNote(d) {
  const pct = (x) => (100 * x).toFixed(x < 0.01 ? 2 : 1) + "%";
  if (d.name === "resolution") return `the grid represents the strategies to ${fmtE(d.value)} (target ${fmtE(d.threshold)})`;
  if (d.name === "settled") return `the transition's last window is ${fmtE(d.value)} of its peak from the new stationary rules (target ${fmtE(d.threshold)})`;
  const which = d.name === "window" ? "" : d.name === "past window" ? "the old regime's " : "the new regime's ";
  return `${which}kernels still move by ${pct(d.value)} of their peak at the end of the window (target ${pct(d.threshold)})`;
}
// For each failed check whose remedy is a setting on this page, a button that raises it by half and solves again,
// past the listed choices if need be.
function fixesFor(checks) {
  if (game === "custom") return [];
  const def = PRESETS[game], out = [], seen = new Set();
  for (const d of checks) {
    let fix = null;
    if (d.name === "settled") {
      // not settled by T: the remedy is a longer transition, not a finer grid
      const S = (def.sliders || []).find((s) => s.key === "T");
      const t = values[game].T;
      if (S && t !== undefined) {
        const next = Math.min(S.max, Math.round(t * 1.5 * 2) / 2);
        if (next > t) fix = { key: "T", value: next, label: `Solve again with T = ${next}` };
      }
    } else if (d.name === "resolution" && def.nodes) {
      // a transition's grid grows in two dimensions and memory runs out past its max (27 nodes: out of memory)
      const n = values[game].nodes, next = Math.min(def.nodes.max || MAX_NODES, Math.max(n + 2, Math.round(n * 1.5)));
      if (next > n) fix = { key: "nodes", value: next, label: `Solve again with ${next} nodes` };
    } else {
      const W = [def.window, def.grid].find((G) => G && G.check === d.name);
      if (W) {
        const w = values[game][W.key], next = Math.min(MAX_WINDOW, Math.round(w * 1.5 * 2) / 2);
        if (next > w) fix = { key: W.key, value: next, label: `Solve again with ${d.name === "past window" ? "a past window" : "a window"} of ${next}` };
      }
    }
    if (fix && !seen.has(fix.key)) { seen.add(fix.key); out.push(fix); }
  }
  return out;
}
function showFixes(checks) {
  let box = $("statusfix");
  if (!box) { box = document.createElement("span"); box.id = "statusfix"; $("statustext").after(box); }
  box.innerHTML = "";
  for (const f of fixesFor(checks)) {
    const b = document.createElement("button");
    b.className = "secondary fix"; b.type = "button"; b.textContent = f.label;
    b.title = "The solver asked for this; it goes past the listed choices if need be, and takes longer";
    b.onclick = () => { values[game][f.key] = f.value; writeHash(); renderControls(); box.innerHTML = ""; requestSolve(0); };
    box.appendChild(b);
  }
}

// ---------------------------------------------------------------------------------------------
// Plots
function baseLayout(extra) {
  const text = css("--text"), muted = css("--muted"), grid = css("--grid");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Inter, sans-serif", size: 12, color: text },
    margin: { l: 52, r: 12, t: 34, b: 70 },
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: muted }, automargin: false },
    yaxis: { gridcolor: grid, zerolinecolor: muted, linecolor: grid, tickfont: { color: muted }, automargin: false, exponentformat: "power" },
    legend: { orientation: "h", x: 0, y: -0.28, yanchor: "top", font: { size: 11, color: muted }, bgcolor: "rgba(0,0,0,0)" },
    showlegend: true, hovermode: "x unified",
  }, extra || {});
}
// on a phone a plot is short and its legend long: the x-axis title moves into the plot's title, so the legend under
// the axis has the room the title took
// The previous solve's curves, by plot, so a re-solve of the same game glides from the old equilibrium to the new
// one instead of redrawing: curves are matched by name and length, the axes hold both ranges meanwhile, and anything
// unmatched (the dotted "before" lines, heatmaps) simply appears. Off under reduced motion.
let plotSnapshot = null;
const plotWork = new WeakMap();
function trackPlot(el, work) {
  let pending = plotWork.get(el);
  if (!pending) { pending = new Set(); plotWork.set(el, pending); }
  const token = {};
  pending.add(token);
  return Promise.resolve(work).finally(() => {
    pending.delete(token);
    if (!pending.size) {
      plotWork.delete(el);
      if (!el.isConnected) Plotly.purge(el);
    }
  });
}
const plotKey = (el) => el.id || (el.parentElement && el.parentElement.id ? el.parentElement.id + ":" + [...el.parentElement.children].indexOf(el) : null);
function snapshotPlots(root) {
  const snap = {};
  for (const gd of root.querySelectorAll(".js-plotly-plot")) {
    const key = plotKey(gd), fl = gd._fullLayout;
    if (!key || !gd.data || !fl || !fl.xaxis || !fl.yaxis) continue;
    snap[key] = { x: fl.xaxis.range.slice(), y: fl.yaxis.range.slice(),
                  traces: gd.data.map((t) => ({ name: t.name, type: t.type || "scatter", x: t.x ? Array.from(t.x) : null, y: t.y ? Array.from(t.y) : null })) };
  }
  return snap;
}
// Plotly's responsive handlers retain their chart after its DOM is detached.
// Dispose before replacing a plot container, including selector/redraw changes.
function clearPlots(root) {
  // Purging during Plotly's own asynchronous layout deletes state its remaining
  // callbacks still need. Detach now; the tracked operation disposes on settling.
  if (window.Plotly) for (const gd of root.querySelectorAll(".js-plotly-plot")) {
    if (!plotWork.has(gd)) Plotly.purge(gd);
  }
  root.replaceChildren();
}
function glide(el, data, layout, cfg) {
  const old = plotSnapshot && plotSnapshot[plotKey(el)];
  const oneAxis = layout && !Object.keys(layout).some((k) => /^[xy]axis\d/.test(k));
  if (!old || !oneAxis) return null;
  const used = new Set(), idx = [], finals = [];
  const start = data.map((t, i) => {
    if ((t.type || "scatter") !== "scatter" || !t.y || !t.x) return t;
    const j = old.traces.findIndex((o, k) => !used.has(k) && o.name === t.name && o.type === "scatter" && o.y && o.x && o.y.length === t.y.length && o.x.length === t.x.length);
    if (j < 0) return t;
    used.add(j); idx.push(i); finals.push({ x: t.x, y: t.y });
    return { ...t, x: old.traces[j].x, y: old.traces[j].y };
  });
  if (!idx.length) return null;
  el.style.visibility = "hidden";
  return Plotly.newPlot(el, data, layout, cfg).then(() => {
    if (!el.isConnected) return;
    // the axes hold the union of the old and new ranges while the curves move (this Plotly does not
    // interpolate a range change inside a transition), then settle on the new equilibrium's own
    const nx = el._fullLayout.xaxis.range, ny = el._fullLayout.yaxis.range;
    const union = (a, b) => [Math.min(a[0], b[0]), Math.max(a[1], b[1])];
    const fixed = { ...layout, xaxis: { ...(layout.xaxis || {}), range: union(old.x, nx), autorange: false }, yaxis: { ...(layout.yaxis || {}), range: union(old.y, ny), autorange: false } };
    return Plotly.react(el, start, fixed, cfg).then(() => {
      if (!el.isConnected) return;
      el.style.visibility = "";
      return Plotly.animate(el, { data: finals, traces: idx }, { transition: { duration: 650, easing: "cubic-in-out" }, frame: { duration: 650, redraw: false } });
    }).then(() => el.isConnected && Plotly.relayout(el, { "xaxis.autorange": true, "yaxis.autorange": true }));
  }).catch(() => { if (!el.isConnected) return; el.style.visibility = ""; return Plotly.newPlot(el, data, layout, cfg); });
}

// Plotly loads in idle time after the page (the template passes its URL); a plot asked for before it arrives waits for it.
// Once it is loaded, plotly() draws synchronously, as it always did.
const PLOTLY_SRC = (document.currentScript && document.currentScript.dataset.plotly) || "vendor/plotly-cartesian-2.35.2.min.js";
let plotlyLoading = null;
function ensurePlotly() {
  if (window.Plotly) return Promise.resolve();
  if (!plotlyLoading) plotlyLoading = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = PLOTLY_SRC; s.onload = resolve; s.onerror = () => { plotlyLoading = null; reject(new Error("Plotly failed to load")); };
    document.head.appendChild(s);
  });
  return plotlyLoading;
}
window.addEventListener("load", () => (window.requestIdleCallback || ((f) => setTimeout(f, 200)))(() => ensurePlotly().catch(() => {})));

function plotly(method, id, data, layout, cfg) {
  // Capture this node before waiting for the lazy script. A later render can
  // reuse its id, but the earlier data must never draw into that new chart.
  const el = typeof id === "string" ? document.getElementById(id) : id;
  if (!el || !el.isConnected) return Promise.resolve();
  if (!window.Plotly) return ensurePlotly().then(() => plotly(method, el, data, layout, cfg));
  if (el && el.clientWidth && el.clientWidth < 520 && layout) {
    layout = { ...layout };
    const t = layout.xaxis && layout.xaxis.title && (layout.xaxis.title.text || layout.xaxis.title);
    if (t && typeof t === "string") {
      layout.xaxis = { ...layout.xaxis, title: undefined };
      if (layout.title && layout.title.text) layout.title = { ...layout.title, text: `${layout.title.text} <span style="font-size:11px">· against ${t}</span>` };
    }
    layout.legend = { ...(layout.legend || {}), y: -0.12 };
  }
  if (method === "newPlot" && el) { const g = glide(el, data, layout, cfg); if (g) return trackPlot(el, g); }
  return trackPlot(el, Plotly[method](el, data, layout, cfg));
}
const plotCfg = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };
const titleOf = (s) => ({ text: s, font: { size: 13 }, x: 0, xanchor: "left", xref: "paper" });
const palette = () => ["--c3", "--c1", "--c2", "--c4", "--c5", "--c6"].map(css);
const line = (x, y, name, color, extra) => Object.assign({ x, y, name, type: "scatter", mode: "lines", line: { color, width: 2 } }, extra || {});
const nonzero = (arr) => arr.some((v) => v !== null && Math.abs(v) > 1e-12);
// one color per player on every panel of a tab, the same one the cost sweep and the transition use: the model's
// agents in order take --c1, --c2, --c4, ...; a state, which no one controls, is drawn in ink (--c3)
function agentColor(res, a) { const pal = palette(), i = (res.agents || []).findIndex((q) => q.name === a); return i < 0 ? pal[0] : pal[(i + 1) % pal.length]; }
function ownerColor(res, name) { const a = (res.agents || []).find((q) => q.controls.includes(name)); return a ? agentColor(res, a.name) : css("--c3"); }

function renderResults(res) {
  const out = $("results");
  const keepVar = out.querySelector("#kvar")?.value, keepCtl = out.querySelector("#fctl")?.value;
  plotSnapshot = prevResult && !window.matchMedia("(prefers-reduced-motion: reduce)").matches ? snapshotPlots(out) : null;
  clearPlots(out);
  const def = PRESETS[game];
  const params = res.params_used || (game !== "custom" ? values[game] : {});
  const cards = Object.entries(res.costs).map(([a, v]) => {
    let shown = v, note = res.cost_kind;
    const extra = def.constCost ? def.constCost(values[game])[a] : 0;
    if (def.constCost) { shown = v + extra; note = def.constNote ? def.constNote(res) : "expected cost over [0, T]"; }
    const parts = res.cost_parts[a];
    const split = parts && Math.abs(parts.mean) > 1e-12 ? `variance ${fmt(parts.variance, 3)}, mean ${fmt(parts.mean + extra, 3)}` : "";
    const before = prevResult && prevResult.costs[a] !== undefined ? prevResult.costs[a] + extra : null;
    const dv = before === null ? 0 : shown - before;
    const delta = before !== null && Math.abs(dv) > 5e-5 ? `<span class="delta ${dv > 0 ? "up" : "down"}">${dv > 0 ? "▲" : "▼"} ${fmt(Math.abs(dv), 4)}</span>` : "";
    return `<div class="card"><div class="k">${esc(AGENT_LABEL[a] || a)}</div><div class="v">${fmt(shown, 4)} ${delta}</div><div class="d">${split || note}</div></div>`;
  }).join("");
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Equilibrium costs</h2><div class="cards">${cards}</div>
    <p class="caption">Expected losses at the equilibrium (${esc(costKind(res))}); smaller is better.${prevResult ? " Arrows: the change from the previous solve." : ""}${res.compare ? ` These are the ${esc(((PRESETS[game].compare || {}).mainLabel || "first").toLowerCase())} market's; the ${esc((res.compare.label || "compared").toLowerCase())} one's are in the next panel.` : ""}</p>
    ${res.engine === "ch6-markov" ? `<p class="caption">The finite-state solution includes the full inventory tail. The response plots show the first ${fmt(res.reference.plot_extent, 1)} units of time; their right edge is a crop, not an end to the response.${res.params_used.gamma === 0 ? " At zero inventory cost, inventory is a random walk and carries no penalty; this is a different limit from a positive penalty on a stationary inventory." : ""}</p>` : ""}
    ${res.warnings && res.warnings.length ? `<p class="caption" style="color:var(--warn)">${res.warnings.map(esc).join("<br>")}</p>` : ""}</section>`);
  if (res.kind === "transition") renderTransition(res, out);
  if (res.compare) renderCompare(res, out);
  if (res.deviation) renderDeviation(res, out);
  if (res.paths) renderPaths(res, out);
  if (res.kind.startsWith("finite") || res.kind === "transition") renderFinite(res, out, keepVar, keepCtl);
  else renderStationary(res, out, keepVar, keepCtl);
  renderDiagnostics(res, out);
}

function renderFinite(res, out, keepVar, keepCtl) {
  const S = res.samples, T = res.T;
  const pathNames = res.names.filter((n) => nonzero(S.means[n] || []));
  // means that stay flat (a change of regime that moves only the responses to shocks) draw as flat lines that say
  // nothing; say so in words instead, with the size of the largest move
  const big = Math.max(0, ...pathNames.map((n) => Math.max(...S.means[n].map(Math.abs))));
  const moved = Math.max(0, ...pathNames.map((n) => Math.max(...S.means[n]) - Math.min(...S.means[n])));
  if (res.has_means && pathNames.length && big > 0 && moved < 0.05 * big) {
    out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Mean paths</h2>
      <p class="caption">The means stay almost flat here: over the horizon none moves by more than ${fmt(100 * moved / big, 1)}% of the largest (${pathNames.map((n) => `mean ${esc(label(n))} ${fmt(S.means[n][0], 3)} to ${fmt(S.means[n][S.means[n].length - 1], 3)}`).join(", ")}), so they are not plotted.</p></section>`);
  } else if (res.has_means && pathNames.length) {
    out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Mean paths</h2><div class="plot" id="p-means"></div>
      <p class="caption">${links((PRESETS[game] && PRESETS[game].meansCaption) || "The expected states and controls over time: the deterministic part that targets, constant drifts and initial states move.")}</p></section>`);
    const PS = prevResult && prevResult.samples && prevResult.samples.means;
    plotly("newPlot", "p-means", pathNames.map((n) => line(S.mean_t, S.means[n], "mean " + n, ownerColor(res, n),
      res.states.includes(n) ? { line: { color: ownerColor(res, n), width: 2, dash: "dash" } } : {}))
      .concat(PS ? pathNames.filter((n) => PS[n]).map((n) => line(prevResult.samples.mean_t, PS[n], "before", ownerColor(res, n), { line: { color: ownerColor(res, n), width: 1, dash: "dot" }, opacity: 0.45, showlegend: false })) : []),
      baseLayout({ xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
  }
  const vars = res.names.concat(res.definitions || []);
  const kv = vars.includes(keepVar) ? keepVar : (PRESETS[game].defaultVar && vars.includes(PRESETS[game].defaultVar) ? PRESETS[game].defaultVar : vars[0]);
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Shock responses</h2>
    <div class="row"><label for="kvar" class="small muted">Response of</label><select id="kvar">${vars.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("")}</select></div>
    <div class="grid3" id="kgrid"></div>
    <p class="caption">Each curve fixes a date t and shows the response at t to a unit shock that struck at time s ≤ t.${res.kind === "transition" ? " Shocks left of the dotted line struck under the old regime; the band reaches back one past window. An initial shock is a single draw at time 0, plotted at s = 0." : ""} Channels with no response are left out.</p></section>`);
  const drawK = (v) => {
    const box = out.querySelector("#kgrid"); clearPlots(box);
    const tr = res.kind === "transition";
    const smin = Math.min(0, ...S.kernels[v][res.channels[0]].map((cv) => cv.s[0]));
    for (const c of (res.shocks || res.channels)) {
      const curves = S.kernels[v][c];
      if (!curves) continue;
      if (!curves.some((cv) => nonzero(cv.v))) continue;
      const div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      // the dates in the color of whoever the quantity belongs to, later dates darker
      plotly("newPlot", div, curves.map((cv, i) => line(cv.s, cv.v, `t = ${fmt(cv.t, 2)}`, ownerColor(res, v), { opacity: 0.3 + 0.7 * (curves.length > 1 ? i / (curves.length - 1) : 1) })),
        baseLayout({ title: titleOf(chLabel(res, c) + (tr && (res.transition.initial || []).includes(c) ? " (initial shock)" : "")),
          xaxis: { ...baseLayout().xaxis, title: { text: "shock time s" }, range: [smin, T] },
          shapes: tr ? [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: css("--faint"), width: 1, dash: "dot" } }] : [] }), plotCfg);
    }
    if (!box.children.length) box.innerHTML = `<p class="small muted">No channel moves ${esc(label(v))}.</p>`;
  };
  const ksel = out.querySelector("#kvar"); ksel.value = kv; ksel.onchange = () => drawK(ksel.value); drawK(kv);
  renderFoc(res, out, keepCtl, `Split at t = ${fmt(T / 2, 2)} across shock times s${res.kind === "transition" ? " from 0 on; the old regime's shocks are not shown" : ""}. The physical part is what the first-order condition would be if nobody reacted to the player's deviation; the information wedge is the rest, which comes from the other agents revising their forecasts.`, "shock time s");
  if (res.kind !== "transition") renderStrategy(res, out, keepCtl, "shock time s");
}

// ---------------------------------------------------------------------------------------------
// A second model solved beside the first (res.compare): Chapter 6's opaque market beside the transparent one. Costs side
// by side, a few numbers the chapter reads off the two markets, and the two equilibria's responses to each shock.
function renderCompare(res, out) {
  const C = res.compare, def = PRESETS[game] || {}, pal = palette();
  const mainLabel = (def.compare && def.compare.mainLabel) || "This model", cmpLabel = C.label || "Compared";
  if (!C.ok) {
    out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>${esc(mainLabel)} or ${esc(cmpLabel.toLowerCase())}</h2><p class="caption" style="color:var(--warn)">The ${esc(cmpLabel.toLowerCase())} model could not be solved: ${esc(C.error || "unknown error")}</p></section>`);
    return;
  }
  const agents = Object.keys(res.costs);
  const card = (a, v, cmp) => {
    const d = cmp === undefined ? "" : (() => { const dv = v - cmp; return Math.abs(dv) > 5e-5 ? `<span class="delta ${dv > 0 ? "up" : "down"}">${dv > 0 ? "▲" : "▼"} ${fmt(Math.abs(dv), 4)}</span>` : ""; })();
    return `<div class="card"><div class="k">${esc(AGENT_LABEL[a] || a)}</div><div class="v">${fmt(v, 4)} ${d}</div></div>`;
  };
  // the chapter's numbers, where the tab defines them: [label, value in this model, value in the compared one]
  const rows = def.compare && def.compare.rows ? def.compare.rows(res, C) : [];
  const table = rows.length ? `<div class="tablewrap"><table class="diag"><thead><tr><th></th><th>${esc(mainLabel)}</th><th>${esc(cmpLabel)}</th></tr></thead>
    <tbody>${rows.map(([l, a, b, d]) => `<tr><td>${l}</td><td class="mono">${fmt(a, d || 3)}</td><td class="mono">${fmt(b, d || 3)}</td></tr>`).join("")}</tbody></table></div>` : "";
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>${esc(mainLabel)} or ${esc(cmpLabel.toLowerCase())}</h2>
    <div class="versus">
      <div class="side"><h3><svg width="22" height="8" aria-hidden="true" style="vertical-align:middle;margin-right:6px"><line x1="0" y1="4" x2="22" y2="4" stroke="currentColor" stroke-width="2"/></svg>${esc(mainLabel)}</h3><div class="cards">${agents.map((a) => card(a, res.costs[a])).join("")}</div></div>
      <div class="side"><h3><svg width="22" height="8" aria-hidden="true" style="vertical-align:middle;margin-right:6px"><line x1="0" y1="4" x2="22" y2="4" stroke="currentColor" stroke-width="2" stroke-dasharray="5 3"/></svg>${esc(cmpLabel)}</h3><div class="cards">${agents.map((a) => card(a, C.costs[a], res.costs[a])).join("")}</div></div>
    </div>
    ${table}
    <div class="row" style="margin-top:12px"><label for="cvar" class="small muted">Response of</label><select id="cvar"></select></div>
    <div class="plot tall" id="pcompare"></div>
    <p class="caption">${def.compare && def.compare.caption ? def.compare.caption : ""}The arrows compare the ${esc(cmpLabel.toLowerCase())} market's costs with the ${esc(mainLabel.toLowerCase())} one's.
      The plot overlays the two equilibria's responses to each shock (solid ${esc(mainLabel.toLowerCase())}, dashed ${esc(cmpLabel.toLowerCase())}).${C.converged ? "" : ` The ${esc(cmpLabel.toLowerCase())} solve did not converge; its numbers are its last iterate.`}</p></section>`);
  const S = res.samples, T = C.samples;
  const vars = res.names.concat(res.definitions || []).filter((v) => T.kernels[v]);
  const sel = out.querySelector("#cvar");
  sel.innerHTML = vars.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
  sel.value = def.defaultVar && vars.includes(def.defaultVar) ? def.defaultVar : vars[0];
  const draw = () => {
    const v = sel.value, tr = [];
    res.channels.forEach((c, i) => {
      const col = pal[(i + 1) % pal.length];
      if (nonzero(S.kernels[v][c])) tr.push(line(S.age, S.kernels[v][c], chLabel(res, c) + ", " + mainLabel.toLowerCase(), col));
      if (T.kernels[v] && T.kernels[v][c] && nonzero(T.kernels[v][c])) tr.push(line(T.age, T.kernels[v][c], chLabel(res, c) + ", " + cmpLabel.toLowerCase(), col, { line: { color: col, width: 2, dash: "dash" } }));
    });
    plotly("react", "pcompare", tr, baseLayout({ title: titleOf("Response of " + label(v)), xaxis: { ...baseLayout().xaxis, title: { text: "shock age" } } }), plotCfg);
  };
  sel.onchange = draw; draw();
}

// ---------------------------------------------------------------------------------------------
// Chapter 6: a deviation's world (res.deviation, the solver's deviation_response). A unit impulse of one player's control
// at age 0, a blip: the players privy to it (its monitors) respond through their response kernels, the others filter it
// as they filter everything, and the deviator carries on from where the blip left the game. With res.compare, the same
// blip in the compared model, dashed.
function renderDeviation(res, out) {
  const D = res.deviation, def = PRESETS[game] || {}, C = res.compare && res.compare.ok ? res.compare : null;
  const origins = Object.keys(D.origins);
  const choices = origins.flatMap((o) => Object.keys(D.origins[o].controls).map((u) => [o, u]));
  if (!choices.length) return;
  const want = def.deviation ? `${def.deviation.origin}:${def.deviation.control}` : null;
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>A deviation and who sees it</h2>
    <div class="row"><label for="dsel" class="small muted">Blip of</label><select id="dsel">${choices.map(([o, u]) => `<option value="${esc(o + ":" + u)}">${esc(label(u))} (${esc(AGENT_LABEL[o] || o)})</option>`).join("")}</select></div>
    <div class="grid3" id="dgrid"></div><p class="caption" id="dcap"></p></section>`);
  const sel = out.querySelector("#dsel");
  if (want && choices.some(([o, u]) => `${o}:${u}` === want)) sel.value = want;
  const draw = () => {
    const [o, u] = sel.value.split(":"), W = D.origins[o], x = res.samples.age;
    const privy = W.privy.filter((a) => a !== o), others = (res.agents || []).map((a) => a.name).filter((a) => a !== o && !W.privy.includes(a));
    const Wc = C && C.deviation && C.deviation.origins[o] ? C.deviation.origins[o] : null;
    const who = (list) => list.map((a) => AGENT_LABEL[a] || a).join(" and ");
    const main = (def.compare && def.compare.mainLabel) || "", cmp = C ? C.label || "compared" : "";
    let cap = `${esc(AGENT_LABEL[o] || o)} moves ${esc(label(u))} by one unit for an instant at age 0 and then plays on, knowing what it did (the blip). `;
    cap += privy.length ? `${esc(who(privy))} ${privy.length > 1 ? "are" : "is"} privy to it and ${privy.length > 1 ? "respond" : "responds"} to the deviation itself. ` : "";
    cap += others.length ? `${esc(who(others))} ${others.length > 1 ? "do" : "does"} not see it for what it is and ${others.length > 1 ? "filter" : "filters"} its effects as noise. ` : "";
    if (Wc) {
      const pc = Wc.privy.filter((a) => a !== o), oc = (res.agents || []).map((a) => a.name).filter((a) => a !== o && !Wc.privy.includes(a));
      cap += `Dashed, the ${esc(cmp.toLowerCase())} market: ${pc.length ? esc(who(pc)) + " privy" : "nobody privy"}${oc.length ? ", " + esc(who(oc)) + " naive" : ""}. `;
      if (main) cap += `Solid, the ${esc(main.toLowerCase())} one. `;
    }
    cap += "Each panel is the response of one quantity against the time since the blip; a control's own blip is a point mass at age 0, not drawn.";
    if (def.deviation && def.deviation.caption) cap += " " + def.deviation.caption(res, C);
    out.querySelector("#dcap").innerHTML = cap;
    const show = def.deviation && def.deviation.show ? def.deviation.show : res.names;
    const box = out.querySelector("#dgrid"); clearPlots(box);
    for (const q of show) {
      const y = W.controls[u].samples[q], yc = Wc && Wc.controls[u] ? Wc.controls[u].samples[q] : null;
      if (!y || (!nonzero(y) && !(yc && nonzero(yc)))) continue;
      const col = ownerColor(res, q), div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      const tr = [line(x, y, main || label(q), col)];
      if (yc) tr.push(line(C.samples.age, yc, cmp, col, { line: { color: col, width: 2, dash: "dash" } }));
      plotly("newPlot", div, tr, baseLayout({ title: titleOf(label(q)), xaxis: { ...baseLayout().xaxis, title: { text: "time since the blip" } } }), plotCfg);
    }
    if (!box.children.length) box.innerHTML = `<p class="small muted">The blip moves nothing.</p>`;
  };
  sel.onchange = draw; draw();
}

// ---------------------------------------------------------------------------------------------
// Sample paths: the solver returns every primary's kernel on a regular grid (res.paths); the shocks are drawn here,
// so a new draw is instant.  X(t) = mean + sum over shocks and shock cells of K(t, s) dW(s), dW ~ N(0, h).
function rng(seed) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let spare = null;
  return () => { if (spare !== null) { const v = spare; spare = null; return v; }
    let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
    const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
}
function simulatePaths(res, draws, seed) {
  if (res.paths.kind === "state-space") return GaussianPaths.simulate(res.paths, draws, d => rng(seed * 7919 + d));
  const P = res.paths, names = Object.keys(P.kernels), sq = Math.sqrt(P.h);
  const out = {};
  if (P.kind === "stationary") {
    const M = P.cells, n = Math.round(2 * M);                     // two windows of time
    const t = Array.from({ length: n + 1 }, (_, j) => j * P.h);
    const chans = [...new Set(names.flatMap((nm) => Object.keys(P.kernels[nm])))];
    for (const nm of names) out[nm] = { t, draws: [], mean: [], sd: [] };
    for (let d = 0; d < draws; d++) {
      const g = rng(seed * 7919 + d);
      const dW = {}; for (const c of chans) dW[c] = Float64Array.from({ length: n + M + 1 }, () => g() * sq);
      for (const nm of names) {
        const mu = res.has_means ? (res.means[nm] || 0) : 0, y = new Array(n + 1).fill(mu);
        for (const [c, K] of Object.entries(P.kernels[nm])) {
          const w = dW[c];
          for (let j = 0; j <= n; j++) { let acc = 0; for (let i = 0; i < M; i++) acc += K[i] * w[j + M - 1 - i]; y[j] += acc; }
        }
        out[nm].draws.push(y);
      }
    }
    for (const nm of names) {
      let v = 0; for (const K of Object.values(P.kernels[nm])) for (const k of K) v += k * k * P.h;
      const mu = res.has_means ? (res.means[nm] || 0) : 0;
      out[nm].mean = t.map(() => mu); out[nm].sd = t.map(() => Math.sqrt(v));
    }
    return out;
  }
  const t = P.times, nb = P.band, M = t.length - 1;
  const chans = [...new Set(names.flatMap((nm) => Object.keys(P.kernels[nm])))];
  const inits = [...new Set(names.flatMap((nm) => Object.keys(P.initial[nm] || {})))];
  for (const nm of names) {
    const mean = P.means && P.means[nm] ? P.means[nm] : t.map(() => 0);
    const sd = t.map((_, j) => { let v = 0;
      for (const K of Object.values(P.kernels[nm])) for (const k of K[j]) v += k * k * P.h;
      for (const I of Object.values(P.initial[nm] || {})) v += I[j] * I[j];
      return Math.sqrt(v); });
    out[nm] = { t, draws: [], mean, sd };
  }
  for (let d = 0; d < draws; d++) {
    const g = rng(seed * 7919 + d);
    const dW = {}; for (const c of chans) dW[c] = Float64Array.from({ length: nb + M }, () => g() * sq);
    const xi = {}; for (const x of inits) xi[x] = g();
    for (const nm of names) {
      const y = out[nm].mean.slice();
      for (const [c, K] of Object.entries(P.kernels[nm])) {
        const w = dW[c];
        for (let j = 0; j <= M; j++) { const row = K[j]; let acc = 0; for (let i = 0; i < row.length; i++) acc += row[i] * w[i]; y[j] += acc; }
      }
      for (const [x, I] of Object.entries(P.initial[nm] || {})) for (let j = 0; j <= M; j++) y[j] += I[j] * xi[x];
      out[nm].draws.push(y);
    }
  }
  return out;
}
function renderPaths(res, out) {
  const markov = res.paths.kind === "state-space";
  const all = markov ? Object.keys(res.paths.outputs) : Object.keys(res.paths.kernels).filter((nm) => Object.keys(res.paths.kernels[nm]).length || Object.keys((res.paths.initial || {})[nm] || {}).length);
  if (!all.length) return;
  const stat = res.paths.kind === "stationary" || markov;
  const ctl = all.filter((nm) => isControl(nm)), sts = all.filter((nm) => !isControl(nm));
  const many = all.length > 6;
  let names = many ? ctl : sts.concat(ctl);
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Sample paths</h2>
    <div class="row"><button class="secondary" id="redraw">Draw new shocks</button><span class="small muted" id="seedlab"></span>
      ${many ? `<label for="pshow" class="small muted" style="margin-left:12px">Show</label><select id="pshow"><option value="c">controls</option><option value="s">states</option><option value="a">all</option></select>` : ""}</div>
    <div class="legend pathkey"><span><i style="opacity:1;height:2px"></i>draw 1</span><span><i style="opacity:0.55;height:1.5px"></i>draw 2</span><span><i style="opacity:0.3;height:1px"></i>draw 3</span><span><i class="band"></i>&plusmn; 2 sd</span><span><i class="dots"></i>mean</span></div>
    <div class="grid3${all.length === 4 ? " four" : ""}" id="pgrid"></div>
    <p class="caption">Each panel is in its player's color, ink for a state. Three draws of the shocks pushed through the equilibrium${markov ? ". The fundamental starts at zero; filtering errors" + (res.params_used.gamma ? " and inventory start in their stationary distribution" : " start in their stationary distribution, and unpenalized inventory starts at zero") + ". Steps between the plotted dates use the exact Gaussian transition" : stat ? ", over two lag windows of the stationary game" : res.kind === "transition" ? ", old shocks before time 0 included" : ""}. The band is the mean plus and minus two standard deviations. The shocks are drawn in your browser, so a new draw is instant.</p></section>`);
  const draw = () => {
    const sim = simulatePaths(res, 3, pathSeed), box = out.querySelector("#pgrid"); clearPlots(box);
    out.querySelector("#seedlab").textContent = `draw ${pathSeed}`;
    const shade = [[1, 2], [0.55, 1.5], [0.3, 1]];     // draw 1, 2, 3: opacity and width, as in the key above the panels
    for (const nm of names) {
      const S = sim[nm], col = ownerColor(res, nm), div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      const hi = S.mean.map((m, j) => m + 2 * S.sd[j]), lo = S.mean.map((m, j) => m - 2 * S.sd[j]);
      const traces = [
        { x: S.t, y: hi, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
        { x: S.t, y: lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: css("--grid"), name: "± 2 sd", hoverinfo: "skip" },
        line(S.t, S.mean, "mean", css("--faint"), { line: { color: css("--faint"), width: 1, dash: "dot" } }),
        ...S.draws.map((y, d) => line(S.t, y, `draw ${d + 1}`, col, { line: { color: col, width: shade[d % 3][1] }, opacity: shade[d % 3][0] })),
      ];
      plotly("newPlot", div, traces, baseLayout({ title: titleOf(label(nm)), showlegend: false, xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
    }
  };
  out.querySelector("#redraw").onclick = () => { pathSeed++; draw(); };
  const ps = out.querySelector("#pshow");
  if (ps) ps.onchange = () => { names = ps.value === "c" ? ctl : ps.value === "s" ? sts : sts.concat(ctl); draw(); };
  draw();
  if (!stat) renderSurface(res, out, all);
}

// the whole kernel K(t, s) of a finite or transition game as a heat map: a row per date, a column per shock time
function renderSurface(res, out, names) {
  const P = res.paths;
  const pairs = names.flatMap((nm) => Object.keys(P.kernels[nm]).map((c) => [nm, c]));
  if (!pairs.length) return;
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Response surface</h2>
    <div class="row"><label for="svar" class="small muted">Response of</label><select id="svar">${pairs.map(([nm, c], i) => `<option value="${i}">${esc(label(nm))} to ${esc(chLabel(res, c))}</option>`).join("")}</select></div>
    <div class="plot tall" id="psurf"></div>
    <p class="caption">Each row is a date t, each column a shock time s: the color is how much a unit shock at s moves the quantity at t. Gray is where s is after t: a shock cannot move anything before it strikes, so there is nothing to show. Rows of the shock-response plots below are horizontal slices of this picture.${res.kind === "transition" ? " Columns left of s = 0 are the old regime's shocks." : ""} The color is on a square-root scale, so weak responses still show.</p></section>`);
  const draw = (i) => {
    const [nm, c] = pairs[i], rows = P.kernels[nm][c], M = P.times.length - 1, nb = P.band;
    const sgrid = Array.from({ length: nb + M }, (_, k) => (k - nb + 0.5) * P.h);
    const z = rows.map((r) => sgrid.map((_, k) => (k < r.length ? r[k] : null)));
    let mx = 0; for (const r of z) for (const v of r) if (v !== null) mx = Math.max(mx, Math.abs(v));
    // color on a signed square-root scale, so a response a tenth of the peak still shows at a third of the color;
    // the color bar and the hover read the response itself
    const m = mx || 1, sq = (v) => (v === null ? null : Math.sign(v) * Math.sqrt(Math.abs(v) / m));
    const tv = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1].map((f) => f * m);
    plotly("react", "psurf", [{ type: "heatmap", x: sgrid, y: P.times, z: z.map((r) => r.map(sq)), zmin: -1, zmax: 1, colorscale: [[0, css("--c2")], [0.5, isDark() ? "rgb(31,32,34)" : "rgb(255,255,240)"], [1, css("--c1")]], hoverongaps: false,
      customdata: z, hovertemplate: "s %{x:.2f}, t %{y:.2f}: %{customdata:.4g}<extra></extra>",
      colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: css("--muted") }, tickvals: tv.map(sq), ticktext: tv.map((v) => Number(v.toPrecision(2)).toString().replace("-", "−")) } }],
      // the cells with s > t have no value (null); the plot's own background shows through them, set apart from zero
      baseLayout({ showlegend: false, hovermode: "closest", plot_bgcolor: css("--track"), xaxis: { ...baseLayout().xaxis, title: { text: "shock time s" } },
        yaxis: { ...baseLayout().yaxis, title: { text: "date t" } }, margin: { l: 52, r: 12, t: 20, b: 50 } }), plotCfg);
  };
  const want = PRESETS[game] && PRESETS[game].surfaceDefault, i0 = want ? Math.max(0, pairs.findIndex(([nm, c]) => nm === want[0] && c === want[1])) : 0;
  const sel = out.querySelector("#svar"); sel.value = String(i0); sel.onchange = () => draw(+sel.value); draw(i0);
}

// a transition: the loss path against the old and new stationary flows, the forecast errors, the march in T
function renderTransition(res, out) {
  const X = res.transition, agents = Object.keys(res.costs);
  // a game that ends at T (continuation "end") or starts from a prior alone has no stationary flows to compare with
  const flows = X.excess_costs && X.old_flows && X.new_flows && agents.every((a) => X.excess_costs[a] !== undefined);
  const cards = flows ? agents.map((a) => `<div class="card"><div class="k">${esc(AGENT_LABEL[a] || a)}</div>
    <div class="v">${fmt(X.excess_costs[a], 4)}</div><div class="d">excess over the new flow; flows ${fmt(X.old_flows[a], 3)} before, ${fmt(X.new_flows[a], 3)} after</div></div>`).join("") : "";
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>${flows ? "Cost of the transition" : "Over the horizon"}</h2>${flows ? `<div class="cards">${cards}</div>
    <p class="caption">The expected loss over [0, T] minus what the new stationary flow would cost over the same span${res.transition.excess_tail && res.transition.excess_tail.source ? ", with the tail after T extrapolated from the " + esc(res.transition.excess_tail.source) : ""}.
    Positive numbers mean the adjustment costs the player more than starting in the new equilibrium would.</p>` : ""}
    <div class="grid3" style="grid-template-columns: repeat(auto-fit, minmax(300px, 1fr))"><div class="plot" id="p-loss"></div><div class="plot" id="p-belief"></div></div>
    <p class="caption">Left: the expected flow loss E[loss(t)] of each player (solid) between its old and new stationary flows (dotted, dashed).
    Right: each player's forecast error variance of the state, Var(X − E<sub>i</sub>[X]). A sharper signal lowers player 1's error at once; player 2 learns only through the state.</p>
    ${X.march ? `<details style="margin-top:10px"${opts.march ? " open" : ""}><summary>The march in T (${esc(X.march_stop)})</summary>
      <div class="tablewrap"><table class="diag"><thead><tr><th>T</th><th>Gap to the stationary rules</th><th>Rounds</th><th>Unknowns</th><th>Seconds</th></tr></thead><tbody>
      ${X.march.map((r) => `<tr><td class="mono">${fmt(r.T, 2)}</td><td class="mono">${Object.entries(r.gap).map(([a, g]) => `${esc(AGENT_LABEL[a] || a)} ${fmtE(g)}`).join(", ")}</td><td class="mono">${r.evaluations}${r.polish ? " + " + r.polish : ""}</td><td class="mono">${r.unknowns}</td><td class="mono">${fmt(r.seconds, 2)}</td></tr>`).join("")}
      </tbody></table></div><p class="caption">The gap is the relative distance of the best-response rules on the last window from the new stationary ones; the march stops once it is under ${fmtE(X.march_settle)}.</p></details>` : ""}
    </section>`);
  const tr = [], br = [];
  const keep = X.times.map((t, k) => t <= res.T * (1 + 1e-9) ? k : -1).filter((k) => k >= 0);   // [0, T]: the buffer is the continuation's
  const cut = (v) => keep.map((k) => v[k]);
  agents.forEach((a, i) => {
    const c = agentColor(res, a), t = cut(X.times), t0 = t[0], t1 = t[t.length - 1];
    tr.push(line(t, cut(X.loss_path[a]), AGENT_LABEL[a] || a, c));
    if (flows) {
      tr.push(line([t0, t1], [X.old_flows[a], X.old_flows[a]], "before", c, { line: { color: c, width: 1, dash: "dot" }, showlegend: i === 0, hoverinfo: "skip" }));
      tr.push(line([t0, t1], [X.new_flows[a], X.new_flows[a]], "after", c, { line: { color: c, width: 1, dash: "dash" }, showlegend: i === 0, hoverinfo: "skip" }));
    }
    for (const [st, v] of Object.entries((X.belief_error || {})[a] || {})) br.push(line(t.slice(0, -1), cut(v).slice(0, -1), `${AGENT_LABEL[a] || a}, ${st}`, c));   // t < T: the value at T itself is the buffer's
  });
  plotly("newPlot", "p-loss", tr, baseLayout({ title: titleOf("Flow loss"), xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
  if (br.length) plotly("newPlot", "p-belief", br, baseLayout({ title: titleOf("Forecast error variance"), xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
  else $("p-belief").remove();
}

function renderStationary(res, out, keepVar, keepCtl) {
  const S = res.samples;
  if (res.has_means) {
    const rows = res.names.map((n) => `<tr><td>${esc(n)}</td><td class="mono">${fmt(res.means[n], 5)}</td></tr>`).join("");
    out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Means</h2><div class="tablewrap"><table class="diag"><thead><tr><th>Quantity</th><th>Stationary mean</th></tr></thead><tbody>${rows}</tbody></table></div></section>`);
  }
  const vars = res.names.concat(res.definitions || []);
  const kv = vars.includes(keepVar) ? keepVar : (PRESETS[game].defaultVar && vars.includes(PRESETS[game].defaultVar) ? PRESETS[game].defaultVar : vars[0]);
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Shock responses</h2>
    <div class="row"><label for="kvar" class="small muted">Response of</label><select id="kvar">${vars.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("")}</select>
      ${PRESETS[game].channelGroups ? `<label for="kgroup" class="small muted" style="margin-left:12px">to</label><select id="kgroup">${PRESETS[game].channelGroups.map((g, i) => `<option value="${i}">${esc(g[0])}</option>`).join("")}</select>` : ""}</div>
    <div class="plot tall" id="pk"></div>
    <p class="caption">Response to a unit shock as a function of its age: how much of a shock that struck a units of time ago is still present. Channels with no response are left out.${prevResult ? " Dotted: the previous solve, before your last change." : ""}</p></section>`);
  const drawK = (v) => {
    const pal = palette(), G = PRESETS[game].channelGroups, gsel = out.querySelector("#kgroup");
    const shown = (c) => !G || !gsel || G[+gsel.value][1](c);
    const style = (c, i) => (PRESETS[game].channelStyle ? PRESETS[game].channelStyle(res, c) : { color: pal[i % pal.length], dash: "solid" });
    const traces = res.channels.map((c, i) => [c, i]).filter(([c]) => shown(c) && nonzero(S.kernels[v][c]))
      .map(([c, i]) => line(S.age, S.kernels[v][c], chLabel(res, c), style(c, i).color, { line: { ...style(c, i), width: 2 } }));
    // the previous solve's curves, faint, to show what the last change did
    const P = prevResult && prevResult.samples && prevResult.samples.kernels && prevResult.samples.kernels[v];
    if (P) res.channels.forEach((c, i) => { if (shown(c) && P[c] && nonzero(P[c])) traces.unshift(line(prevResult.samples.age, P[c], chLabel(res, c) + " (before)", style(c, i).color, { line: { color: style(c, i).color, width: 1, dash: "dot" }, opacity: 0.5, showlegend: false })); });
    plotly("react", "pk", traces, baseLayout({ title: titleOf("Response of " + label(v)), xaxis: { ...baseLayout().xaxis, title: { text: "shock age" } } }), plotCfg);
  };
  const ksel = out.querySelector("#kvar"); ksel.value = kv; ksel.onchange = () => drawK(ksel.value);
  const gsel = out.querySelector("#kgroup");
  if (gsel) { gsel.value = String(keepGroup); gsel.onchange = () => { keepGroup = +gsel.value; drawK(ksel.value); }; }
  drawK(kv);
  renderFoc(res, out, keepCtl, "Split by shock age. The physical part is what the first-order condition would be if nobody reacted to the agent's deviation; the information wedge is the rest, which comes from the other agents revising their forecasts.", "shock age");
  renderStrategy(res, out, keepCtl, "shock age");
}

function renderFoc(res, out, keepCtl, caption, xlabel) {
  const F = res.samples.foc; const ctls = Object.keys(F); if (!ctls.length) return;
  const def = PRESETS[game] || {};
  const kc = ctls.includes(keepCtl) ? keepCtl : (def.defaultCtl && ctls.includes(def.defaultCtl) ? def.defaultCtl : ctls[0]);
  const x = res.kind !== "stationary" ? null : res.samples.age;
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>First-order condition: physical part and information wedge</h2>
    <div class="row"><label for="fctl" class="small muted">Control</label><select id="fctl">${ctls.map((c) => `<option value="${esc(c)}">${esc(label(c))} (${esc(AGENT_LABEL[F[c].agent] || F[c].agent)})</option>`).join("")}</select></div>
    <div class="grid3" id="fgrid"></div><p class="caption">${caption}</p></section>`);
  const drawF = (ctl) => {
    const box = out.querySelector("#fgrid"); clearPlots(box);
    const f = F[ctl];
    const xs = x || f.s || (() => { const n = f.channels[res.channels[0]].physical.length, t = f.t; return Array.from({ length: n }, (_, i) => t * i / (n - 1)); })();
    for (const c of (res.shocks || res.channels)) {
      const d = f.channels[c];
      if (!d || (d.physical.length !== xs.length)) continue;
      if (!nonzero(d.physical) && !nonzero(d.wedge)) continue;
      const div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      plotly("newPlot", div, [
        line(xs, d.physical, "physical", css("--c3")),
        line(xs, d.wedge, "information wedge", css("--c4"), { line: { color: css("--c4"), width: 2, dash: "dash" } }),
      ], baseLayout({ title: titleOf(chLabel(res, c)), xaxis: { ...baseLayout().xaxis, title: { text: xlabel } } }), plotCfg);
    }
    if (!box.children.length) box.innerHTML = `<p class="small muted">The first-order condition of ${esc(label(ctl))} has no stochastic part.</p>`;
  };
  const fsel = out.querySelector("#fctl"); fsel.value = kc; fsel.onchange = () => drawF(fsel.value); drawF(kc);
}

// ---------------------------------------------------------------------------------------------
// The same action as a rule on the player's noise-state (Remark 1.13): the first-order condition makes the action
// the player's estimate of -(G^DD)^-1 (G^DX X + B' H), so its weight on the estimate of the shock at u is
// D_t(u) = D_W,t(u) - phi_t(u) / G^DD, where D_W is the response to the shocks and phi the first-order-condition
// kernel (both from the solve) and G^DD the Hessian of the player's loss in its own control. Shown only where that
// identity holds: an agent with one control and no delays or leads anywhere in the model.
function lossHessian(model, ctl) {
  const agent = Object.values(model.agents || {}).find((a) => (a.controls || []).includes(ctl));
  if (!agent || agent.controls.length !== 1) return null;
  if (/"(delay|lag|lead|leads|lags|delays)"\s*:\s*(?!0(\.0*)?\s*[,}\]])/.test(JSON.stringify(model))) return null;
  const P = model.params || {};
  const val = (c) => { if (typeof c === "number") return c; try { return Number(new Function("P", "with (Math) { with (P) { return (" + String(c).replace(/\^/g, "**") + "); } }")(P)); } catch (e) { return NaN; } };
  let g = 0;
  for (const term of agent.loss || []) {
    if (!Array.isArray(term) || term.length !== 3) continue;
    const [c, a, b] = term;
    if (typeof a !== "string" || typeof b !== "string" || /[\[(]/.test(a + b)) { if (String(a) + String(b) !== "" && (String(a).includes(ctl) || String(b).includes(ctl))) return null; continue; }
    if (a === ctl && b === ctl) g += 2 * val(c);
  }
  return isFinite(g) && g > 0 ? g : null;
}
function renderStrategy(res, out, keepCtl, xlabel) {
  const F = res.samples.foc; if (!F) return;
  // the model as the solver read it (the grammar, also for a file written as equations), else the page's own
  let model = res.model; if (!model) { try { model = currentModel(); } catch (e) { return; } }
  const stat = res.kind === "stationary";
  // a risk-averse agent's action is the first-order condition at the risk-adjusted noise-state, not this decomposition
  const ok = Object.keys(F).filter((c) => !(res.risk && res.risk[F[c].agent]) && lossHessian(model, c) && F[c].channels && (stat || (res.samples.kernels[c] && res.samples.kernels[c][res.channels[0]].some((q) => Math.abs(q.t - F[c].t) < 1e-9))));
  if (!ok.length) return;
  const kc = ok.includes(keepCtl) ? keepCtl : ok[0];
  const at = stat ? "" : ` at t = ${fmt(F[kc].t, 2)}`;
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>The same action as a rule on the player's estimates</h2>
    <div class="row"><label for="sctl" class="small muted">Control</label><select id="sctl">${ok.map((c) => `<option value="${esc(c)}">${esc(label(c))} (${esc(AGENT_LABEL[F[c].agent] || F[c].agent)})</option>`).join("")}</select></div>
    <div class="grid3" id="sgrid"></div>
    <p class="caption" id="scap"></p></section>`);
  const interp = (xs, ys, x) => { let k = 1; while (k < xs.length - 1 && xs[k] < x) ++k; const w = (x - xs[k - 1]) / ((xs[k] - xs[k - 1]) || 1); return ys[k - 1] + w * (ys[k] - ys[k - 1]); };
  const drawS = (ctl) => {
    const box = out.querySelector("#sgrid"); clearPlots(box);
    const f = F[ctl], g = lossHessian(model, ctl);
    out.querySelector("#scap").innerHTML = `Solid: the action${stat ? "" : at} as a response to each primitive shock, D<sub>W</sub>, the kernel plotted above. Dashed: the weight the action puts on the player's estimate of that shock, D, the strategy on the noise-state of Chapter 1. By Remark 1.13 the action is the player's estimate of its own shadow price, so D = D<sub>W</sub> &minus; &phi; / G<sup>DD</sup>, with &phi; the first-order-condition kernel above and G<sup>DD</sup> = ${fmt(g, 4)} the curvature of the player's loss in its own action.`;
    const xs = stat ? res.samples.age : f.s;
    for (const c of (res.shocks || res.channels)) {
      const d = f.channels[c]; if (!d || !d.foc || d.foc.length !== xs.length) continue;
      let dw;
      if (stat) dw = res.samples.kernels[ctl][c];
      else { const q = (res.samples.kernels[ctl][c] || []).find((z) => Math.abs(z.t - f.t) < 1e-9); if (!q) continue; dw = xs.map((x) => interp(q.s, q.v, x)); }
      if (!dw || dw.length !== xs.length) continue;
      const dn = dw.map((v, i) => v - d.foc[i] / g);
      if (!nonzero(dw) && !nonzero(dn)) continue;
      const div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      plotly("newPlot", div, [
        line(xs, dw, "response to the shock, D<sub>W</sub>", agentColor(res, f.agent)),
        line(xs, dn, "weight on its estimate, D", agentColor(res, f.agent), { line: { color: agentColor(res, f.agent), width: 2, dash: "dash" } }),
      ], baseLayout({ title: titleOf(chLabel(res, c)), xaxis: { ...baseLayout().xaxis, title: { text: xlabel } } }), plotCfg);
    }
    if (!box.children.length) box.innerHTML = `<p class="small muted">${esc(label(ctl))} responds to no shock.</p>`;
  };
  const ssel = out.querySelector("#sctl"); ssel.value = kc; ssel.onchange = () => drawS(ssel.value); drawS(kc);
}

const CHECK_MEANING = {
  converged: "Whether the fixed-point iteration on the best-response map reached its tolerance.",
  resolution: "Whether the strategies are represented accurately on this grid (representation error against its threshold).",
  window: "Whether the stationary kernels (the shock-response curves) have died out before the end of the lag window.",
  second_order: "Whether each agent's best response is a minimum, not only a stationary point (lowest curvature of its loss).",
  refinement: "Whether the costs and the kernels (the shock-response curves) stay put when the game is re-solved on a finer grid.",
  "past window": "Whether the old regime's kernels have died out before the end of its window, the depth of the band.",
  settled: "Whether the best-response rules on the last window of the transition are close to the new stationary ones, so T is long enough.",
  "continuation window": "Whether the new stationary kernels have died out before the end of their window.",
};
function checkName(n) { const [root, who] = n.split(":"); return who ? `${root.replace("_", " ")}, ${AGENT_LABEL[who] || who}` : root.replace("_", " "); }
function checkValue(d) {
  if (d.value !== null && typeof d.value === "object")
    return `costs ${fmtE(d.value.cost_change)} / ${fmtE(d.threshold.cost_change)}<br>kernels ${fmtE(d.value.kernel_change)} / ${fmtE(d.threshold.kernel_change)}<br>at ${d.value.nodes} nodes`;
  return (d.value === null ? "–" : fmtE(d.value)) + (d.threshold === null ? "" : " / " + fmtE(d.threshold));
}
// The best-response map's leading eigenvalues on the complex plane, with the unit circle. Where an eigenvalue sits
// matters, not only how far out: past -1 on the real axis the iteration overshoots with the sign flipped each round,
// and averaging each response with the last (damping by a = 1/2, which sends lambda to 1/2 + lambda/2) brings it
// inside; past +1 it runs the same way every round, and no damping helps. The words follow noisestate's
// diagnostics.classify, and describe best-response iteration near this point, not the equilibrium's stability.
function spectrumPanel(st) {
  const ev = (st.eigenvalues || []).map(([re, im]) => ({ re, im, r: Math.hypot(re, im), dre: 0.5 + 0.5 * re, dim: 0.5 * im }));
  if (!ev.length) return `<p class="small" style="margin:10px 0 0">Stability: spectral radius of the best-response map ${fmt(st.radius, 4)} (${esc(st.method)}, ${st.evaluations} evaluations).</p>`;
  const dom = ev.reduce((m, e) => (e.r > m.r ? e : m), ev[0]);
  const realDom = Math.abs(dom.im) <= 1e-8 * Math.max(1, Math.abs(dom.re));
  const full = st.radius < 1 ? "converges" : realDom && dom.re < -1 ? "oscillates" : "diverges";
  const sampled = Math.max(...ev.map((e) => Math.hypot(e.dre, e.dim))), minR = Math.min(...ev.map((e) => e.r));
  const bound = st.radius < 1 ? Math.max(sampled, 0.5 + 0.5 * st.radius) : st.method === "arnoldi" && minR < 1 ? Math.max(sampled, 0.5 + 0.5 * minR) : null;
  const damped = sampled >= 1 ? "diverges" : bound !== null && bound < 1 ? "converges" : "not certified";
  // the picture: a square view about the origin that holds the unit circle, every eigenvalue and its damped image
  const R = Math.max(1.25, ...ev.map((e) => 1.1 * e.r), ...ev.map((e) => 1.1 * Math.hypot(e.dre, e.dim)));
  const W = 260, c = W / 2, k = (W / 2 - 14) / R, X = (x) => c + k * x, Y = (y) => c - k * y;
  const tick = (v) => `<line x1="${X(v)}" y1="${c - 4}" x2="${X(v)}" y2="${c + 4}" stroke="currentColor" opacity="0.5"/><text x="${X(v)}" y="${c + 16}" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">${v}</text>`;
  const pts = ev.map((e) => {
    const out = e.r >= 1, col = out ? "var(--warn)" : "var(--accent)";
    return `<line x1="${X(e.re)}" y1="${Y(e.im)}" x2="${X(e.dre)}" y2="${Y(e.dim)}" stroke="${col}" stroke-dasharray="2 3" opacity="0.6"/>
      <circle cx="${X(e.dre)}" cy="${Y(e.dim)}" r="4" fill="none" stroke="${col}" stroke-width="1.5"><title>damped: ${fmt(e.dre, 4)} ${e.dim < 0 ? "-" : "+"} ${fmt(Math.abs(e.dim), 4)}i</title></circle>
      <circle cx="${X(e.re)}" cy="${Y(e.im)}" r="4.5" fill="${col}"><title>${fmt(e.re, 4)} ${e.im < 0 ? "-" : "+"} ${fmt(Math.abs(e.im), 4)}i, modulus ${fmt(e.r, 4)}</title></circle>`;
  }).join("");
  const svg = `<svg viewBox="0 0 ${W} ${W}" width="${W}" height="${W}" role="img" aria-label="Leading eigenvalues of the best-response map, with the unit circle" style="flex:none;color:var(--ink-3, #777)">
    <line x1="6" y1="${c}" x2="${W - 6}" y2="${c}" stroke="currentColor" opacity="0.35"/><line x1="${c}" y1="6" x2="${c}" y2="${W - 6}" stroke="currentColor" opacity="0.35"/>
    <circle cx="${c}" cy="${c}" r="${k}" fill="none" stroke="currentColor" stroke-width="1.2"/>
    ${tick(-1)}${tick(1)}<text x="${W - 8}" y="${c - 6}" text-anchor="end" font-size="10" fill="currentColor" opacity="0.7">Re</text><text x="${c + 6}" y="14" font-size="10" fill="currentColor" opacity="0.7">Im</text>
    ${pts}</svg>`;
  const list = ev.map((e) => `${fmt(e.re, 4)}${Math.abs(e.im) > 1e-12 ? ` ${e.im < 0 ? "&minus;" : "+"} ${fmt(Math.abs(e.im), 4)}i` : ""} (|&lambda;| ${fmt(e.r, 4)})`).join(", ");
  return `<div style="display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;margin-top:12px">${svg}
    <div class="small" style="flex:1;min-width:220px">
      <p style="margin:0 0 6px"><b>Stability.</b> Spectral radius of the best-response map ${fmt(st.radius, 4)}. Near this point, best-response iteration <b>${full}</b>;
        damped by a = &frac12; it ${damped === "converges" ? `<b>converges</b> (radius at most ${fmt(bound, 4)})` : damped === "diverges" ? "<b>still diverges</b>" : "is <b>not certified</b> to converge: the eigenvalues shown move inside the circle, but the ones Arnoldi did not return cannot be bounded"}.</p>
      <p style="margin:0 0 6px">Leading eigenvalues (${esc(st.method)}, ${st.evaluations} evaluations): ${list}. Filled points are the eigenvalues, hollow ones their images &frac12; + &frac12;&lambda; under damping.</p>
      <p class="muted" style="margin:0">Inside the unit circle a round of best responses shrinks a deviation. Past &minus;1 on the real axis each round overshoots with the sign flipped, and damping pulls it inside; past +1 each round pushes the same way, and damping cannot. ${st.method === "arnoldi" ? "Arnoldi returns the largest eigenvalues in modulus; the others are smaller." : ""}</p>
    </div></div>`;
}
function renderDiagnostics(res, out) {
  const rows = res.checks.map((d) => `<tr><td>${esc(checkName(d.name))}</td><td><span class="chip ${d.ok === false ? "warn" : "ok"}">${d.ok === false ? "failed" : "passed"}</span></td>
    <td class="mono">${checkValue(d)}</td>
    <td>${esc(d.ok === false ? (d.flag || d.meaning) : (CHECK_MEANING[d.name.split(":")[0]] || d.meaning))}</td></tr>`).join("");
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Diagnostics</h2>
    <p class="small muted" style="margin-top:0">${esc(res.version)}${res.engine !== "ch6-markov" && solverThreads > 1 ? ` · ${solverThreads} threads` : ""} · ${esc(res.message)} · solve ${fmt(res.seconds, 2)} s.
    ${res.engine === "ch6-markov" ? "These checks test the coupled equilibrium equations and the stabilizing branch. The general solver view supplies grid checks and the best-response spectrum." : "A failed check means the numbers may be off in the digits shown here; the row says what to raise."}</p>
    <div class="tablewrap"><table class="diag checks"><thead><tr><th>Check</th><th>Status</th><th>Value / threshold</th><th>What it means</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${res.stability ? spectrumPanel(res.stability) : ""}
    ${res.refinement ? `<p class="small" style="margin:6px 0 0">Refinement: re-solved at ${res.refinement.nodes} nodes in ${fmt(res.refinement.seconds, 1)} s; costs moved ${fmtE(res.refinement.cost_change)}, kernels ${fmtE(res.refinement.kernel_change)}.</p>` : ""}
    ${game !== "custom" ? modelPanel(currentModel()) : ""}</section>`);
}

// ---------------------------------------------------------------------------------------------
$("solvebtn").onclick = () => requestSolve(0);
$("stopbtn").onclick = stopSolve;
// the model as solved and everything the solver returned, for use outside the page
$("savebtn").onclick = () => {
  if (!lastResult) return;
  let model = null;
  try { model = currentModel(); } catch (e) { /* the editor holds an unfinished edit */ }
  const blob = new Blob([JSON.stringify({ model, result: lastResult }, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${(lastResult.name || game).replace(/[^\w.-]+/g, "_")}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
$("sharebtn").onclick = async () => {
  writeHash();
  try { await navigator.clipboard.writeText(location.href); $("sharebtn").textContent = "Copied"; }
  catch (e) { prompt("Copy this link:", location.href); }
  setTimeout(() => { $("sharebtn").textContent = "Copy link"; }, 1500);
};
$("solvecustom").onclick = () => { customYaml = $("yaml").value; renderParamControls(); writeHash(); requestSolve(0); };
$("saveyaml").onclick = () => {
  const text = $("yaml").value, name = ((/^name:\s*(\S+)/m.exec(text) || [])[1] || "model").replace(/[^\w.-]/g, "_");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/yaml" })); a.download = name + ".yaml";
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
$("loadexample").onclick = () => { customYaml = EXAMPLES[$("example").value]; $("yaml").value = customYaml; renderParamControls(); writeHash(); };
$("yaml").addEventListener("input", () => { customYaml = $("yaml").value; });
$("yaml").addEventListener("change", () => { renderParamControls(); writeHash(); });
// the site's theme switch (and the system's) redraws the plots in the new colors
document.body.addEventListener("set-theme", () => setTimeout(() => { if (lastResult) { renderResults(lastResult); plotSnapshot = null; } }, 30));
readHash(); renderAll(); writeHash(); startWorker();
// a link or the back button that changes the hash (writeHash replaces it without firing this) opens that game
window.addEventListener("hashchange", () => {
  const before = game;
  readHash(); renderAll();
  // a new game starts from a clean page; new settings for the same game (a link in a caption) re-solve in place
  if (game !== before) { lastResult = null; clearPlots($("results")); $("savebtn").disabled = true; $("tabs").scrollIntoView({ block: "nearest" }); }
  if (game !== "custom") requestSolve(0); else customReady();
});

// ---------------------------------------------------------------------------------------------
// Sweep: the equilibrium costs as one slider runs across its range, the others held.  A second worker solves the
// points one after another, each from the last one's equilibrium, so the page's own solves are never queued behind it.
var sweepWorker = null, sweepReady = null, sweepReject = null, sweep = null, sweepSeq = 0;   // var: renderAll reads them before this line runs
var SWEEP_POINTS = 11;
function disposeSweepWorker(message) {
  if (sweepWorker) sweepWorker.terminate();
  sweepWorker = null; sweepReady = null;
  if (sweepReject) sweepReject(new Error(message));
  sweepReject = null;
}
function ensureSweepWorker() {
  if (sweepWorker) return sweepReady;
  const sweepUrl = new URL(WORKER_URL, location.href); if (game === "ch6" && opts.reference) sweepUrl.searchParams.set("lazy", "1");
  sweepWorker = new Worker(sweepUrl);
  const activeWorker = sweepWorker;
  sweepReady = new Promise((resolve, reject) => {
    sweepReject = reject;
    const fail = message => {
      if (sweepWorker !== activeWorker) return;
      disposeSweepWorker(message);
      if (sweep && sweep.running) { sweep.running = false; sweep.error = message; updateSweepPanel(); }
    };
    sweepWorker.onmessage = (ev) => {
      if (sweepWorker !== activeWorker) return;
      const m = ev.data;
      if (m.type === "ready") { sweepReject = null; resolve(); }
      else if (m.type === "fatal") fail(m.message);
      else if (m.type === "result") onSweepResult(m);
    };
    sweepWorker.onerror = (e) => fail(e.message || "the sweep worker stopped");
  });
  return sweepReady;
}
function modelWith(over) {
  const saved = values[game];
  values[game] = { ...saved, ...over };
  try { return currentModel(); } finally { values[game] = saved; }
}
function sweepBase(key) { try { return JSON.stringify([modelWith({ [key]: 0 }), currentRequest()]); } catch (e) { return null; } }
function sweepable(g) { return g !== "custom" && g !== "ch5" && PRESETS[g] && PRESETS[g].sliders && PRESETS[g].sliders.length; }
function sweepPoints(s) {
  const out = [];
  for (let i = 0; i < SWEEP_POINTS; ++i) {
    const u = i / (SWEEP_POINTS - 1);
    out.push(s.log ? Math.exp(Math.log(s.min) + (Math.log(s.max) - Math.log(s.min)) * u) : s.min + (s.max - s.min) * u);
  }
  return out;
}
function stopSweep() {
  if (sweepWorker && sweep && sweep.running) disposeSweepWorker("The sweep was stopped");
  if (sweep) sweep.running = false;
}
async function runSweep() {
  if (sweep && sweep.running) { const same = sweep.game === game; stopSweep(); if (same) { updateSweepPanel(); return; } }
  const def = PRESETS[game], s = def.sliders.find((k) => k.key === $("sweepkey").value);
  if (!s) return;
  sweep = { game, key: s.key, label: s.label, log: !!s.log, xs: sweepPoints(s), i: 0, costs: {}, naive: {}, start: lastStart[game] || null, startCompare: lastStartCompare[game] || null,
    values: { ...values[game] }, base: sweepBase(s.key), running: true, id: ++sweepSeq * 100, t0: performance.now(), failed: 0 };
  const S = sweep;
  // Freeze all eleven models before yielding: moving another slider or opening
  // another game must not change what the remaining sweep points mean.
  S.jobs = S.xs.map(x => {
    const model = modelWith({ [S.key]: x });
    const request = { ...currentRequest(), refine: false, stability: false, return_start: true };
    addCompare(request, model, S.game, null);
    delete request.deviation;
    return { model: forSolver(model), request };
  });
  updateSweepPanel();
  try { await ensureSweepWorker(); } catch (e) {
    if (sweep === S && S.running) { S.running = false; S.error = e.message; updateSweepPanel(); }
    return;
  }
  if (sweep !== S || !S.running) return;
  nextSweep();
}
function nextSweep() {
  const S = sweep;
  if (!S || !S.running) return;
  if (S.i >= S.xs.length) { S.running = false; S.wall = (performance.now() - S.t0) / 1000; updateSweepPanel(); return; }
  const { model, request: savedRequest } = S.jobs[S.i], request = { ...savedRequest };
  if (request.compare && S.startCompare) request.compare = { ...request.compare, start: S.startCompare };
  if (S.start) request.start = S.start; else request.start_policy = "coarse";
  sweepWorker.postMessage({ type: "solve", id: S.id + S.i, model, request });
  updateSweepPanel();
}
function onSweepResult(m) {
  const S = sweep;
  if (!S || !S.running || m.id !== S.id + S.i) return;
  const res = JSON.parse(m.result), x = S.xs[S.i], def = PRESETS[S.game];
  if (res.ok && window.siteTally) window.siteTally("solve");
  const extra = def.constCost ? def.constCost({ ...S.values, [S.key]: x }) : {};
  if (res.ok && res.converged) {
    for (const [a, v] of Object.entries(res.costs)) (S.costs[a] = S.costs[a] || []).push([x, v + (extra[a] || 0)]);
    const C = res.compare;
    if (C && C.ok && C.converged) { for (const [a, v] of Object.entries(C.costs)) (S.naive[a] = S.naive[a] || []).push([x, v + (extra[a] || 0)]); if (C.start) S.startCompare = C.start; }
    if (res.start) S.start = res.start;
  } else S.failed++;
  S.i++;
  if (S.game === game) drawSweep();
  nextSweep();
}
function drawSweep() {
  const S = sweep;
  if (!S || S.game !== game || !$("psweep")) return;
  const pal = palette(), agents = Object.keys(S.costs), tr = [];
  agents.forEach((a, i) => {
    const col = pal[(i + 1) % pal.length], P = S.costs[a];
    tr.push({ x: P.map((p) => p[0]), y: P.map((p) => p[1]), name: AGENT_LABEL[a] || a, type: "scatter", mode: "lines+markers", line: { color: col, width: 2 }, marker: { size: 5, color: col } });
    const Q = S.naive[a];
    if (Q && Q.length) tr.push({ x: Q.map((p) => p[0]), y: Q.map((p) => p[1]), name: (AGENT_LABEL[a] || a) + ", " + (((PRESETS[S.game].compare || {}).label) || "compared").toLowerCase(), type: "scatter", mode: "lines+markers", line: { color: col, width: 2, dash: "dash" }, marker: { size: 5, color: col, symbol: "circle-open" } });
  });
  const cur = values[game][S.key], L = baseLayout();
  plotly("react", "psweep", tr, baseLayout({
    title: titleOf("Equilibrium costs as " + S.label + " varies"),
    xaxis: { ...L.xaxis, type: S.log ? "log" : "linear", dtick: S.log ? "D2" : undefined, title: { text: S.label } },
    shapes: [{ type: "line", xref: "x", yref: "paper", x0: cur, x1: cur, y0: 0, y1: 1, line: { color: css("--faint"), width: 1, dash: "dot" } }],
    hovermode: "x unified",
  }), plotCfg);
}
function updateSweepPanel() {
  const panel = $("sweeppanel");
  if (!panel) return;
  const ok = sweepable(game);
  panel.hidden = !ok;
  if (!ok) return;
  const def = PRESETS[game], sel = $("sweepkey");
  const opts_ = def.sliders.filter((s) => !(s.march === false && opts.march));
  const want = sweep && sweep.game === game ? sweep.key : sel.value;
  if (sel.dataset.game !== game) {
    sel.innerHTML = opts_.map((s) => `<option value="${s.key}">${esc(s.label)}</option>`).join("");
    sel.dataset.game = game;
  }
  if (want && opts_.some((s) => s.key === want)) sel.value = want;
  const S = sweep && sweep.game === game ? sweep : null;
  const btn = $("sweepbtn"), note = $("sweepnote"), plot = $("psweep");
  btn.textContent = S && S.running ? "Stop" : S ? "Sweep again" : "Sweep";
  plot.style.display = S ? "" : "none";
  plot.classList.toggle("stale-plot", !!(S && !S.running && S.base !== sweepBase(S.key)));
  if (!S) { note.textContent = `Eleven solves across the slider's range, the others held where they are.`; $("sweepcap").textContent = ""; return; }
  if (S.error) note.textContent = `The sweep stopped: ${S.error}. Press Sweep again to retry.`;
  else if (S.running) note.textContent = `Solving ${Math.min(S.i + 1, S.xs.length)} of ${S.xs.length}…`;
  else if (S.i < S.xs.length) note.textContent = `Stopped after ${S.i} of ${S.xs.length}.`;
  else note.textContent = `${S.xs.length} solves in ${S.wall.toFixed(1)} s${S.failed ? `, ${S.failed} did not converge and are left out` : ""}.`;
  if (!S.running && S.base !== sweepBase(S.key)) note.textContent += " The other parameters have moved since; sweep again to update.";
  $("sweepcap").textContent = `Each point is a full equilibrium; the dotted line is the slider's current value.${PRESETS[game].compare ? ` Solid: the ${(PRESETS[game].compare.mainLabel || "first").toLowerCase()} market; dashed: the ${PRESETS[game].compare.label.toLowerCase()} one.` : ""} Checks are off in the sweep.`;
}
$("sweepbtn").onclick = runSweep;
$("sweepkey").onchange = () => { if (sweep && sweep.game === game && !sweep.running) { sweep = null; } updateSweepPanel(); };
