"use strict";
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
      <div class="eq"><div>dX = (D<sup>1</sup> + D<sup>2</sup>) dt + &sigma; dW<sup>0</sup>, &nbsp; X<sub>0</sub> = 0</div>
      <div>dY<sup>i</sup> = &radic;p<sub>i</sub> X dt + dW<sup>i</sup></div>
      <div>player i minimises E &int;<sub>0</sub><sup>T</sup> [ (X &minus; b<sub>i</sub>)<sup>2</sup> + r<sub>i</sub> (D<sup>i</sup>)<sup>2</sup> ] dt</div></div>
      <p class="small muted" style="margin:0">p<sub>i</sub> is signal precision. Reported costs include the constant b<sub>i</sub><sup>2</sup>T.</p>`,
    yaml: `name: ch1_tracking_with_targets
params: {p1: 9.0, p2: 9.0, r1: 0.1, r2: 0.1, b1: 1.0, b2: -1.0, sigma: 1.0, T: 1.0}
channels: [w0, w1, w2]
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
    nodes: { def: 12, options: [[8, "8: fastest, rough"], [10, "10: quick"], [12, "12: converged at defaults"], [16, "16: fine, slowest"]] },
    constCost: (p) => ({ player1: p.b1 * p.b1 * p.T, player2: p.b2 * p.b2 * p.T }),
    meansCaption: "Expected controls and state over time. With opposite targets the players pull in opposite directions; as precision falls the mean controls approach the open-loop solution, and as it rises they approach the full-information one.",
  },
  ch3: {
    tab: "Stationary tracking",
    ch: "Ch. 3",
    title: "Stationary two-player tracking game",
    desc: `<p class="small muted" style="margin:0">Chapter 3. The same tracking problem run forever, scored by average cost per unit time.
      Responses are kernels in shock age: how a unit shock of a given age still moves the state or a control.</p>
      <div class="eq"><div>dX = (D<sup>1</sup> + D<sup>2</sup>) dt + dW<sup>0</sup></div>
      <div>dY<sup>i</sup> = &radic;p<sub>i</sub> X dt + dW<sup>i</sup></div>
      <div>player i minimises the average of &frac12; X<sup>2</sup> + &frac12; r<sub>i</sub> (D<sup>i</sup>)<sup>2</sup></div></div>`,
    yaml: `name: ch3_stationary_tracking
params: {p1: 3.0, p2: 10.0, r1: 1.0, r2: 1.0}
channels: [w0, w1, w2]
states:
  X: {drift: {D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[0.5, X, X], ["0.5*r1", D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
    loss: [[0.5, X, X], ["0.5*r2", D2, D2]]
horizon: {kind: stationary, discount: 0.0, window: 8.0}
numerics: {nodes: 32}
`,
    sliders: [
      { key: "p1", label: "p₁ precision, player 1", min: 0.1, max: 100, log: true },
      { key: "p2", label: "p₂ precision, player 2", min: 0.1, max: 100, log: true },
      { key: "r1", label: "r₁ effort cost, player 1", min: 0.1, max: 10, log: true },
      { key: "r2", label: "r₂ effort cost, player 2", min: 0.1, max: 10, log: true },
    ],
  },
  ch4: {
    tab: "Kyle–Back market",
    ch: "Ch. 4",
    title: "Stationary Kyle–Back market",
    desc: `<p class="small muted" style="margin:0">Chapter 4. An informed trader watches a noisy signal of a drifting value V and trades against
      noise flow. A competitive market maker sets the price from the order flow alone.</p>
      <div class="eq"><div>dV = &sigma;<sub>V</sub> dW<sup>V</sup>, &nbsp; order flow dZ = D<sup>1</sup> dt + &sigma;<sub>Z</sub> dW<sup>Z</sup></div>
      <div>trader sees dY<sup>1</sup> = &gamma;<sub>1</sub> (V &minus; P) dt + dW<sup>1</sup> and the flow</div>
      <div>trader maximises E &int; e<sup>&minus;&rho;t</sup> [ D<sup>1</sup>(V &minus; P) &minus; &epsilon; (D<sup>1</sup>)<sup>2</sup> ] dt, &nbsp; P = E[V | flow]</div></div>
      <p class="small muted" style="margin:0">Costs are flow losses, so the trader's profit shows as a negative number.
      The market maker's number omits the V<sup>2</sup> term and is not a welfare measure. Small trading costs make the
      fixed point hard to reach; the status bar says when a solve did not converge.</p>`,
    yaml: `name: ch4_kyle_back
params: {eps: 0.2, rho: 0.5, gamma1: 1.0, sigma_V: 1.0, sigma_Z: 1.0}
channels: [wV, wZ, w1]
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
horizon: {kind: stationary, discount: rho, window: 8.0}
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
  },
  ch5: {
    tab: "Supply-chain cycle",
    ch: "Ch. 5",
    title: "Three firms in a supply-chain cycle",
    desc: `<p class="small muted" style="margin:0">Chapter 5. Firm i buys from firm i - 1 and sells to firm i + 1 and to consumers, around a cycle of three.
      Each firm sets a price P<sub>i</sub> and an order o<sub>i</sub>, which take effect after a delay &tau;, and sees only noisy signals:
      its own sales, its supplier's price, its customer's order book and the order upstream. Demand, cost and firm-level shocks drive the market.</p>
      <div class="eq"><div>sales<sub>i</sub> = q + (&theta; &minus; 1) &middot; price index &minus; &theta; P<sub>i</sub> + &eta;<sub>i</sub> &nbsp; (all prices at lag &tau;)</div>
      <div>firm i minimises the average of (price deviation)<sup>2</sup> + m (inventory mismatch)<sup>2</sup> + r o<sub>i</sub><sup>2</sup> + &hellip; &minus; 2&kappa; (revenue terms)</div></div>
      <p class="small muted" style="margin:0">The firms are symmetric, so the solver finds one firm's strategy and relabels it around the cycle.
      This is the heaviest game on the page: a solve takes a few seconds on the quick grid and about half a minute on the dissertation's.</p>`,
    yaml: `name: ch5_cycle_market
params: {theta: 4.0, xi: 0.15, zeta: 0.5, kappa: 0.3, m: 1.0, r: 0.2, rP: 0.0, c: 0.2, sigma_u: 1.0, theta_a: 0.5,
  sigma_a: 1.0, theta_eta: 0.5, sigma_eta: 1.0, s1: 2.5, s2: 0.3, s3: 0.3, s4: 2.0, tau: 0.5}
channels: [w_q, w_a0, w_eta0, w_0_0, w_0_1, w_0_2, w_0_3, w_a1, w_eta1, w_1_0, w_1_1, w_1_2, w_1_3, w_a2, w_eta2,
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
    grid: { key: "window", label: "Lag window", options: [[10, "10: quick, a few seconds"], [24, "24: the dissertation's, about 30 s"]],
      apply: (d, v) => { d.horizon.window = v; d.numerics.unit_range = v >= 16 ? 8 : 4; } },
    defaultVar: "P0", defaultCtl: "P0",
    channelNames: {"w_q": "aggregate demand shock", "w_a0": "cost shock, firm 0", "w_eta0": "demand shock, firm 0", "w_0_0": "sales-signal noise, firm 0", "w_0_1": "price-signal noise, firm 0", "w_0_2": "order-book noise, firm 0", "w_0_3": "upstream-order noise, firm 0", "w_a1": "cost shock, firm 1", "w_eta1": "demand shock, firm 1", "w_1_0": "sales-signal noise, firm 1", "w_1_1": "price-signal noise, firm 1", "w_1_2": "order-book noise, firm 1", "w_1_3": "upstream-order noise, firm 1", "w_a2": "cost shock, firm 2", "w_eta2": "demand shock, firm 2", "w_2_0": "sales-signal noise, firm 2", "w_2_1": "price-signal noise, firm 2", "w_2_2": "order-book noise, firm 2", "w_2_3": "upstream-order noise, firm 2"},
  },
  ch6: {
    tab: "Naive or privy",
    ch: "Ch. 6",
    title: "Naive distortions and privy responses",
    desc: `<p class="small muted" style="margin:0">Chapter 6. The Kyle–Back market of Chapter 4, solved twice. In the privy equilibrium the
      trader's best response accounts for how the market maker's price reacts to its orders, as it does in Chapter 4. In the naive one
      the trader treats the market maker as naive to its deviations: in its first-order condition the price does not move when it trades
      more, so it trades as if it had no price impact, while the market maker still prices the trader's actual strategy.</p>
      <div class="eq"><div>privy: the trader's deviation moves the flow, the market maker's forecast and the price, and the trader pays for it</div>
      <div>naive: the same deviation, with the market maker's strategy switched off in the trader's calculation</div></div>
      <p class="small muted" style="margin:0">Both are solved on every change (the naive one starts from the privy equilibrium). The comparison below
      shows what the naive belief costs the trader: it trades harder on its information, the price becomes more informative
      faster, and its profit turns into a loss.</p>`,
    yaml: `name: ch6_naive_kyle_back
params: {eps: 0.2, rho: 0.5, gamma1: 1.0, sigma_V: 1.0, sigma_Z: 1.0}
channels: [wV, wZ, w1]
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
horizon: {kind: stationary, discount: rho, window: 8.0}
numerics: {nodes: 16}
`,
    naive: { trader1: ["market_maker"] },
    sliders: [
      { key: "eps", label: "ε trading cost", min: 0.05, max: 2, log: true },
      { key: "gamma1", label: "γ₁ signal loading", min: 0.1, max: 4, log: true },
      { key: "sigma_Z", label: "σ_Z noise-flow volatility", min: 0.2, max: 3, step: 0.05 },
      { key: "rho", label: "ρ discount rate", min: 0.1, max: 2, step: 0.05 },
    ],
    defaultVar: "D1", defaultCtl: "D1",
    channelNames: { wV: "value shock", wZ: "noise-trader flow shock", w1: "trader's signal noise" },
  },
  tr: {
    tab: "Regime change",
    ch: "Ch. 3",
    title: "A change of regime in the tracking game",
    desc: `<p class="small muted" style="margin:0">Chapter 3's tracking game, with mean reversion, has run in a stationary equilibrium for ever.
      At time 0 player 1's signal precision jumps. The shocks born before 0 still drive the state and both players' forecasts,
      so the players move from the old equilibrium toward the new one. After T the new stationary equilibrium takes over.</p>
      <div class="eq"><div>dX = (&minus;a X + D<sup>1</sup> + D<sup>2</sup>) dt + dW<sup>0</sup></div>
      <div>dY<sup>i</sup> = &radic;p<sub>i</sub>(t) X dt + dW<sup>i</sup>, &nbsp; p<sub>1</sub>(t) = p<sub>1</sub><sup>before</sup> for t &lt; 0, p<sub>1</sub> after</div>
      <div>player i minimises E &int;<sub>0</sub><sup>T</sup> [ &frac12; X<sup>2</sup> + &frac12; r<sub>i</sub> (D<sup>i</sup>)<sup>2</sup> ] dt, then the new stationary flow</div></div>
      <p class="small muted" style="margin:0">The strip carries the old shocks on a band of depth L = 3 below s = 0.
      "Until settled" lets the solver pick T: it marches T = 0, 3, 6, &hellip; until the best-response rules on the last window are within 2% of the new stationary ones.</p>`,
    yaml: `name: regime_change
params: {p1: 6.0, p2: 3.0, r1: 1.0, r2: 1.0, a: 1.0, T: 6.0}
channels: [w0, w1, w2]
states:
  X: {drift: {X: "-a", D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[0.5, X, X], ["0.5*r1", D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
    loss: [[0.5, X, X], ["0.5*r2", D2, D2]]
horizon:
  kind: transition
  T: T
  past:
    model:
      name: before
      params: {p1: 1.0, p2: 3.0, r1: 1.0, r2: 1.0, a: 1.0}
      channels: [w0, w1, w2]
      states:
        X: {drift: {X: "-a", D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
      agents:
        player1:
          controls: [D1]
          signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
          loss: [[0.5, X, X], ["0.5*r1", D1, D1]]
        player2:
          controls: [D2]
          signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
          loss: [[0.5, X, X], ["0.5*r2", D2, D2]]
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
      { key: "T", label: "T end of the transition", min: 3, max: 12, step: 0.5, march: false },
    ],
    nodes: { def: 8, options: [[5, "5: fastest, rough"], [6, "6: quick"], [8, "8: about a second"], [10, "10: fine, a few seconds"]] },
    march: true,
    defaultVar: "D1", defaultCtl: "D1",
    meansCaption: "",
  },
  custom: {
    tab: "Your model",
    title: "Your model",
    desc: `<p class="small muted" style="margin:0">Write or paste a model file, or start from one of the examples of the noisestate package.
      The solver takes any model the package's grammar allows: stationary, finite (spectral or cells engine) and transition
      horizons, with the past model written inline under horizon.past.model and an optional horizon.settle for the march in T.</p>`,
  },
};
const EXAMPLES = {
  "tracking game with targets (Ch. 1)": PRESETS.ch1.yaml,
  "stationary tracking (Ch. 3)": PRESETS.ch3.yaml,
  "Kyle–Back market (Ch. 4)": PRESETS.ch4.yaml,
  "change of regime (Ch. 3 transition)": PRESETS.tr.yaml,
  "transition shorter than the past's window, initial shock": `# T = 1 < L = 2: the old shocks stay alive on the buffer; an initial shock xi
# loads on the state and player 2 sees it at once; player 1 is myopic.
name: tr_short
params: {p1: 10.0, p2: 3.0, r1: 1.0, r2: 1.0}
channels: [w0, w1, w2]
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
      channels: [w0, w1, w2]
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
channels: [wZ]
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
  "tracking with naive observers (Ch. 6)": `# Each player's best response ignores the other's reaction to its deviations
# (naive_observers: agent -> the observers treated as naive). Compare with the stationary tracking example.
name: ch6_naive_tracking
params: {p1: 3.0, p2: 10.0, r1: 1.0, r2: 1.0}
channels: [w0, w1, w2]
states:
  X: {drift: {D1: 1.0, D2: 1.0}, noise: {w0: 1.0}}
agents:
  player1:
    controls: [D1]
    signals: {y1: {drift: {X: "sqrt(p1)"}, noise: {w1: 1.0}}}
    loss: [[0.5, X, X], ["0.5*r1", D1, D1]]
  player2:
    controls: [D2]
    signals: {y2: {drift: {X: "sqrt(p2)"}, noise: {w2: 1.0}}}
    loss: [[0.5, X, X], ["0.5*r2", D2, D2]]
naive_observers: {player1: [player2], player2: [player1]}
horizon: {kind: stationary, discount: 0.0, window: 8.0}
numerics: {nodes: 32}
`,
  "tracking game on the cells engine (Ch. 1)": PRESETS.ch1.yaml.replace("numerics: {nodes: 12}", "numerics: {engine: cells, nodes: 24}"),
  "delayed control and observation (Ch. 1), a few seconds": `# Finite-horizon two-player game with a control delay and a delayed observation.
name: ch1_delayed_finite
params: {p1: 3.0, p2: 3.0, r1: 0.1, r2: 0.1, sigma: 1.0, tau: 0.25}
channels: [w0, w1, w2]
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
channels: [w0, w1, w2]
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

const NAME_LABEL = { X: "state X", V: "value V", P: "price P" };
const CHANNEL_LABEL = { w0: "common shock W⁰", w1: "signal noise W¹", w2: "signal noise W²", w3: "signal noise W³" };
const AGENT_LABEL = { player1: "Player 1", player2: "Player 2", market_maker: "Market maker", trader1: "Informed trader", firm0: "Firm 0", firm1: "Firm 1", firm2: "Firm 2" };

// ---------------------------------------------------------------------------------------------
// State
let game = "ch1";
const values = {};              // preset -> {slider key: value, nodes}
const opts = { refine: false, stability: false, engine: "spectral", march: false };
const lastStart = {};
let solverThreads = 1;
let prevResult = null;            // the result before the last change, drawn faintly for comparison
let pathSeed = 1;             // preset -> the raw maps of its last equilibrium
let customYaml = "";
let worker = null, workerReady = false;
let reqId = 0, inFlight = null, pending = false, debounce = null, lastResult = null;
let solveStart = 0, timerHandle = null, progress = null;

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => (x === null || x === undefined || !isFinite(x)) ? "–" : Number(x).toFixed(d);
const fmtE = (x) => (x === null || x === undefined || !isFinite(x)) ? "–" : Number(x).toExponential(1);
// the explorer's colours live on its root element (they follow the site's light and dark themes)
const css = (v) => getComputedStyle(document.querySelector(".explorer") || document.documentElement).getPropertyValue(v).trim();
const isDark = () => document.documentElement.classList.contains("dark");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function presetModel(g) { return jsyaml.load(PRESETS[g].yaml); }
// a slider's parameter lives in the model's params ("now"), the past model's ("past"), or both
function sliderParams(d, s) {
  const where = s.where || "now", out = [];
  if (where !== "past") out.push(d.params);
  if (where !== "now" && d.horizon.past && d.horizon.past.model) out.push(d.horizon.past.model.params);
  return out;
}
function isControl(name) { return !!(lastResult && lastResult.agents && lastResult.agents.some((a) => a.controls.includes(name))); }
function label(name) { return NAME_LABEL[name] || (isControl(name) || name.match(/^[A-Z]\d*$/) ? `control ${name}` : name); }
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
    for (const s of PRESETS[g].sliders) values[g][s.key] = sliderParams(m, s)[0][s.param || s.key];
  }
  opts.refine = h.get("refine") === "1"; opts.stability = h.get("stability") === "1";
  opts.engine = h.get("engine") === "cells" ? "cells" : "spectral"; opts.march = h.get("march") === "1";
  customYaml = PRESETS.ch1.yaml;
  if (game === "custom" && h.get("model")) {
    try { customYaml = b64decode(h.get("model")); } catch (e) { /* keep the default */ }
  } else if (game !== "custom") {
    for (const s of PRESETS[game].sliders) {
      const v = parseFloat(h.get(s.key));
      if (isFinite(v)) values[game][s.key] = Math.min(s.max, Math.max(s.min, v));
    }
    const n = parseInt(h.get("nodes"));
    if (PRESETS[game].nodes && PRESETS[game].nodes.options.some((o) => o[0] === n)) values[game].nodes = n;
    const G = PRESETS[game].grid;
    if (G) { const w = parseFloat(h.get(G.key)); if (G.options.some((o) => o[0] === w)) values[game][G.key] = w; }
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
    h = new URLSearchParams({ game, ...v });
  }
  if (opts.refine) h.set("refine", "1");
  if (opts.stability) h.set("stability", "1");
  if (opts.engine === "cells" && game === "ch1") h.set("engine", "cells");
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
function showVal(p, v) { const d = Math.abs(v) >= 10 ? 1 : Math.abs(v) >= 1 ? 2 : 3; return Number(v).toFixed(p.step && p.step >= 0.25 ? 2 : d); }

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
      game = g; renderAll(); writeHash(); lastResult = null; $("results").innerHTML = "";
      if (g !== "custom") requestSolve(0); else setStatus("idle", "Ready", "Edit the model and press Solve.");
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
  const box = $("controls"); box.innerHTML = "";
  $("editor").hidden = game !== "custom";
  if (game === "custom") { renderEditor(); renderOptions($("paramopts")); return; }
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
  if (def.grid) {
    const G = def.grid, wrap = document.createElement("div"); wrap.className = "ctl";
    wrap.innerHTML = `<label for="ctl-grid"><span>${G.label}</span></label><select id="ctl-grid">${G.options.map((o) => `<option value="${o[0]}">${o[1]}</option>`).join("")}</select>
      <div class="hint">A longer window resolves the slow-decaying responses; it costs time.</div>`;
    box.appendChild(wrap);
    const sel = wrap.querySelector("select"); sel.value = values[game][G.key];
    sel.addEventListener("change", () => { values[game][G.key] = +sel.value; writeHash(); requestSolve(0); });
  }
  if (def.nodes) {
    const wrap = document.createElement("div"); wrap.className = "ctl";
    wrap.innerHTML = `<label for="ctl-nodes"><span>Grid (nodes per side)</span></label><select id="ctl-nodes">${def.nodes.options.map((o) => `<option value="${o[0]}">${o[1]}</option>`).join("")}</select>
      <div class="hint">More nodes are more accurate and slower.</div>`;
    box.appendChild(wrap);
    const sel = wrap.querySelector("select"); sel.value = values[game].nodes;
    sel.addEventListener("change", () => { values[game].nodes = +sel.value; writeHash(); requestSolve(0); });
  }
  renderOptions(box);
}

// solver options: the end of a transition, the engine of the finite game, and the after-solve checks
function renderOptions(box) {
  const def = PRESETS[game];
  const wrap = document.createElement("div"); wrap.className = "ctl opts";
  let html = "";
  if (def.march) html += `<label class="pick"><span>End of the transition</span><select id="opt-march">
      <option value="0">Fixed T (the slider)</option><option value="1">Until settled (march in T)</option></select></label>`;
  if (game === "ch1") html += `<label class="pick"><span>Engine</span><select id="opt-engine">
      <option value="spectral">Spectral (triangle grid)</option><option value="cells">Cells (forward march)</option></select></label>`;
  html += `<label class="check" title="Re-solve on a grid 1.5 times finer and report how much costs and kernels move"><input type="checkbox" id="opt-refine"> Refinement check</label>
      <label class="check" title="The spectral radius of the best-response map"><input type="checkbox" id="opt-stability"> Stability</label>
      <span class="hint">Checks run after the solve and add time.</span>`;
  wrap.innerHTML = html; box.appendChild(wrap);
  const on = (id, f) => { const e = wrap.querySelector("#" + id); if (e) e.addEventListener("change", () => { f(e); writeHash(); requestSolve(0); }); return e; };
  const m = on("opt-march", (e) => { opts.march = e.value === "1"; renderControls(); });
  if (m) m.value = opts.march ? "1" : "0";
  const g = on("opt-engine", (e) => { opts.engine = e.value; });
  if (g) g.value = opts.engine;
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
  if (!d || typeof d !== "object") throw new Error("The model file must be a mapping (name, channels, states, agents, horizon).");
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
  if (reset || !timerHandle) { solveStart = performance.now(); progress = null; }
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

// ---------------------------------------------------------------------------------------------
// Worker
function startWorker() {
  workerReady = false;
  try { worker = new Worker("worker.js"); }
  catch (e) { setStatus("bad", "Failed", "This browser could not start a Web Worker: " + e.message); return; }
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === "ready") {
      workerReady = true; solverThreads = m.threads || 1;
      const tf = document.getElementById("threadfact");
      if (tf) tf.textContent = solverThreads > 1 ? `${solverThreads} threads in this browser` : "single-threaded in this browser";
      if (game === "custom" && !inFlight && !lastResult) setStatus("idle", "Ready", "Edit the model and press Solve.");
      else requestSolve(0);
    } else if (m.type === "progress") {
      if (inFlight && m.id === inFlight.id) progress = m;
    } else if (m.type === "result") {
      onSolved(m);
    } else if (m.type === "fatal") {
      setStatus("bad", "Failed", "The solver could not load: " + m.message);
    }
  };
  worker.onerror = (e) => { setStatus("bad", "Failed", "The solver stopped: " + (e.message || "unknown error") + ". Reload the page."); };
}
function stopSolve() {
  if (!inFlight) return;
  worker.terminate(); inFlight = null; pending = false; stopTimer();
  $("stopbtn").disabled = true; $("solvebtn").disabled = false;
  setStatus("warn", "Stopped", "The solve was stopped. Change a parameter or press Solve again.");
  startWorker();
}

function currentModel() {
  if (game === "custom") return customModel();
  const def = PRESETS[game], d = presetModel(game);
  for (const s of def.sliders) for (const P of sliderParams(d, s)) P[s.param || s.key] = values[game][s.key];
  if (def.nodes) d.numerics.nodes = values[game].nodes;
  if (def.grid) def.grid.apply(d, values[game][def.grid.key]);
  if (game === "ch1" && opts.engine === "cells") { d.numerics.engine = "cells"; d.numerics.nodes = 2 * values[game].nodes; }
  if (def.march && opts.march) { delete d.horizon.T; delete d.params.T; d.horizon.settle = 0.02; }
  return d;
}

function currentRequest() { return { refine: opts.refine, stability: opts.stability }; }
function requestSolve(delay) {
  clearTimeout(debounce);
  if (lastResult) $("results").classList.add("stale");
  debounce = setTimeout(() => {
    if (!workerReady) return;
    if (inFlight) { pending = true; startTimer(false); return; }
    sendSolve();
  }, delay);
}
function sendSolve() {
  let model;
  try { model = currentModel(); }
  catch (e) { $("yamlerror").textContent = e.message; setStatus("bad", "Error", e.message); return; }
  $("yamlerror").textContent = "";
  inFlight = { id: ++reqId, game, key: JSON.stringify([model, currentRequest()]) };
  pending = false;
  $("solvebtn").disabled = true; $("stopbtn").disabled = false;
  startTimer();
  // a warm start: the last equilibrium of this tab, which the solver uses when the shapes match (a parameter moved)
  // and ignores otherwise (a new grid or a new model)
  const request = { ...currentRequest(), return_start: true };
  if (game !== "custom" && PRESETS[game].naive) request.naive_compare = PRESETS[game].naive;
  const hk = model.horizon && model.horizon.kind;
  if (!(model.numerics && model.numerics.engine === "cells")) request.path_grid = hk === "stationary" ? 150 : hk === "transition" ? 48 : 60;
  if (lastStart[game]) request.start = lastStart[game];
  else if (!(model.numerics && model.numerics.engine === "cells")) request.start_policy = "coarse";   // a cold solve starts from the same model on a coarser grid (the cell engine has none)
  worker.postMessage({ type: "solve", id: inFlight.id, model, request });
}
function onSolved(m) {
  const req = inFlight; inFlight = null; stopTimer();
  $("solvebtn").disabled = false; $("stopbtn").disabled = true;
  let key = null;
  try { key = JSON.stringify([currentModel(), currentRequest()]); } catch (e) { /* the editor holds an unfinished edit */ }
  const stale = pending || !req || req.game !== game || (key !== null && key !== req.key);
  const res = JSON.parse(m.result);
  if (!res.ok) {
    if (req && req.game === game) {
      setStatus("bad", "Error", res.error);
      if (game === "custom") $("yamlerror").textContent = res.error;
      $("results").classList.remove("stale");
    }
    if (stale && req && req.game === game) sendSolve();
    return;
  }
  if (res.start && req) { lastStart[req.game] = res.start; delete res.start; }
  if (req && req.game === game) {
    prevResult = lastResult && lastResult.name === res.name && lastResult.kind === res.kind ? lastResult : null;
    lastResult = res;
    try { renderResults(res); updateSweepPanel(); drawSweep(); }
    catch (e) { console.error(e); setStatus("warn", "Solved, drawing failed", "The solve finished but a plot could not be drawn: " + e.message); $("results").classList.remove("stale"); return; }
  }
  if (stale) { sendSolve(); return; }
  $("results").classList.remove("stale");
  const failed = res.checks.filter((d) => d.ok === false && d.name !== "converged");
  const t = m.wall < 0.1 ? "under 0.1" : m.wall.toFixed(1), how = res.warm_start ? " from the last equilibrium" : "";
  if (!res.converged) setStatus("bad", "Not converged", `The fixed point did not converge (residual ${fmtE(res.residual)} after ${res.evaluations} rounds). Try a finer grid or less extreme parameters.`);
  else if (failed.length && failed.every((d) => d.name === "resolution"))
    setStatus("warn", "Solved, coarse grid", `Solved in ${t} s${how}. The grid is coarse for these parameters (representation error ${fmtE(failed[0].value)}): costs are good to a few digits; more nodes give more.`);
  else if (failed.length) setStatus("warn", "Converged, with warnings", `Solved in ${t} s${how}. Failed: ${failed.map((d) => d.name).join(", ")}; the Diagnostics table says what to raise.`);
  else setStatus("ok", "Solved", `Solved in ${t} s${how}, ${res.evaluations} best-response rounds, residual ${fmtE(res.residual)}. All checks passed.`);
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
const plotCfg = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };
const titleOf = (s) => ({ text: s, font: { size: 13 }, x: 0, xanchor: "left", xref: "paper" });
const palette = () => ["--c3", "--c1", "--c2", "--c4", "--c5", "--c6"].map(css);
function ramp(i, n) {
  const dark = isDark();
  const a = n > 1 ? i / (n - 1) : 1;
  const from = dark ? [60, 90, 120] : [170, 200, 225], to = dark ? [140, 200, 245] : [20, 70, 120];
  return `rgb(${from.map((f, k) => Math.round(f + (to[k] - f) * a)).join(",")})`;
}
const line = (x, y, name, color, extra) => Object.assign({ x, y, name, type: "scatter", mode: "lines", line: { color, width: 2 } }, extra || {});
const nonzero = (arr) => arr.some((v) => v !== null && Math.abs(v) > 1e-12);

function renderResults(res) {
  const out = $("results");
  const keepVar = out.querySelector("#kvar")?.value, keepCtl = out.querySelector("#fctl")?.value;
  out.innerHTML = "";
  const def = PRESETS[game];
  const params = res.params_used || (game !== "custom" ? values[game] : {});
  const cards = Object.entries(res.costs).map(([a, v]) => {
    let shown = v, note = res.cost_kind;
    const extra = def.constCost ? def.constCost(values[game])[a] : 0;
    if (def.constCost) { shown = v + extra; note = "expected cost over [0, T]"; }
    const parts = res.cost_parts[a];
    const split = parts && Math.abs(parts.mean) > 1e-12 ? `variance ${fmt(parts.variance, 3)}, mean ${fmt(parts.mean + extra, 3)}` : "";
    const before = prevResult && prevResult.costs[a] !== undefined ? prevResult.costs[a] + extra : null;
    const dv = before === null ? 0 : shown - before;
    const delta = before !== null && Math.abs(dv) > 5e-5 ? `<span class="delta ${dv > 0 ? "up" : "down"}">${dv > 0 ? "▲" : "▼"} ${fmt(Math.abs(dv), 4)}</span>` : "";
    return `<div class="card"><div class="k">${esc(AGENT_LABEL[a] || a)}</div><div class="v">${fmt(shown, 4)} ${delta}</div><div class="d">${split || note}</div></div>`;
  }).join("");
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Equilibrium costs</h2><div class="cards">${cards}</div>
    <p class="caption">Expected losses at the equilibrium (${esc(res.cost_kind)}); smaller is better.${prevResult ? " Arrows: the change from the previous solve." : ""}</p>
    ${res.warnings && res.warnings.length ? `<p class="caption" style="color:var(--warn)">${res.warnings.map(esc).join("<br>")}</p>` : ""}</section>`);
  if (res.kind === "transition") renderTransition(res, out);
  if (res.naive) renderNaive(res, out);
  if (res.paths) renderPaths(res, out);
  if (res.kind.startsWith("finite") || res.kind === "transition") renderFinite(res, out, keepVar, keepCtl);
  else renderStationary(res, out, keepVar, keepCtl);
  renderDiagnostics(res, out);
}

function renderFinite(res, out, keepVar, keepCtl) {
  const S = res.samples, T = res.T;
  const pathNames = res.names.filter((n) => nonzero(S.means[n] || []));
  if (res.has_means && pathNames.length) {
    out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Mean paths</h2><div class="plot" id="p-means"></div>
      <p class="caption">${esc((PRESETS[game] && PRESETS[game].meansCaption) || "The expected states and controls over time: the deterministic part that targets, constant drifts and initial states move.")}</p></section>`);
    const pal = palette();
    const PS = prevResult && prevResult.samples && prevResult.samples.means;
    Plotly.newPlot("p-means", pathNames.map((n, i) => line(S.mean_t, S.means[n], "mean " + n, pal[(i + 1) % pal.length],
      res.states.includes(n) ? { line: { color: pal[(i + 1) % pal.length], width: 2, dash: "dash" } } : {}))
      .concat(PS ? pathNames.filter((n) => PS[n]).map((n, i) => line(prevResult.samples.mean_t, PS[n], "before", pal[(i + 1) % pal.length], { line: { color: pal[(i + 1) % pal.length], width: 1, dash: "dot" }, opacity: 0.45, showlegend: false })) : []),
      baseLayout({ xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
  }
  const vars = res.names.concat(res.definitions || []);
  const kv = vars.includes(keepVar) ? keepVar : (PRESETS[game].defaultVar && vars.includes(PRESETS[game].defaultVar) ? PRESETS[game].defaultVar : vars[0]);
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Shock responses</h2>
    <div class="row"><label for="kvar" class="small muted">Response of</label><select id="kvar">${vars.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("")}</select></div>
    <div class="grid3" id="kgrid"></div>
    <p class="caption">Each curve fixes a date t and shows the response at t to a unit shock that struck at time s ≤ t.${res.kind === "transition" ? " Shocks left of the dotted line struck under the old regime; the band reaches back one past window. An initial shock is a single draw at time 0, plotted at s = 0." : ""} Channels with no response are left out.</p></section>`);
  const drawK = (v) => {
    const box = out.querySelector("#kgrid"); box.innerHTML = "";
    const tr = res.kind === "transition";
    const smin = Math.min(0, ...S.kernels[v][res.channels[0]].map((cv) => cv.s[0]));
    for (const c of (res.shocks || res.channels)) {
      const curves = S.kernels[v][c];
      if (!curves) continue;
      if (!curves.some((cv) => nonzero(cv.v))) continue;
      const div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      Plotly.newPlot(div, curves.map((cv, i) => line(cv.s, cv.v, `t = ${fmt(cv.t, 2)}`, ramp(i, curves.length))),
        baseLayout({ title: titleOf(chLabel(res, c) + (tr && (res.transition.initial || []).includes(c) ? " (initial shock)" : "")),
          xaxis: { ...baseLayout().xaxis, title: { text: "shock time s" }, range: [smin, T] },
          shapes: tr ? [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: css("--faint"), width: 1, dash: "dot" } }] : [] }), plotCfg);
    }
    if (!box.children.length) box.innerHTML = `<p class="small muted">No channel moves ${esc(label(v))}.</p>`;
  };
  const ksel = out.querySelector("#kvar"); ksel.value = kv; ksel.onchange = () => drawK(ksel.value); drawK(kv);
  renderFoc(res, out, keepCtl, `Split at t = ${fmt(T / 2, 2)} across shock times s${res.kind === "transition" ? ", old shocks included" : ""}. The physical part is what the first-order condition would be if nobody reacted to the player's deviation; the information wedge is the rest, which comes from the other agents revising their forecasts.`, "shock time s");
}

// ---------------------------------------------------------------------------------------------
// Chapter 6: the privy equilibrium (the result) against the same game with naive observers (res.naive)
function renderNaive(res, out) {
  const N = res.naive, agents = Object.keys(res.costs), pal = palette();
  const who = Object.entries(N.naive_observers).map(([a, obs]) => `${AGENT_LABEL[a] || a} treats ${obs.map((o) => AGENT_LABEL[o] || o).join(" and ")} as naive`).join("; ");
  const card = (a, v, cmp) => {
    const d = cmp === undefined ? "" : (() => { const dv = v - cmp; return Math.abs(dv) > 5e-5 ? `<span class="delta ${dv > 0 ? "up" : "down"}">${dv > 0 ? "▲" : "▼"} ${fmt(Math.abs(dv), 4)}</span>` : ""; })();
    return `<div class="card"><div class="k">${esc(AGENT_LABEL[a] || a)}</div><div class="v">${fmt(v, 4)} ${d}</div></div>`;
  };
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Privy or naive</h2>
    <div class="versus">
      <div class="side"><h3><span class="swatch" style="background:${pal[1]}"></span>Privy (the equilibrium of Chapter 4)</h3><div class="cards">${agents.map((a) => card(a, res.costs[a])).join("")}</div></div>
      <div class="side"><h3><span class="swatch" style="background:${pal[2]}"></span>Naive: ${esc(who)}</h3><div class="cards">${agents.map((a) => card(a, N.costs[a], res.costs[a])).join("")}</div></div>
    </div>
    <div class="row" style="margin-top:12px"><label for="nvar" class="small muted">Response of</label><select id="nvar"></select></div>
    <div class="plot tall" id="pnaive"></div>
    <p class="caption">Costs are flow losses (a trader's profit is negative). The arrows compare the naive equilibrium with the privy one.
      The plot overlays the two equilibria's responses to each shock (solid privy, dashed naive).${N.converged ? "" : " The naive solve did not converge; its numbers are its last iterate."}</p></section>`);
  const S = res.samples, T = N.samples;
  const vars = res.names.concat(res.definitions || []).filter((v) => T.kernels[v]);
  const sel = out.querySelector("#nvar");
  sel.innerHTML = vars.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("");
  const def = PRESETS[game] || {};
  sel.value = def.defaultVar && vars.includes(def.defaultVar) ? def.defaultVar : vars[0];
  const draw = () => {
    const v = sel.value, tr = [];
    res.channels.forEach((c, i) => {
      const col = pal[(i + 1) % pal.length];
      if (nonzero(S.kernels[v][c])) tr.push(line(S.age, S.kernels[v][c], chLabel(res, c) + ", privy", col));
      if (T.kernels[v] && nonzero(T.kernels[v][c])) tr.push(line(T.age, T.kernels[v][c], chLabel(res, c) + ", naive", col, { line: { color: col, width: 2, dash: "dash" } }));
    });
    Plotly.react("pnaive", tr, baseLayout({ title: titleOf("Response of " + label(v)), xaxis: { ...baseLayout().xaxis, title: { text: "shock age" } } }), plotCfg);
  };
  sel.onchange = draw; draw();
}

// ---------------------------------------------------------------------------------------------
// Sample paths: the solver returns every primary's kernel on a regular grid (res.paths); the shocks are drawn here,
// so a new draw is instant.  X(t) = mean + sum over channels and shock cells of K(t, s) dW(s), dW ~ N(0, h).
function rng(seed) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  let spare = null;
  return () => { if (spare !== null) { const v = spare; spare = null; return v; }
    let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
    const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
}
function simulatePaths(res, draws, seed) {
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
  const all = Object.keys(res.paths.kernels).filter((nm) => Object.keys(res.paths.kernels[nm]).length || Object.keys((res.paths.initial || {})[nm] || {}).length);
  if (!all.length) return;
  const stat = res.paths.kind === "stationary";
  const ctl = all.filter((nm) => isControl(nm)), sts = all.filter((nm) => !isControl(nm));
  const many = all.length > 6;
  let names = many ? ctl : sts.concat(ctl);
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Sample paths</h2>
    <div class="row"><button class="secondary" id="redraw">Draw new shocks</button><span class="small muted" id="seedlab"></span>
      ${many ? `<label for="pshow" class="small muted" style="margin-left:12px">Show</label><select id="pshow"><option value="c">controls</option><option value="s">states</option><option value="a">all</option></select>` : ""}</div>
    <div class="grid3" id="pgrid"></div>
    <p class="caption">Three draws of the shocks pushed through the equilibrium${stat ? ", over two lag windows of the stationary game" : res.kind === "transition" ? ", old shocks before time 0 included" : ""}. The band is the mean plus and minus two standard deviations. The shocks are drawn in your browser, so a new draw is instant.</p></section>`);
  const draw = () => {
    const sim = simulatePaths(res, 3, pathSeed), box = out.querySelector("#pgrid"); box.innerHTML = "";
    out.querySelector("#seedlab").textContent = `draw ${pathSeed}`;
    const pal = palette();
    for (const nm of names) {
      const S = sim[nm], div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      const hi = S.mean.map((m, j) => m + 2 * S.sd[j]), lo = S.mean.map((m, j) => m - 2 * S.sd[j]);
      const traces = [
        { x: S.t, y: hi, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
        { x: S.t, y: lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: css("--grid"), name: "± 2 sd", hoverinfo: "skip" },
        line(S.t, S.mean, "mean", css("--faint"), { line: { color: css("--faint"), width: 1, dash: "dot" } }),
        ...S.draws.map((y, d) => line(S.t, y, `draw ${d + 1}`, pal[(d + 1) % pal.length], { line: { color: pal[(d + 1) % pal.length], width: 1.5 } })),
      ];
      Plotly.newPlot(div, traces, baseLayout({ title: titleOf(label(nm)), showlegend: false, xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
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
    <p class="caption">Each row is a date t, each column a shock time s: the colour is how much a unit shock at s moves the quantity at t. Rows of the shock-response plots above are horizontal slices of this picture.${res.kind === "transition" ? " Columns left of s = 0 are the old regime's shocks." : ""}</p></section>`);
  const draw = (i) => {
    const [nm, c] = pairs[i], rows = P.kernels[nm][c], M = P.times.length - 1, nb = P.band;
    const sgrid = Array.from({ length: nb + M }, (_, k) => (k - nb + 0.5) * P.h);
    const z = rows.map((r) => sgrid.map((_, k) => (k < r.length ? r[k] : null)));
    let mx = 0; for (const r of z) for (const v of r) if (v !== null) mx = Math.max(mx, Math.abs(v));
    Plotly.react("psurf", [{ type: "heatmap", x: sgrid, y: P.times, z, zmin: -mx, zmax: mx, colorscale: [[0, css("--c2")], [0.5, css("--panel")], [1, css("--c1")]], hoverongaps: false,
      colorbar: { thickness: 10, outlinewidth: 0, tickfont: { color: css("--muted") } } }],
      baseLayout({ showlegend: false, hovermode: "closest", xaxis: { ...baseLayout().xaxis, title: { text: "shock time s" } },
        yaxis: { ...baseLayout().yaxis, title: { text: "date t" } }, margin: { l: 52, r: 12, t: 20, b: 50 } }), plotCfg);
  };
  const sel = out.querySelector("#svar"); sel.onchange = () => draw(+sel.value); draw(0);
}

// a transition: the loss path against the old and new stationary flows, the forecast errors, the march in T
function renderTransition(res, out) {
  const X = res.transition, pal = palette(), agents = Object.keys(res.costs);
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
    const c = pal[(i + 1) % pal.length], t = cut(X.times), t0 = t[0], t1 = t[t.length - 1];
    tr.push(line(t, cut(X.loss_path[a]), AGENT_LABEL[a] || a, c));
    if (flows) {
      tr.push(line([t0, t1], [X.old_flows[a], X.old_flows[a]], "before", c, { line: { color: c, width: 1, dash: "dot" }, showlegend: i === 0, hoverinfo: "skip" }));
      tr.push(line([t0, t1], [X.new_flows[a], X.new_flows[a]], "after", c, { line: { color: c, width: 1, dash: "dash" }, showlegend: i === 0, hoverinfo: "skip" }));
    }
    for (const [st, v] of Object.entries((X.belief_error || {})[a] || {})) br.push(line(t.slice(0, -1), cut(v).slice(0, -1), `${AGENT_LABEL[a] || a}, ${st}`, c));   // t < T: the value at T itself is the buffer's
  });
  Plotly.newPlot("p-loss", tr, baseLayout({ title: titleOf("Flow loss"), xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
  if (br.length) Plotly.newPlot("p-belief", br, baseLayout({ title: titleOf("Forecast error variance"), xaxis: { ...baseLayout().xaxis, title: { text: "time t" } } }), plotCfg);
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
    <div class="row"><label for="kvar" class="small muted">Response of</label><select id="kvar">${vars.map((v) => `<option value="${esc(v)}">${esc(label(v))}</option>`).join("")}</select></div>
    <div class="plot tall" id="pk"></div>
    <p class="caption">Response to a unit shock as a function of its age: how much of a shock that struck a units of time ago is still present. Channels with no response are left out.${prevResult ? " Dotted: the previous solve, before your last change." : ""}</p></section>`);
  const drawK = (v) => {
    const pal = palette();
    const traces = res.channels.map((c, i) => [c, i]).filter(([c]) => nonzero(S.kernels[v][c]))
      .map(([c, i]) => line(S.age, S.kernels[v][c], chLabel(res, c), pal[i % pal.length]));
    // the previous solve's curves, faint, to show what the last change did
    const P = prevResult && prevResult.samples && prevResult.samples.kernels && prevResult.samples.kernels[v];
    if (P) res.channels.forEach((c, i) => { if (P[c] && nonzero(P[c])) traces.unshift(line(prevResult.samples.age, P[c], chLabel(res, c) + " (before)", pal[i % pal.length], { line: { color: pal[i % pal.length], width: 1, dash: "dot" }, opacity: 0.5, showlegend: false })); });
    Plotly.react("pk", traces, baseLayout({ title: titleOf("Response of " + label(v)), xaxis: { ...baseLayout().xaxis, title: { text: "shock age" } } }), plotCfg);
  };
  const ksel = out.querySelector("#kvar"); ksel.value = kv; ksel.onchange = () => drawK(ksel.value); drawK(kv);
  renderFoc(res, out, keepCtl, "Split by shock age. The physical part is what the first-order condition would be if nobody reacted to the agent's deviation; the information wedge is the rest, which comes from the other agents revising their forecasts.", "shock age");
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
    const box = out.querySelector("#fgrid"); box.innerHTML = "";
    const f = F[ctl];
    const xs = x || f.s || (() => { const n = f.channels[res.channels[0]].physical.length, t = f.t; return Array.from({ length: n }, (_, i) => t * i / (n - 1)); })();
    for (const c of (res.shocks || res.channels)) {
      const d = f.channels[c];
      if (!d || (d.physical.length !== xs.length)) continue;
      if (!nonzero(d.physical) && !nonzero(d.wedge)) continue;
      const div = document.createElement("div"); div.className = "plot"; box.appendChild(div);
      Plotly.newPlot(div, [
        line(xs, d.physical, "physical", css("--c3")),
        line(xs, d.wedge, "information wedge", css("--c4"), { line: { color: css("--c4"), width: 2, dash: "dash" } }),
      ], baseLayout({ title: titleOf(chLabel(res, c)), xaxis: { ...baseLayout().xaxis, title: { text: xlabel } } }), plotCfg);
    }
    if (!box.children.length) box.innerHTML = `<p class="small muted">The first-order condition of ${esc(label(ctl))} has no stochastic part.</p>`;
  };
  const fsel = out.querySelector("#fctl"); fsel.value = kc; fsel.onchange = () => drawF(fsel.value); drawF(kc);
}

const CHECK_MEANING = {
  converged: "Whether the fixed-point iteration on the best-response map reached its tolerance.",
  resolution: "Whether the strategies are represented accurately on this grid (representation error against its threshold).",
  window: "Whether the stationary kernels have died out before the end of the lag window.",
  second_order: "Whether each agent's best response is a minimum, not only a stationary point (lowest curvature of its loss).",
  refinement: "Whether the costs and kernels stay put when the game is re-solved on a finer grid.",
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
function renderDiagnostics(res, out) {
  const rows = res.checks.map((d) => `<tr><td>${esc(checkName(d.name))}</td><td><span class="chip ${d.ok === false ? "warn" : "ok"}">${d.ok === false ? "failed" : "passed"}</span></td>
    <td class="mono">${checkValue(d)}</td>
    <td>${esc(d.ok === false ? (d.flag || d.meaning) : (CHECK_MEANING[d.name.split(":")[0]] || d.meaning))}</td></tr>`).join("");
  out.insertAdjacentHTML("beforeend", `<section class="panel"><h2>Diagnostics</h2>
    <p class="small muted" style="margin-top:0">${esc(res.version)}${solverThreads > 1 ? ` · ${solverThreads} threads` : ""} · ${esc(res.message)} · solve ${fmt(res.seconds, 2)} s.
    A failed check means the numbers may be off in the digits shown here; the row says what to raise.</p>
    <div class="tablewrap"><table class="diag"><thead><tr><th>Check</th><th>Status</th><th>Value / threshold</th><th>What it means</th></tr></thead><tbody>${rows}</tbody></table></div>
    ${res.stability ? `<p class="small" style="margin:10px 0 0">Stability: spectral radius of the best-response map ${fmt(res.stability.radius, 4)}
      (${res.stability.stable ? "below 1, so the equilibrium is locally stable under best-response iteration" : "1 or more, so best-response iteration moves away from it"}; ${esc(res.stability.method)}, ${res.stability.evaluations} evaluations).</p>` : ""}
    ${res.refinement ? `<p class="small" style="margin:6px 0 0">Refinement: re-solved at ${res.refinement.nodes} nodes in ${fmt(res.refinement.seconds, 1)} s; costs moved ${fmtE(res.refinement.cost_change)}, kernels ${fmtE(res.refinement.kernel_change)}.</p>` : ""}
    ${game !== "custom" ? `<details style="margin-top:12px"><summary>The model file</summary><pre class="describe" style="overflow-x:auto">${esc(jsyaml.dump(currentModel(), { flowLevel: 3 }))}</pre></details>` : ""}</section>`);
}

// ---------------------------------------------------------------------------------------------
$("solvebtn").onclick = () => requestSolve(0);
$("stopbtn").onclick = stopSolve;
$("sharebtn").onclick = async () => {
  writeHash();
  try { await navigator.clipboard.writeText(location.href); $("sharebtn").textContent = "Copied"; }
  catch (e) { prompt("Copy this link:", location.href); }
  setTimeout(() => { $("sharebtn").textContent = "Copy link"; }, 1500);
};
$("solvecustom").onclick = () => { customYaml = $("yaml").value; renderParamControls(); writeHash(); requestSolve(0); };
$("loadexample").onclick = () => { customYaml = EXAMPLES[$("example").value]; $("yaml").value = customYaml; renderParamControls(); writeHash(); };
$("yaml").addEventListener("input", () => { customYaml = $("yaml").value; });
$("yaml").addEventListener("change", () => { renderParamControls(); writeHash(); });
// the site's theme switch (and the system's) redraws the plots in the new colours
document.body.addEventListener("set-theme", () => setTimeout(() => { if (lastResult) renderResults(lastResult); }, 30));
readHash(); renderAll(); writeHash(); startWorker();
// a link or the back button that changes the hash (writeHash replaces it without firing this) opens that game
window.addEventListener("hashchange", () => {
  readHash(); renderAll(); lastResult = null; $("results").innerHTML = "";
  if (game !== "custom") requestSolve(0); else setStatus("idle", "Ready", "Edit the model and press Solve.");
  $("tabs").scrollIntoView({ block: "nearest" });
});

// ---------------------------------------------------------------------------------------------
// Sweep: the equilibrium costs as one slider runs across its range, the others held.  A second worker solves the
// points one after another, each from the last one's equilibrium, so the page's own solves are never queued behind it.
var sweepWorker = null, sweepReady = null, sweep = null, sweepSeq = 0;   // var: renderAll reads them before this line runs
var SWEEP_POINTS = 11;
function ensureSweepWorker() {
  if (sweepWorker) return sweepReady;
  sweepWorker = new Worker("worker.js");
  sweepReady = new Promise((resolve, reject) => {
    sweepWorker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === "ready") resolve();
      else if (m.type === "fatal") reject(new Error(m.message));
      else if (m.type === "result") onSweepResult(m);
    };
    sweepWorker.onerror = (e) => reject(new Error(e.message || "the sweep worker stopped"));
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
  if (sweepWorker && sweep && sweep.running) { sweepWorker.terminate(); sweepWorker = null; sweepReady = null; }
  if (sweep) sweep.running = false;
}
async function runSweep() {
  if (sweep && sweep.running) { const same = sweep.game === game; stopSweep(); if (same) { updateSweepPanel(); return; } }
  const def = PRESETS[game], s = def.sliders.find((k) => k.key === $("sweepkey").value);
  if (!s) return;
  sweep = { game, key: s.key, label: s.label, log: !!s.log, xs: sweepPoints(s), i: 0, costs: {}, naive: {}, start: lastStart[game] || null,
    base: sweepBase(s.key), running: true, id: ++sweepSeq * 100, t0: performance.now(), failed: 0 };
  updateSweepPanel();
  try { await ensureSweepWorker(); } catch (e) { sweep.running = false; $("sweepnote").textContent = "The sweep could not start: " + e.message; return; }
  nextSweep();
}
function nextSweep() {
  const S = sweep;
  if (!S || !S.running) return;
  if (S.i >= S.xs.length) { S.running = false; S.wall = (performance.now() - S.t0) / 1000; updateSweepPanel(); return; }
  const def = PRESETS[S.game];
  const model = modelWith({ [S.key]: S.xs[S.i] });
  const request = { ...currentRequest(), refine: false, stability: false, return_start: true };
  if (def.naive) request.naive_compare = def.naive;
  const cells = model.numerics && model.numerics.engine === "cells";
  if (S.start) request.start = S.start; else if (!cells) request.start_policy = "coarse";
  sweepWorker.postMessage({ type: "solve", id: S.id + S.i, model, request });
  updateSweepPanel();
}
function onSweepResult(m) {
  const S = sweep;
  if (!S || !S.running || m.id !== S.id + S.i) return;
  const res = JSON.parse(m.result), x = S.xs[S.i], def = PRESETS[S.game];
  const extra = def.constCost ? def.constCost({ ...values[S.game], [S.key]: x }) : {};
  if (res.ok && res.converged) {
    for (const [a, v] of Object.entries(res.costs)) (S.costs[a] = S.costs[a] || []).push([x, v + (extra[a] || 0)]);
    if (res.naive && res.naive.converged) for (const [a, v] of Object.entries(res.naive.costs)) (S.naive[a] = S.naive[a] || []).push([x, v + (extra[a] || 0)]);
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
    if (Q && Q.length) tr.push({ x: Q.map((p) => p[0]), y: Q.map((p) => p[1]), name: (AGENT_LABEL[a] || a) + ", naive", type: "scatter", mode: "lines+markers", line: { color: col, width: 2, dash: "dash" }, marker: { size: 5, color: col, symbol: "circle-open" } });
  });
  const cur = values[game][S.key], L = baseLayout();
  Plotly.react("psweep", tr, baseLayout({
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
  if (S.running) note.textContent = `Solving ${Math.min(S.i + 1, S.xs.length)} of ${S.xs.length}…`;
  else if (S.i < S.xs.length) note.textContent = `Stopped after ${S.i} of ${S.xs.length}.`;
  else note.textContent = `${S.xs.length} solves in ${S.wall.toFixed(1)} s${S.failed ? `, ${S.failed} did not converge and are left out` : ""}.`;
  if (!S.running && S.base !== sweepBase(S.key)) note.textContent += " The other parameters have moved since; sweep again to update.";
  $("sweepcap").textContent = `Each point is a full equilibrium; the dotted line is the slider's current value.${PRESETS[game].naive ? " Dashed: the naive equilibrium." : ""} Checks are off in the sweep.`;
}
$("sweepbtn").onclick = runSweep;
$("sweepkey").onchange = () => { if (sweep && sweep.game === game && !sweep.running) { sweep = null; } updateSweepPanel(); };
