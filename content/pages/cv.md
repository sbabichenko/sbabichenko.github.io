+++
title = "Curriculum Vitae"
path = "cv"
+++

---

## Education

**PhD in Statistics and Applied Probability, UC Santa Barbara** &mdash; Sep 2021 &ndash; Jun 2026
Advisor: Tomoyuki Ichiba. Thesis: *Continuous-time LQG games with endogenous signals.*
Passed all qualifying exams prior to the first term of the PhD.
Graduate Teaching Assistant. Mentored 10+ undergraduates, majority of whom are now in PhD programs. Presented at UCSD Stochastic Systems Seminar.

**BS in Mathematics, UC San Diego** &mdash; Sep 2019 &ndash; Jun 2021
Dean's Undergraduate Excellence Award (top 0.5%, Physical Sciences).

---

## Research

**Forecasting and Manipulating the Forecasts of Others** &mdash; [arXiv: 2603.12140](https://arxiv.org/abs/2603.12140)
Solo-authored. Submitted March 2026.

First exact solution to a class of trading games with private information open since Townsend (1983). Solves for equilibrium trading strategies, price impact, and welfare costs in markets where traders learn from each other's order flow. Works in the exact model class behind Kyle-type market microstructure, optimal execution, and multi-agent control, with no approximation or large-population limit.

Quantifies how information asymmetry distorts prices and trading behavior: not just added noise, but systematic bias in mean prices, excess volatility, and wasted effort from strategic belief manipulation. Decomposes the total cost into an estimation component and a strategic component, showing the strategic channel dominates by an order of magnitude.

C++ solver computes equilibria in ~10ms. Compiled to WebAssembly for a browser-side interactive explorer at ~60ms. [Live demo: sbabichenko.com/lqg](/lqg).

**Asymmetric Competition Among Endogenously Informed Traders** &mdash; *In preparation*

Models markets where traders choose how much to invest in private information, then compete through order flow that reveals their knowledge to others. Shows that endogenous information acquisition systematically distorts prices and amplifies volatility, quantitatively resolving the Grossman&ndash;Stiglitz paradox.

---

## Experience

**Wells Fargo**, Charlotte, NC &mdash; *Quantitative Researcher Intern, Mortgage Model Development* &mdash; Jun &ndash; Aug 2025

- Completed assigned mortgage pool segmentation project in two days; independently built an improved tree-based alternative with diagnostic tools in one week. Work led the buy-side trading desk to pull me onto a harder open-ended problem.
- With no prior MBS background, identified a structural flaw in the team's prepayment models: normal approximations miss the binomial structure of discrete loan-level prepayment. Built a binomial model correcting for effective loan count, partitioned pools to isolate systemic errors, and found them highly persistent (R&sup2; &asymp; 0.8 at 5-month horizon), converting model misspecification into a tradeable signal for a $200B portfolio.
- Delivered a model upgrade projected to generate $100M in additional annual profit. Bridged model development and validation teams by synthesizing ideas from cross-team talks that neither group had connected.

---

## Selected Projects

**Multi-Agent Traffic Simulation** (Waymo Open Motion Dataset)
Reimplemented BehaviorGPT for multi-agent traffic simulation: decoder-only transformer with agent&ndash;agent attention, relative spacetime embeddings, and next-patch prediction generating closed-loop 10 Hz trajectories on the Waymo Open Motion Dataset.

Extending to a two-stage architecture: the transformer predicts only at branch points where agent behavior becomes multimodal (lane changes, yielding decisions, intersection entries), then a conditional diffusion model interpolates realistic trajectories between branch points with guided sampling.

**Adaptive Triangle-Mesh Regression**
Continuous surface model on an adaptive right-triangle bisection mesh with hierarchical partial pooling and wavelet shrinkage, addressing the discontinuity problem of tree-based models observed in the Wells Fargo prepayment project.

---

## Skills

**Programming:** C++23, Python, CUDA, Triton, Linux, bash, SQL

**ML & Data:** Transformer architectures, diffusion models, neural networks for control, PyTorch, NumPy, Polars, Pandas, LightGBM, XGBoost

**Methods:** Stochastic calculus, stochastic control, Kalman filtering, Monte Carlo, PDE methods, time series, derivatives pricing

---

<div style="text-align:center; margin:24px 0 8px;">
<a href="/Resume_Samuel_Babichenko.docx" download style="display:inline-block; padding:5px 16px; border:1px solid rgba(255,255,255,0.2); color:rgba(255,255,255,0.6); border-radius:6px; text-decoration:none; font-size:0.85em; transition:all 0.15s;">Download Resume</a>
</div>

Last updated: **March 2026**
