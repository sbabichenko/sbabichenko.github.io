+++
title = "Curriculum vitae"
path = "cv"
+++

<div class="cv-print-head">
<p class="n">Samuel Babichenko</p>
<p class="c">sam@sbabichenko.com &middot; sbabichenko.com &middot; github.com/sbabichenko &middot; Charlotte, NC</p>
</div>

---

## Education

**PhD in Statistics and Applied Probability, UC Santa Barbara** &middot; Sep 2021 &ndash; Jun 2026\
Advisor: Tomoyuki Ichiba.\
Teaching assistant for PSTAT 176/276. Mentored 10+ undergraduate researchers.

**BS in Mathematics, UC San Diego** &middot; Sep 2019 &ndash; Jun 2021\
Dean's Undergraduate Excellence Award (top 0.5%, Physical Sciences).

---

## Research

**Noise-State Calculus for Dynamic Games with Strategic Information** &middot; PhD dissertation, 2026 &middot; [sbabichenko.com/dissertation](/dissertation/)\
Continuous-time linear-quadratic-Gaussian games in which players learn from each other's actions. Each player keeps estimates of the primitive shocks instead of belief hierarchies, so beliefs, prices and policies are impulse responses and equilibrium is a fixed point in them, checked against arbitrary deviations. The information wedge, the shadow price of changing an opponent's beliefs, explains where these games depart from their full-information versions. Applications to Kyle&ndash;Back markets with several informed traders, networks of local markets, delayed public signals, and deviations that only some players can detect.

**Forecasting and Manipulating the Forecasts of Others** &middot; [arXiv: 2603.12140](https://arxiv.org/abs/2603.12140)\
Solo-authored, submitted March 2026. The two-player core of the dissertation: an exact solution, with no large-population limit, to a class of games with private signals open since Townsend (1983). Splits the cost of dispersed information into an estimation part and a strategic part, and finds cases where more precise private information raises total cost.

**Software.** The noisestate solver in C++, about 10 ms per equilibrium, also compiled to WebAssembly for the browser explorer at [sbabichenko.com/noisestate](/noisestate/). Python package: [pip install noisestate](https://pypi.org/project/noisestate/).

**Talk.** UCSD Stochastic Systems Seminar.

---

## Experience

**Wells Fargo**, Charlotte, NC &middot; *Quantitative Researcher* &middot; Jul 2026 &ndash; present

**Wells Fargo**, Charlotte, NC &middot; *Quantitative Researcher Intern, Mortgage Model Development* &middot; Jun &ndash; Aug 2025

- Delivered the assigned mortgage pool segmentation early, then built a tree-based alternative with diagnostic tools. The buy-side trading desk then brought me onto an open-ended problem.
- Found that the team's prepayment models used normal approximations that miss the binomial structure of loan-level prepayment. Built a binomial model that corrects for effective loan count and partitioned pools to isolate systematic errors. The errors were persistent (R&sup2; &asymp; 0.8 at a 5-month horizon), which makes them usable as a trading signal.

---

## Selected projects

**Decision Mesh** &middot; [sbabichenko.com/gate](/gate/) &middot; [github.com/sbabichenko/Decision-Mesh](https://github.com/sbabichenko/Decision-Mesh)\
Mesh regression that fits continuous surfaces where trees fit steps. A right-triangle or rectangular mesh is refined only where a false-discovery gate says the data justify a cut, and stops on its own. Built for the discontinuities that tree models showed in the prepayment work.

**Multi-agent traffic simulation** (Waymo Open Motion Dataset)\
Reimplemented BehaviorGPT, a decoder-only transformer with agent&ndash;agent attention, for closed-loop 10 Hz trajectory generation. Extending it so the transformer predicts only at branch points where behavior becomes multimodal and a conditional diffusion model fills in between.

---

## Skills

**Programming:** C++23, Python, CUDA, SQL, WebAssembly, Linux

**ML and data:** PyTorch, transformers, diffusion models, LightGBM, XGBoost, Polars, NumPy

**Methods:** stochastic calculus and control, Kalman filtering, Monte Carlo, PDE methods, hierarchical Bayesian models, time series, derivatives pricing

---

<div style="text-align:center; margin:24px 0 8px;">
<a class="cv-download" href="/Resume_Samuel_Babichenko.pdf" download>Download as PDF</a>
</div>

Last updated: **September 2026**
