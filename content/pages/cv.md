+++
title = "Curriculum Vitae"
description = "Sam Babichenko: quantitative researcher, PhD in Statistics and Applied Probability from UC Santa Barbara (2026). Research, experience, projects and skills."
path = "cv"
[extra]
og_image = "images/og-cv.png"
+++

<div class="cv-print-head">
<p class="n">Sam Babichenko</p>
<p class="c">sam@sbabichenko.com &middot; sbabichenko.com &middot; github.com/sbabichenko &middot; Charlotte, NC</p>
</div>

---

## Education

**PhD in Statistics and Applied Probability, UC Santa Barbara** &middot; Sep 2021 &ndash; Jun 2026\
Advisor: Tomoyuki Ichiba. MA in Statistics, 2026.\
Teaching assistant for 12 courses over four years: PSTAT 5A, 5LS, 8, 10, 120A/B, 123, 160A/B, 170, 174/274 and 176/276. Mentored 10+ undergraduate researchers.

**BS in Mathematics, UC San Diego** &middot; Sep 2019 &ndash; Jun 2021\
Dean's Undergraduate Excellence Award (top 0.5%, Physical Sciences).

---

## Research

**Noise-State Calculus for Dynamic Games with Strategic Information** &middot; PhD dissertation, 2026 &middot; [sbabichenko.com/dissertation](/dissertation/)\
Dynamic games in which players learn from each other's actions and can shape what others believe: continuous-time linear-quadratic-Gaussian games with any finite number of players. Each player keeps estimates of the primitive shocks instead of belief hierarchies, so beliefs, prices and policies are impulse responses and equilibrium is a fixed point in them, checked against arbitrary deviations. The information wedge, the shadow price of changing an opponent's beliefs, explains where these games depart from their full-information versions. Applications to Kyle&ndash;Back markets with several informed traders, networks of local markets, delayed public signals, and deviations that only some players can detect.

**Forecasting and Manipulating the Forecasts of Others** &middot; [arXiv: 2603.12140](https://arxiv.org/abs/2603.12140)\
Solo-authored, submitted March 2026. The two-player core of the dissertation. Splits the cost of dispersed information into an estimation part and a strategic part, and finds cases where more precise private information raises total cost.

**Software.** The noisestate solver in C++, about 10 ms per equilibrium, also compiled to WebAssembly for the browser explorer at [sbabichenko.com/noisestate](/noisestate/). Python package: [pip install noisestate](https://pypi.org/project/noisestate/).

**Talks.** "Forecasting and Manipulating the Forecasts of Others," Southern California Quantitative Finance Forum (SCQF), UC Santa Barbara, April 2026. "Mean Field Games and Interacting Particle Systems" (following Daniel Lacker), UCSD Stochastic Systems Seminar, January 2021.

---

## Experience

**Wells Fargo**, Charlotte, NC &middot; *Quantitative Researcher, rotational program* &middot; Jul 2026 &ndash; present

- Unifying the team's backtesting workflow.
- Refining the prepayment trading signal from my internship with the Decision Mesh.

**Wells Fargo**, Charlotte, NC &middot; *Quantitative Researcher Intern, Mortgage Model Development* &middot; Jun &ndash; Aug 2025

- Delivered the assigned mortgage pool segmentation early, then built a tree-based alternative with diagnostic tools. The buy-side trading desk then brought me onto an open-ended problem.
- Found that the team's prepayment models used normal approximations that miss the binomial structure of loan-level prepayment. Built a binomial model that corrects for effective loan count and partitioned pools to isolate systematic errors. The errors were persistent (R&sup2; &asymp; 0.8 at a 5-month horizon), which makes them usable as a trading signal.

---

## Selected Projects

**Decision Mesh** &middot; [sbabichenko.com/gate](/gate/) &middot; [github.com/sbabichenko/triangular-decision-mesh](https://github.com/sbabichenko/triangular-decision-mesh)\
Adaptive, continuous regression, inspired by decision trees, that refines only where a false-discovery gate finds support, and decomposes the data into a surface, unit-level effects, and noise. The rectangular version cuts one axis at a time and scales to higher dimensions.

**Multi-agent traffic simulation** (Waymo Open Motion Dataset)\
Reimplemented BehaviorGPT, a decoder-only transformer with agent&ndash;agent attention, for closed-loop 10 Hz trajectory generation. Extending it so the transformer predicts only at branch points where behavior becomes multimodal and a conditional diffusion model fills in between.

---

## Skills

**Programming:** C++23, Python, CUDA, SQL, Linux

**ML and data:** PyTorch, transformers, diffusion models, LightGBM, XGBoost, Polars, NumPy

**Methods:** stochastic calculus and control, Kalman filtering, Monte Carlo, PDE methods, hierarchical Bayesian models, time series, derivatives pricing

---

<div style="text-align:center; margin:24px 0 8px;">
<a class="cv-download" href="/Resume_Samuel_Babichenko.pdf" download>Download as PDF</a>
</div>

Last updated: **September 2026**
