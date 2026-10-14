# Chapter 6 finite-state reference

`ch6-markov.js` computes the equilibrium of the transparent/opaque market preset without truncating the inventory tail.
It is a separate, explicitly named reference alongside the general noisestate solver. Modified equations go through the
general solver; the reference rejects a changed model structure. The browser loads its small module on demand and can
visit the reference directly without downloading or allocating the WebAssembly runtime.

## Equations

Write `e = eps`, `g = gamma`, `r = rho`, `s = sigma_Z`, and `l = 1/s`. The trader's information edge is
`x = Vhat1 - Vhat0`, its order `D = beta*x + delta*Q`, and the quote `P = Vhat0 + p*Q`.
The trader's value coefficients `(a,b,c)` solve

```
n1 = 1 - l*a - b                 beta = n1/(2*e)
n2 = -p - l*b - c                delta = n2/(2*e)
r*a/2 = n1^2/(4*e)
r*b   = n1*n2/(2*e) + l*delta*a
r*c/2 = n2^2/(4*e) + l*delta*b
```

For the transparent maker, `gQ = -(l*b+c)/(2*e)`, `p = e*gQ-u/2`, and

```
r*u = 2*g - e*gQ^2 - gQ*u - u^2/(4*e).
```

For the opaque maker, use its inventory and the naive trader's misperception `(Q,xi)`. Define

```
cn = (beta*l+delta)/(l-p)         cx = beta*p+delta
kx = l*beta*p/(l-p)              kn = l^2*beta/(l-p)^2
A = [[-delta,-cx],[0,kx]]         B = [-cn,kn]
R = [[-p*delta+g,-p*cx/2],[-p*cx/2,0]]
N = [(delta-cn*p)/2,cx/2]
```

Its symmetric value matrix `U` solves `r*U = R + A'*U + U*A - v*v'/cn`, where `v = N+U*B`.
The feedback is `K = -v/cn`; equilibrium requires `K[0]=0`. The continuing drift in `xi` is essential.
Together these are seven coupled equations in `(a,b,c,U11,U12,U22,p)`.

The trader's equations imply `b=d/(r+d)`, `c=(2*e*d*d+2*l*d*b)/r`, and
`p=-2*e*d-l*b-c`, where `d=delta`. Put `h=r/(r+d)`; then
`a=h*h/(l*h+e*r+sqrt(e*r*(2*l*h+e*r)))` and `beta=sqrt(r*a/(2*e))`.
The transparent maker reduces to the strictly increasing scalar equation

```
2*g = 4*e*r*d + 6*e*d*d + l*d*(r+2*d)/(r+d).
```

Its positive root is bracketed by `0` and `2*g/(4*e*r+l)`. For the opaque maker,
equilibrium gives `v[0]=0`; its first two Riccati equations yield
`U11=(g-p*d)/(r+2*d)` and `U12=-cx*(p/2+U11)/(r+d-kx)`. Define

```
T = (l*beta*(r+2*d)+d*(r+d))/((l-p)*(r+d)-l*beta*p)
2*g = d*(r+2*d)/T-r*p.
```

This brackets a positive root between `0` and `2*g/(2*e*r+l)`. The remaining
Riccati entry is a scalar quadratic: with `v0=cx/2-cn*U12`, its coefficients are
`A=kn*kn/cn`, `B=r-2*kx+2*v0*kn/cn`, `C=2*cx*U12+v0*v0/cn`. We select
`U22=-2*C/(B+sqrt(B*B-4*A*C))` and check the resulting drift. Both scalar roots
are solved in `delta/gamma`, preserving relative accuracy for tiny positive
inventory penalties. Opaque continuation remains a fallback if the eliminated
candidate fails the original equations.

The opaque scalar root is unique too. Put `t=d/r`, `eta=e*r/l`, `w=l*beta/r`
and `v=-p/l`. Then `2*eta*w*(1+w)=1/(1+t)`, so `w` decreases with `t`, while
`v=2*eta*t*(1+t)+t*(1+2*t)/(1+t)` increases. Its equation is
`2*g/(l*r)=v+H`, where
`H=((1+v)*(1+t)+v*w)/(w/t+(1+t)/(1+2*t))` for `t>0`.
At fixed `w,v`, the numerator increases and denominator decreases with `t`;
`H` also increases with `v` and decreases with `w`. The full right side is
therefore strictly increasing from zero to infinity.

We check the original equation residual and require positive trading intensity
and, for positive inventory cost, positive inventory reversion, negative inventory
quote loading and stabilizing maker feedback. Independent continuation comparisons
cover 150 transparent and 250 opaque parameter draws, with coefficient differences
below `9e-13`. This selects the checked stabilizing branch; it is not a proof of
global equilibrium uniqueness. The reported residual is an algebraic equation
residual, not a best-response spectral radius.

## Costs and responses

For positive inventory cost, `E[x^2]=s/beta`, `E[x*Q]=0`, `E[Q^2]=s^2/(2*delta)`. Consequently

```
gross trader profit = s - p*s^2/2
trading cost        = e*(beta*s + delta*s^2/2)
inventory cost      = g*s^2/(2*delta)
trader loss         = trading cost - gross trader profit
maker loss          = gross trader profit + inventory cost
```

These are the preset's loss conventions. The chapter's total maker loss also subtracts noise-trader losses
`(l-p)*s^2`; do not confuse those accounting conventions. At `g=0`, inventory is a random walk with no loss weight,
so its penalty is zero. This differs from the limit of a positive penalty on stationary inventory.
At the default trading cost, discount and noise, the positive-penalty limit of
inventory cost is `0.35` (transparent) and `0.4395643924` (opaque). Inventory
loadings vanish while stationary inventory variance diverges. Small-penalty
regressions cover `gamma=1e-4` through `1e-16`, as well as exactly zero.

Shock responses use the equivalent triangular state `(E=V-Vhat1,X=Vhat1-Vhat0,Q)`:
`E'=-E`, `X'=E-(beta/s)*X`, `Q'=-beta*X-delta*Q`. Two- and three-exponential
convolutions evaluate this cascade directly. `expm1` handles two close decay
rates; a convergent divided-difference series handles three nearby rates,
including exact coincidences. This avoids a full matrix exponential at each
plot age while preserving the same continuous-time system.

A quote blip starts the transparent
inventory at `1/(2*e)` and then decays at rate `delta`. In the opaque market it starts `(Q,xi)=(cn,-kn)` and evolves
under `A+B*K`. Quote and order responses use that same state. A two-state
exponential uses its real decay rates and the same stable convolution; complex
rates retain the general matrix-exponential fallback. The slow rate is recovered
from the determinant and fast rate to avoid subtracting nearly equal numbers.
The plot's right edge does not impose a zero boundary. Plot ranges are chosen only for display; costs include all ages.

## Sample paths

Paths use the state `(E,X,Q,V)`, where `E=V-Vhat1` and `X=Vhat1-Vhat0`. Its drift and noise matrices are

```
F = [[-1,0,0,0], [1,-beta/s,0,0], [0,-beta,-delta,0], [0,0,0,0]]
B = [[1,0,-1], [0,-1,1], [0,-s,0], [1,0,0]]
```

For each displayed step `h`, the transition is `exp(F*h)` and innovation covariance is
`integral_0^h exp(F*u)*B*B'*exp(F'*u) du`. A block matrix exponential evaluates this integral; the browser
draws the resulting correlated Gaussian innovation. This is the exact sampled linear diffusion, rather than
an Euler approximation or a finite shock-history convolution. The implementation takes linear work and storage
in the number of displayed dates for this fixed four-dimensional state.

The fundamental is anchored at zero at time zero. Filtering errors start with variances `1` and `s/beta`;
inventory starts with variance `s*s/(2*delta)` when its penalty is positive. These three components have zero
cross-covariances. At zero inventory penalty, inventory instead starts at zero and follows its random walk.
The page states these initial conditions. Exact covariance propagation supplies the uncertainty bands.

The SciPy fixtures integrate the covariance independently using adaptive quadrature and check output variances
at times 0, 1, and 40 across all 50 cases. Worst relative error including these path checks is `8.7e-11`.
`node tools/noisestate/check-gaussian-paths.cjs` also checks recursion and seeded sample moments.

## Validation and provenance

The independent derivation comes from the author's September 29 Chapter 6 math review. A separate SciPy
implementation is included in `tools/noisestate/ch6_reference.py`. Regenerate the fixture with
`python tools/noisestate/generate_ch6_reference.py` in an environment with NumPy and SciPy. The fixtures use SciPy's
root solver and matrix exponential, not this JavaScript module. The extended random checks and long-window
comparisons are preserved with the September 30 project handoff.

Run `node tools/noisestate/check-ch6.cjs` from the repository root. The 50 saved cases cover both markets, every
slider-box corner at inventory costs 0, .01 and .2, plus the default point. An additional 200 random cases were
checked during development. Across all 250 comparisons, worst relative errors were 1.7e-11 in coefficients,
6.9e-12 in costs, 8.4e-12 in kernels and 2.9e-11 in deviation responses. The independent long-window noisestate
references agree within 2.1e-6 on response ages 0..20; their remaining cost differences follow their finite windows.

At the default parameters, maker/trader losses are `1.5978658138 / -0.9969306653` (transparent) and
`1.7138347039 / -0.9436176318` (opaque). The former eight-unit window substantially
understated both makers' losses and forced the deviation curves to zero before inventory had unwound.

A 40-solve warmed Node benchmark of the same two-market preset measured median 21.5 ms for the reference including sample-path transitions versus
174.0 ms for the eight-unit WASM calculation. Peak process RSS was 75.7 versus 132.8 MB; these are Node process
measurements, not total browser-tab memory. The reference allocates no WASM heap. Both paths retained less than
0.1 MB of extra JS heap after warm-up and explicit garbage collection. Separately, 15 repeated CARA solves kept
the ST/MT WASM heaps constant after warm-up (60.2/59.4 MB), with stable process RSS and under 0.1 MB JS retention.

Generating three 401-date sample paths and their uncertainty bands takes a median 2.7 ms in warmed Node, excluding
Plotly rendering. The state-transition descriptor is 647 bytes at the default parameters. Browser redraw checks
cover delayed Plotly loading and interrupted animations: old plots are detached immediately and purged after
pending layout work settles. DOM nodes and event listeners remain constant across repeated redraws.

After scalar elimination, three isolated, interleaved pairs measured the two
equilibrium solves at `359.85 -> 5.33` microseconds. The complete two-market
payload, including response curves and path transitions, improved from
`17.46 -> 16.51` ms. Peak Node RSS for the same workload fell from `88.9 -> 72.1`
MiB; retained JS after repeated payloads stayed below `0.1` MiB for both versions.
The unchanged 50 SciPy fixtures now agree within `1.3e-12`. Actual browser
checks passed slider endpoints, solver switching, redraw/animation interruption,
worker cancellation/retry, and hidden-worker release. These paired measurements
use the earlier finite-state implementation as baseline, not the WASM solver.

Eight additional propagation fixtures cover exact and nearly repeated rates,
zero and widely separated rates, a nonnormal repeated-root system, and the
complex-rate fallback. They use an independent 80-digit matrix exponential in
the original state coordinates. This also caught a double-precision reference
error of `9.1e-7` in one nearly repeated-rate case; the high-precision fixture
agrees with the direct formulas. Regenerate them with
`python tools/noisestate/generate_ch6_kernel_reference.py`. The combined 50
equilibrium and eight propagation checks agree within `1.1e-12`.

Three further isolated pairs compare direct propagation with the already
scalarized equilibrium implementation. Over 200 complete payloads per process,
median payload time fell from `15.03 -> 0.578` ms; batch mean time, including
natural garbage collection, fell from `15.17 -> 0.630` ms. Peak Node RSS fell
from `133.3 -> 77.5` MiB for this workload. Retained JS after collection stayed
below `0.2` MiB in both versions. These measurements include curves and path
transitions, but exclude Plotly rendering and browser worker startup.
