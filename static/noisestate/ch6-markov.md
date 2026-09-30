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

We follow the competitive branch from zero inventory cost using damped Newton steps, check the original equation
residual, and require positive trading intensity and, for positive inventory cost, positive inventory reversion,
negative inventory quote loading and stabilizing maker feedback. This selects the stabilizing branch; it is not a
proof that no other admissible roots exist. The reported residual is an algebraic equation residual, not a
best-response spectral radius or a global equilibrium uniqueness certificate.

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

Shock responses propagate `(V,Vhat1,Vhat0,Q)` with its constant drift matrix. A quote blip starts the transparent
inventory at `1/(2*e)` and then decays at rate `delta`. In the opaque market it starts `(Q,xi)=(cn,-kn)` and evolves
under `A+B*K`. Quote and order responses use that same state. Small matrix exponentials handle repeated/zero rates;
the plot's right edge does not impose a zero boundary. Plot ranges are chosen only for display; costs include all ages.

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

A 40-solve warmed Node benchmark of the same two-market preset measured median 17.0 ms for the reference versus
174.0 ms for the eight-unit WASM calculation. Peak process RSS was 75.3 versus 132.8 MB; these are Node process
measurements, not total browser-tab memory. The reference allocates no WASM heap. Both paths retained less than
0.1 MB of extra JS heap after warm-up and explicit garbage collection. Separately, 15 repeated CARA solves kept
the ST/MT WASM heaps constant after warm-up (60.2/59.4 MB), with stable process RSS and under 0.1 MB JS retention.
