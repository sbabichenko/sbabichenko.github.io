"""Figure 1.2: one shock to the state at time 0, followed through the state,
each player's estimate of the state, and each player's action.
Needs noisestate 1.1 or later and matplotlib."""
import numpy as np
import matplotlib.pyplot as plt
import noisestate as ns

# The game, written as its equations in fig1_2.yaml, and its equilibrium.
game = ns.load("fig1_2.yaml")
res = game.solve(start_policy="coarse")

# Follow one unit shock to the state, W0, that struck at time 0.
t = np.linspace(0, 1, 201)
X = res.response("X", to="W0", at=0).over(t)
X1 = res.response("X", to="W0", at=0, seen_by="player1").over(t)
X2 = res.response("X", to="W0", at=0, seen_by="player2").over(t)
D1 = res.response("D1", to="W0", at=0).over(t)
D2 = res.response("D2", to="W0", at=0).over(t)

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))

left.plot(t, X, color="k", lw=2, label=r"$X(t,0)$")
left.plot(t, X1, color="C0", lw=2, ls="--", label=r"$\hat X^1(t,0)$")
left.plot(t, X2, color="C3", lw=2, ls="--", label=r"$\hat X^2(t,0)$")
left.set_title("State and posterior estimates")
left.set_ylabel("response", fontsize=12)

right.plot(t, D1, color="C0", lw=2, label=r"$\mathcal{D}^1(t,0)$")
right.plot(t, D2, color="C3", lw=2, label=r"$\mathcal{D}^2(t,0)$")
right.set_title("Primitive-control responses")

for ax in (left, right):
    ax.axhline(0, color="k", lw=0.6, alpha=0.6)
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"$t$", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(r"Source-$W^0$ responses for $p_1=3,\ p_2=10$", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("fig2_impulse_w0_estimates.pdf", bbox_inches="tight")
