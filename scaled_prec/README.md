# Scale-preconditioned Su--Olson JFNK

This is an additive implementation for the upstream
[`JFNK-ADvsFD`](https://github.com/marco-pas/JFNK-ADvsFD) repository. It leaves
`raddiffSolver.py` intact and imports its Crank--Nicolson residual, Laplacian,
source, initial-condition, and boundary-condition functions.

The transformed problem is

\[
x=S y, \qquad \widehat F(y)=R^{-1}F(Sy), \qquad
\widehat J(y)p=R^{-1}J(Sy)Sp.
\]

The code freezes `S` and `R` for each time step, solves for the dimensionless
Newton correction `delta_y`, and applies the physical update
`delta_x = S delta_y`. Both the AD and FD paths act on exactly the same scaled
residual graph.

## Runs

Unscaled AD reference:

```bash
python scaled_raddiffSolver.py \
  --preset classic-su-olson \
  --nx 128 --ny 4 --steps 100 \
  --jvp ad --scale-mode none \
  --metrics-csv results/ad_unscaled.csv \
  --output results/ad_unscaled_state.npz
```

Fixed scale-preconditioned AD run:

```bash
python scaled_raddiffSolver.py \
  --preset classic-su-olson \
  --nx 128 --ny 4 --steps 100 \
  --epsilon 10 --jvp ad \
  --scale-mode fixed \
  --scale-u 1.0 --scale-v 0.01 \
  --residual-scale-u 1.0 --residual-scale-v 0.01 \
  --metrics-csv results/ad_scaled.csv \
  --output results/ad_scaled_state.npz
```

Matched FD run (same transformed operator and tolerances):

```bash
python scaled_raddiffSolver.py \
  --preset classic-su-olson \
  --nx 128 --ny 4 --steps 100 \
  --epsilon 10 --jvp fd \
  --scale-mode fixed \
  --scale-u 1.0 --scale-v 0.01 \
  --residual-scale-u 1.0 --residual-scale-v 0.01 \
  --metrics-csv results/fd_scaled.csv \
  --output results/fd_scaled_state.npz
```

For an adaptive block scale frozen once per time step, replace `fixed` with
`state`. In that mode, `--scale-u` and `--scale-v` are positive lower/reference
values, and each actual block scale is
`max(reference, infinity_norm(field), scale_floor)`.

