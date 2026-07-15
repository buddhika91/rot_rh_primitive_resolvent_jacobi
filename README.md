# Canonical Primitive Xi-Resolvent Jacobi Research Codebase

A reproducible research repository for the canonical chain

\[
\Xi ightarrow \Phi(u) ightarrow R(u)=-\Phi'(u)/\Phi(u)
ightarrow \{r_n\} ightarrow \{lpha_n,eta_n\}
ightarrow J_d ightarrow \gamma_jpprox\lambda_j(J_d)^{-1/2}.
\]

This release covers much more than GUE audits. It contains the resolvent construction, generator/operator realization, zero-alignment demonstrations, positivity gates, control attacks, stability tests, data, and plots.

> **Scientific status:** strong finite computational evidence for a canonical resolvent-to-Jacobi realization. This repository does **not** prove the Riemann Hypothesis.

## Main achievements represented here

- Canonical, unfitted Jacobi construction from the primitive Xi resolvent.
- Positive moments, Hankel/shifted-Hankel gates, and positive S-fraction tests in the tested range.
- Self-adjoint finite operator with positive off-diagonal Jacobi coefficients.
- One nested coefficient sequence across depths.
- Depth-24 reconstruction of 12 zeros with relative RMSE about `1.71e-4` in the included raw validation data.
- Precision/radius/sample stability and direct-derivative agreement.
- Survival of the authentic signal while sign-flip, permutation, and Gaussian controls break the positive construction.
- User-reported preregistered depth-64 holdout: 28 stable spacings, 3/3 GUE methods, bootstrap minimum 0.990, and zero relative RMSE `3.633e-12`.

## Repository map

```text
src/audits/   full numerical audits
src/demos/    operator, generator, zero alignment, controls, features
scripts/      plot generation
data/reported/ raw available validation outputs
data/user_reported/ explicitly labeled terminal-summary data
figures/      generated publication-quality plots
docs/         architecture, features, experiments, limitations, provenance
configs/      reproducible commands
tests/        automated checks
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash configs/run_full_research_demo.sh
pytest -q
```

## Operator construction

```bash
bash configs/run_operator_demo.sh
```

Exports the finite matrix, Jacobi coefficients, spectrum, and self-adjointness report.

## Generator construction

```bash
bash configs/run_generator_demo.sh
```

Uses the same Jacobi matrix in `U(t)=exp(-itJ)` and verifies unitarity. The operator and generator are the same mathematical object viewed in spectral and dynamical roles.

## Zero alignment

```bash
bash configs/run_zero_alignment_demo.sh
```

## Controls

```bash
bash configs/run_controls_demo.sh
```

## Figures

```bash
bash configs/run_all_plots.sh
```

Included figures cover zero alignment, error decay, depth convergence, control survival, moments, Jacobi coefficients, Carleman growth, and the preregistered depth summary.

## Full audits

See `configs/` and `docs/EXPERIMENTS.md` for stable validation, GUE frontier, and preregistered holdout commands.

## Central open theorem

The decisive unresolved analytic problem is to prove unconditionally that

\[
R(u)=\int_0^\infty rac{d\pi(x)}{1-u x},\qquad d\pi(x)\ge 0,
\]

globally, and to establish the corresponding infinite self-adjoint Jacobi operator and exact spectral correspondence without assuming RH.

## Citation and licence

See `CITATION.cff` and `LICENSE`.
