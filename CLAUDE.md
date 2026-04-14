# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

permABC is the official implementation of the paper "Permutations accelerate Approximate Bayesian Computation" by A. Luciano, C. Andral, C.P. Robert, and R.J. Ryder (arXiv:2507.06037).

The core idea: in hierarchical models with K exchangeable compartments (e.g. K regions in an epidemic), standard ABC wastes computation because simulated and observed compartments must align by index. permABC solves an optimal assignment problem (linear sum assignment) to find the best permutation, effectively gaining a factor ~K! in acceptance probability.

The paper was rejected from Biometrika (reviews in `paper/reject_letter.tex`). The editor acknowledged it as "a good paper" but reviewers wanted: (1) more numerical comparisons, (2) stronger theory, (3) clarification on the SIR model and Wasserstein ABC relationship. We are preparing a revised version for resubmission.

## Paper Structure

- `paper/permABC.tex` — main document, includes sections from `paper/sections/`
- `paper/sections/0_intro.tex` — introduction and hierarchical model framework
- `paper/sections/1_abc_to_perm.tex` — core method: permABC distance, Lemma 1, Theorem 1
- `paper/sections/2_sequential.tex` — SMC extension, Over-Sampling, Under-Matching, kernels
- `paper/sections/3_numerical.tex` — experiments: uniform toy, Gaussian, ABC-Gibbs comparison, OS/UM, SIR COVID
- `paper/sections/4_conclusion.tex` — conclusion
- `paper/sections/5_appendix.tex` — proofs, stratified permABC, technical details for OS/UM, blockwise kernel
- `paper/reject_letter.tex` — rejection letter with 3 referee reports + AE + editor comments
- Author comment macros: `\al{}` (Antoine, orange), `\rr{}` (Robin Ryder, blue), `\x{}` (Christian Robert, red)

## Key Theoretical Results

- **Lemma 1**: For epsilon < epsilon* = min_{sigma != Id} (1/2) d(y, y_sigma), at most one permutation satisfies the ABC criterion
- **Theorem 1**: When epsilon < epsilon*, permABC and standard ABC produce identical posteriors
- **Corollary**: permABC inherits convergence guarantees of ABC as epsilon -> 0

## Environment

**Always use the `permabc` pyenv virtualenv** (Python 3.10.13). Never use system Python or conda base.

```bash
# All commands must be prefixed with:
PYENV_VERSION=permabc pyenv exec python ...
PYENV_VERSION=permabc pyenv exec pytest ...

# Or set locally in the project:
pyenv local permabc
```

To recreate the environment:
```bash
pyenv virtualenv 3.10.13 permabc
pyenv activate permabc
pip install -r requirements.txt
pip install -e .
```

## Commands

### Install
```bash
pip install -e ".[dev]"
```

### Tests
```bash
PYENV_VERSION=permabc pyenv exec pytest tests/ -p no:cov
PYENV_VERSION=permabc pyenv exec pytest tests/test_algorithms.py -v
```

### Experiments (paper reproduction)
```bash
PYENV_VERSION=permabc pyenv exec python experiments/scripts/figures/fig6_performance_comparison_with_osum.py
PYENV_VERSION=permabc pyenv exec python experiments/scripts/run_performance_comparison.py --K 5 --K_outliers 0 --no-osum --seed 0
```

### Build C extensions (optional, for performance)
```bash
cd permabc/core
bash build_lsa.sh     # Custom Jonker-Volgenant LSA solver
bash build_cgal.sh    # CGAL Hilbert curve (requires CGAL)
```

## Architecture

### Core Data Structure: `Theta`
All parameter containers use `Theta` (a JAX-registered pytree in `permabc/utils/functions.py`):
- `loc`: shape `(n_particles, K, dim_loc)` — local (compartment-specific) parameters
- `glob`: shape `(n_particles, dim_glob)` — global (shared) parameters

### Algorithm Pipeline (SMC)
Each iteration of `perm_abc_smc` (in `permabc/algorithms/smc.py`):
1. **Resample** particles by weight
2. **MCMC move** (`permabc/sampling/moves.py`): propose -> simulate -> assign -> accept/reject
3. The assignment step calls `optimal_index_distance` (`permabc/assignment/dispatch.py`) which solves a linear sum assignment problem to find the best permutation mapping simulated to observed compartments

### Assignment Package (`permabc/assignment/`)
The key computational bottleneck — finding optimal permutations. Organized as:
- `dispatch.py`: Entry point (`optimal_index_distance`) + smart progressive strategies
- `distances.py`: Distance utilities (`compute_total_distance`, etc.)
- `solvers/lsa.py`: Exact O(K^3) via scipy or custom Jonker-Volgenant C solver
- `solvers/hilbert.py`: Fast O(K log K) via Hilbert curve sorting (Bernton et al. 2019)
- `solvers/sinkhorn.py`: Approximate O(K^2 * iters) via entropic regularization
- `solvers/swap.py`: Pairwise swap refinement O(K^2) (Numba/JAX/NumPy backends)

Assignment is controlled via boolean flags (`try_identity`, `try_hilbert`, `try_sinkhorn`, `try_swaps`, `try_lsa`) which build a progressive cascade in order: identity → hilbert/sinkhorn → swap → lsa. Error if both `try_hilbert` and `try_sinkhorn` are True. First iteration always uses full LSA regardless of flags. `resolve_assignment_bools()` in `algorithms/smc.py` converts bools to a cascade list.

### Sampling Package (`permabc/sampling/`)
- `kernels.py`: Proposal kernels (`KernelRW`, `KernelTruncatedRW`)
- `moves.py`: MCMC moves (`move_smc`, `move_smc_gibbs_blocks`)

### Legacy Shims (`permabc/core/`)
`permabc/core/` re-exports from `assignment/` and `sampling/`. The C extensions (.so files, build scripts) remain in `core/`.

### Models (`permabc/models/`)
Each model inherits from `ModelBase` (`permabc/models/__init__.py`) and implements:
- `prior_generator(key, n_particles, K) -> Theta`
- `data_generator(key, thetas) -> simulated_data`
- `distance_matrices_loc(sim, obs) -> pairwise_distances` — cost matrix for assignment
- `distance_global(sim, obs) -> scalar` — distance not subject to permutation

Key models:
- `Gaussian_with_no_summary_stats.py` — main toy model for Figures 3-6 (K components, shared variance sigma^2)
- `SIR.py` — epidemic model for Figures 7-8 (3 classes: `SIRWithKnownInit`, `SIRWithUnknownInit`, `SIR_real_world`). Uses **stochastic** simulation with multiplicative log-normal noise on transmission rate (sigma=0.05 in `simulate_sir_jax`, line 72-76).

### Algorithm Variants
- `perm_abc_smc` (`algorithms/smc.py`): Main algorithm — permutation-enhanced SMC
- `perm_abc_smc_os` (`algorithms/over_sampling.py`): Over-sampling — simulates M > K components, reduces toward K
- `perm_abc_smc_um` (`algorithms/under_matching.py`): Under-matching — starts with L < K, increases toward K
- `abc_smc` / `abc_vanilla`: Baselines without permutation
- `abc_pmc` (`algorithms/pmc.py`): Population Monte Carlo variant (requires numba)

### Experiments Layout
- `experiments/scripts/run_performance_comparison.py` — main benchmark runner (CLI with --K, --osum, etc.)
- `experiments/scripts/diagnostics.py` — KL divergence and posterior quality metrics (extracted from runner)
- `experiments/scripts/figures/fig*.py` — one script per paper figure
- `experiments/scripts/run_sir_real_world.py` — SIR on French COVID data (94 departments, d=283)
- `experiments/data/` — raw CSV data for SIR experiments

### C Extensions (in `permabc/core/`, loaded by `assignment/solvers/`)
- `librectangular_lsap_custom.so` + `lsa_ctypes.py`: Custom warm-start LSA solver (falls back to scipy)
- `cgal_hilbert.cpython-*.so` + `cgal_hilbert.cpp`: CGAL Hilbert curve (falls back to Python PCA+2D)

### JAX Usage
The codebase uses JAX with 64-bit precision enabled. JIT-compiled functions appear throughout. `Theta` is registered as a JAX pytree so it passes through `jit`/`vmap`/`lax.scan`.

## TODO — Resubmission (target: JASA)

Paper: `paper/resubmission.tex` (modulaire, \input{sections/...})
Archive Biometrika: `paper/biometrika.tex`
Reviews Biometrika: `paper/reject_letter.tex`
Mock review JASA: `paper/review_jasa_mock.tex`

### Etat au 14 avril 2026

FAIT :
- [x] 1. SIR stochasticite : formule log-normal ajoutee dans 3_numerical.tex
- [x] 2. Wasserstein multivarie : paragraphe WABC ajoute dans 1_abc_to_perm.tex (\new{})
- [x] 6. Pushforward OS : formalisme + M->inf dans 2_sequential.tex (\new{})
- [x] 8a. Doublons supprimes (2x Motivations, 2x Over-Sampling, 2x SIR)
- [x] 8b. Labels fixes (sec:gaussian_toy, appendix_secstrat, eq (1) \\\\)
- [x] 8c. Typo abstract (“Recovered”)
- [x] SIR v2 : reecriture prudente avec mention misspecification
- [x] resubmission.tex modulaire fonctionnel (compile, 40 pages, 0 erreurs)

RESTE A FAIRE :

[PRIORITE 1 — bloquant pour soumission JASA]

1. **Metriques de qualite posterieure** (mock review M1, ancien point 12)
   - Ajouter KL marginale, W2, ou couverture sur le toy Gaussian (posterieure en forme close disponible).
   - Remplacer ou completer les figures epsilon-vs-cost par des figures metrique-vs-cost.
   - Le code des metriques (sw2_joint, kl_mu_avg, w2_sigma2, w2_mu_avg) existe deja dans diagnostics.py.
   - 2 pickles outliers_4 restent a reprocesser (voir memoire project_resume_point.md).

2. **Baseline ABC-SMC standard sans permutations** (mock review M2, ancien point 4)
   - Ajouter ABC-SMC (Del Moral 2012) sans permutation dans les benchmarks.
   - Permet d’isoler le gain des permutations vs. le gain du cadre SMC.
   - Code : `abc_smc` existe deja dans `permabc/algorithms/smc.py`.

3. **Integrer asymptotics.tex** (mock review M4, ancien point 5)
   - Le contenu est pret dans `paper/sections/asymptotics.tex`.
   - Decider : sous-section du main text ou appendice ?
   - Contient : lien WABC, K->inf, M->inf, concentration locaux, non-commutation des limites.
   - Formaliser les resultats M->inf de la Section 2.3 comme Propositions (pas juste des equations \new{}).

4. **Theorem 1 : attenuer ou etendre** (mock review M3, ancien point 7)
   - Option A : borne a epsilon fini sur TV(pi*, pi).
   - Option B : reformulation plus prudente + discussion explicite de la limitation.
   - Supprimer “at no cost to theoretical consistency” si on ne peut pas le prouver pour epsilon fini.

5. **Validation SIR sur donnees synthetiques** (mock review M5)
   - Generer des donnees SIR synthetiques a parametres connus.
   - Montrer que permABC retrouve les vrais parametres.
   - Ajouter posterior predictive checks sur les vraies donnees.
   - Justifier le choix sigma=0.05 (sensibilite ?).

6. **Resoudre les \old{} / \new{} restants**
   - 1_abc_to_perm.tex : 3 blocs \old{} (phrases barrees a accepter ou rejeter).
   - 2_sequential.tex : 1 gros bloc \old{} (ancien texte informel OS).
   - Decider pour chaque : on accepte la suppression et on enleve le markup.

[PRIORITE 2 — forte valeur ajoutee]

7. **Texte introductif Section 2** (mock review m1)
   - Ajouter un paragraphe chapeau entre \section{Sequential...} et \subsection{State of the art}.

8. **Specifier les hyperparametres du Gaussian toy** (mock review m2)
   - Donner (a, b, s, n, K) dans le texte ou en legende de figure.

9. **Discuter Figure 2 a la lumiere de M->inf** (mock review m3)
   - Pointer explicitement sur la figure le comportement prior-like de beta et la concentration des locaux.
   - Regenerer la figure si les parametres ont change (prior plus vague, epsilon plus serre).

10. **Comparaison equilibree avec ABC-Gibbs** (mock review m4)
    - Ajouter un cas ou ABC-Gibbs marche bien (Gaussian toy standard, dependances faibles).

11. **Desambiguer la notation M** (mock review m6)
    - Section 2.1 : M = nb datasets par parametre. Section 2.3 : M = nb compartiments simules.
    - Renommer l’un des deux.

12. **Reporter ou utiliser le “unique particle rate”** (mock review m7)
    - Soit l’utiliser dans les experiences, soit supprimer la discussion.

13. **Benchmark cout computationnel assignment** (mock review m8)
    - Pour K=94 : fraction du temps dans l’assignment vs. simulation.

14. **Petit benchmark WABC / Hilbert / LSA** (ancien point 9)
    - Sur le toy Gaussian : cas ou permABC = WABC, cout exact vs approx.

15. **Exemple explicite “WABC = permABC”** (ancien point 10)
    - Proposition simple ou les deux coincident.

16. **Mettre a jour conclusion et abstract**
    - Mentionner les resultats asymptotiques, le lien WABC, le pushforward OS.

[PRIORITE 3 — reporter sauf si tout le reste est fini]

17. Nouveaux modeles epidemiques (SEIHR etc.)
18. Nouveau schema de bruit SIR
19. ABC-Gibbs sur modeles lourds
20. Multiplication des exemples LFI

[PLAN D’EXECUTION]

Etape A — Numerique
- Reprocesser les 2 pickles outliers_4
- Ajouter ABC-SMC standard dans les benchmarks
- Generer figures avec metriques de qualite (KL, W2)
- Validation SIR synthetique

Etape B — Theorie
- Integrer asymptotics.tex (section ou appendice)
- Formaliser les Propositions M->inf dans Section 2.3
- Reviser Theorem 1 (attenuer ou etendre)

Etape C — Redaction
- Resoudre tous les \old{}/\new{}
- Texte introductif Section 2
- Hyperparametres Gaussian toy
- Discussion Figure 2
- Desambiguer notation M
- Mettre a jour abstract + intro + conclusion

Etape D — Finition
- Relecture complete style / coherence
- Unique particle rate : utiliser ou supprimer
- Benchmark cout assignment
- Comparaison equilibree ABC-Gibbs
- Preparer version soumission propre (supprimer \new{}/\old{} definitions)