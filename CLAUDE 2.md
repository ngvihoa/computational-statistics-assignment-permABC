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

## TODO — Resubmission after Biometrika rejection

Paper: `paper/permABC.tex` | Reviews: `paper/reject_letter.tex`

TODO — Resubmission permABC (priorisée)

[PRIORITÉ 1 — indispensable avant resoumission]

1. SIR stochasticity
- Mentionner explicitement dans le papier que le modèle SIR utilisé en ABC est stochastique.
- Décrire clairement le bruit multiplicatif log-normal déjà présent dans le simulateur.
- Vérifier que cette précision apparaisse :
  - dans la section modèle,
  - dans la section expérience réelle,
  - et dans la légende / commentaire associé si nécessaire.

2. Corriger le passage sur Wasserstein multivarié
- Supprimer toute affirmation disant que Wasserstein ne s’applique pas au cas multivarié.
- Expliquer correctement le lien avec WABC :
  - permutation-invariance au niveau des compartiments,
  - différences de cible statistique,
  - différence entre paramètres globaux et paramètres locaux.
- Ajouter une comparaison conceptuelle propre avec Hilbert / Sinkhorn / exact assignment.

3. Clarifier la Figure 2 (Over-Sampling)
- Expliquer pourquoi la prior peut sembler plus concentrée que la pseudo-postérieure pour certains panneaux.
- Ajouter une lecture “unprojected vs projected posterior”.
- Appuyer cette clarification par le toy model gaussien si possible.

4. Comparaison avec ABC-SMC standard
- Ajouter une baseline explicite avec suite classique d’epsilons décroissants.
- Répondre clairement dans le texte à la question :
  “pourquoi Over-Sampling / Under-Matching plutôt qu’une simple suite standard de seuils ?”
- Expliquer que OS/UM sont des distributions de bridging complémentaires, pas un remplacement d’ABC-SMC standard.

5. Intégrer les asymptotiques dans le main text
- Ne pas laisser asymptotics.tex isolé.
- Ajouter une sous-section courte dans le papier principal.
- Y inclure au minimum :
  - lien permABC / WABC au niveau compartiment,
  - interprétation asymptotique en K grand,
  - résultat principal sur Over-Sampling quand M tend vers l’infini.

6. Résultat théorique propre sur Over-Sampling
- Formaliser Over-Sampling comme pushforward.
- Distinguer clairement :
  - loi non projetée,
  - loi projetée.
- Montrer que, sous condition de support / faisabilité,
  la marginale en beta tend vers la prior quand M tend vers l’infini.
- Expliquer que les paramètres locaux projetés restent concentrés autour des meilleurs matches.

7. Lemma 1 / Theorem 1
- Soit étendre proprement le résultat au cas avec matches exacts / epsilon*,
- soit reformuler le résultat de manière plus prudente si l’extension est trop lourde.
- Ne pas laisser ce point ambigu.

8. Nettoyage final du manuscrit
- Supprimer toutes les traces \new{}, \old{}, commentaires auteurs, passages doublonnés.
- Vérifier que toutes les sections utilisées sont bien incluses dans le main tex.
- Harmoniser notations, labels, renvois, appendices et figures.


[PRIORITÉ 2 — forte valeur ajoutée si coût raisonnable]

9. Petit benchmark WABC / Hilbert / LSA exact
- Ajouter un benchmark ciblé sur le toy Gaussian.
- Montrer :
  - cas où permABC et WABC coïncident,
  - coût exact vs approximation,
  - régime où LSA exact reste raisonnable pour K modéré.

10. Exemple explicite “WABC = permABC”
- Ajouter un exemple / proposition simple où les deux approches sont équivalentes.
- L’utiliser pour renforcer le positionnement théorique du papier.

11. Commentaire sur la portée générale des idées séquentielles
- Ajouter un paragraphe expliquant que les idées de OS / UM / kernels peuvent s’étendre au-delà du cadre strict d’ABC hiérarchique.
- Rester bref mais explicite.

12. Métriques de qualité postérieure
- Ajouter une ou deux métriques de qualité au-delà de epsilon.
- Priorité à une métrique jointe robuste :
  - sliced Wasserstein, ou
  - score joint bien interprété,
  - plus métriques marginales si utile.
- Ne pas surcharger les figures.


[PRIORITÉ 3 — à repousser sauf si tout le reste est fini]

13. Nouveaux modèles épidémiques complexes
- Reporter SEIHR / modèles plus riches à plus tard.
- Ne pas refaire toute la section réelle avant la resoumission.

14. Nouveau schéma de bruit SIR
- Ne pas changer le modèle bruité actuel tant que la version existante n’est pas clairement documentée.

15. ABC-Gibbs sur modèles lourds
- Reporter les expériences ABC-Gibbs sur SIR / SEIHR.
- Trop coûteux pour un gain de resoumission incertain.

16. Multiplication des exemples LFI
- Ne pas viser 3 à 5 nouveaux exemples.
- Mieux vaut un ou deux ajouts très ciblés et propres qu’un élargissement massif.

17. Refonte complète des figures epsilon -> KL
- Ne pas tout remplacer.
- Ajouter des métriques complémentaires seulement si elles clarifient vraiment l’histoire.


[PLAN D’EXÉCUTION CONSEILLÉ]

Étape A
- Corriger SIR
- Corriger Wasserstein
- Clarifier Figure 2
- Nettoyer le manuscrit

Étape B
- Intégrer asymptotiques dans le papier
- Ajouter résultat Over-Sampling M -> inf
- Réviser Lemma 1 / Theorem 1

Étape C
- Ajouter baseline ABC-SMC standard
- Ajouter petit benchmark WABC / Hilbert / LSA
- Réviser abstract + introduction + conclusion pour refléter le nouveau positionnement

Étape D
- Relecture complète style / cohérence / figures / appendices
- Préparer version resoumission propre