# Entropic causets: physics notes

## 1. Motivation: why entropy and gravity?

Verlinde’s program (2010) treats gravity as an emergent effect driven by an entropy gradient rather than as a fundamental force. The standard form is `F = T ∇S`, where `S` is the entropy of the information carrier and `T` is an effective temperature (Unruh / holographic-screen style).

In this repository we use causets as discrete carriers of microstructure:

- the causal order is taken as primitive,
- geometry is emergent,
- local combinatorics can play the role of “microstates”.

Causets are natural for several reasons:

- discreteness avoids UV continuum artifacts,
- a partial order enforces a causal arrow,
- intuitions from Sorkin’s program suggest the correct macro scaling in a suitable mean limit.

The “game causet” construction (A1) adds a source of structure: events carry a rank and a past, built rank-by-rank, which enforces controlled topological sparsity and a tunable complexity budget.

## 2. Four levels of equivalence in Conway games

In the analysis of Conway games one can distinguish four identification levels:

1. position graph,
2. canonical form,
3. value class,
4. outcome class.

Each level identifies more objects than the previous one, so the number of distinguishable states does not increase. This yields a natural hierarchy of entropies:

`Spos ≥ Scan ≥ Sval ≥ Sout`.

Thermodynamically, each step is a further coarse-graining. This module implements an MVP on the **positional** level only (Hasse structure + ranks + automorphisms).

## 3. The sign in `S = ln Ω ± ln |Aut|`

Two standard readings remain in the literature:

- **Boltzmann / labeled (`S_A`):** symmetries are redundancies of labeling, so one **subtracts**
  `ln|Aut|` from a microstate count anchored at `ln n!`.
- **Algebraic (`S_B`):** one reports `ln|Aut|` as a structural term; it is convenient for
  group-theoretic bookkeeping but is **not** extensive on disjoint unions of identical
  objects unless one adds the Gibbs factorials for indistinguishable copies explicitly.

We **compute and report both** `S_A` and `S_B` (and twin- and rank-level variants) so
that scaling and distance laws can be diagnosed from the data without committing to a
single sign choice up front. See `docs/MATH.md` §5 for the corrected discussion and the
acknowledgment of an earlier over-strong recommendation of the plus sign alone.

## 4. Bridge to Verlinde

Starting point:

- `F = T ∇S`,
- `ΔS` is measured between configurations with different “mass” placement,
- temperature must be defined separately.

This repository does **not** define a dynamic temperature. A working hypothesis for later work is:

`T ~ d(canonical_reduction)/d(rank)`.

Intuition: temperature is the rate of loss of positional information per rank step.

Experiment E3 only measures `ΔS(r)` and fits a power-law trend; it is a sanity check for the entropic part, not a full force derivative.

**Sign and Verlinde:** The hypothesis that `ΔS` scales like a power of `1/r` is tested
against the same **Δ** in each convention. With a fixed total `n`, the `ln n!` anchor
cancels between the “at *r*” and “at infinity” configurations, so
`ΔS_A = - ΔS_B` (only the sign flips; moduli agree). Verlinde’s force law
`F = T ∇S` has a definite qualitative expectation for **attractive** vs **repulsive**
behavior as the separation changes; when mapping that qualitative sign to our numerics,
use the **Boltzmann** `S_A` version of `ΔS` so that “more entropy when approaching” is
not reversed by the automorphism-only sign. Reporting both `ΔS_A` and `ΔS_B` (as in E3’s
CSV/JSON) keeps the diagnostics explicit.

## 5. Three experiments: physical hypotheses

### E1: scaling of `ln|Aut|`

Hypothesis: `ln|Aut| ~ N^α`.

- `α ≈ 1`: bulk-like,
- `α ≈ 1/2`: surface-like,
- `α ~ 0+`: almost rigid.

### E2: Gibbs paradox

Discrete (exact) hypothesis:

- `|Aut(C ⊔ C)| = 2|Aut(C)|^2` for isomorphic components,
- `|Aut(C1 ⊔ C2)| = |Aut(C1)| |Aut(C2)|` for non-isomorphic components.

Failure means either a bug in `|Aut|` or a nontrivial subtlety in the symmetry definition.

### E3: combinatorial distance and `ΔS`

Definition:

`ΔS(r) = ln|Aut(C(r))| - ln|Aut(C(∞))|`.

Fit `ΔS ~ r^β`:

- `β ≈ -2` suggests Newton-like scaling,
- `β ≈ -1` suggests a different effective dimension,
- `β ≈ 0` suggests no clear gradient.

## 6. Limitations and caveats

- This is entropy bookkeeping, not a full theory of gravity.
- Small `N` may be dominated by finite-size effects.
- Defining “mass” as a local motif is ad hoc and not unique.
- The optional `pynauty` backend is faster, but tests must also run on `networkx`.
- Conclusions from E3 are qualitative: without a temperature model there is no full effective force.

## 7. What next

1. A formal temperature definition from canonical reduction.
2. Better definitions of a “mass motif” (e.g. local chain density / complexity concentration).
3. Study of off-shell automorphism sectors.
4. Continuum limit and links to `Cl(1,3)`-style models.
5. Comparing many causet generators under the same `|Aut|` pipeline.
6. Transfinite / surreal-flavored extensions.
7. Connection to CSG (Rideout–Sorkin) and growth dynamics.
8. Robustness of exponents to rank budget and sample rate.

## 8. Conway extension: edge-colored causets and four-valued outcome

The Conway extension replaces single-option events with short-game events
`e = {P_L | P_R}`. In graph terms, each Hasse edge is colored by membership in
Left options (`L`), Right options (`R`), or both (`LR`).

This extension is motivated by two diagnostics:

- Whether color constraints break rigidity at the twin-quotient level and
  produce nontrivial skeleton automorphisms.
- Whether four-valued outcomes `o(e) ∈ {L, R, =, ||}` expose nontrivial
  population-level structure (especially fuzzy classes).

Hypotheses tested in E4-E7:

1. **E4:** `log|Aut_twin|` is sometimes positive (unlike the rigid bare regime).
2. **E5:** outcome histogram has nondegenerate mass in all four classes.
3. **E6:** hierarchy collapse terms identify the dominant coarse-graining step.
4. **E7:** Gibbs identities remain exact with colored automorphisms.

Scope limit: this is a **structural** Conway extension. We do not implement full
Conway algebra (disjunctive sum, canonical reduction, surreal values) in this
iteration.

### 8.3 Conway extension hypothesis

For bare random causets (entropic-causets baseline), measurements indicated
`|Aut(C/~_twin)| = 1` in the tested regime. Conway edge-coloring asks whether
this past-only rigidity still holds.

Two alternatives:

- **(H_break):** edge coloring breaks past-only rigidity, so typical Conway
  causets satisfy `|Aut_LR(C/~_canonical)| > 1`.
- **(H_keep):** edge coloring keeps past-only rigidity, so typical Conway
  causets satisfy `|Aut_LR(C/~_canonical)| = 1` (or numerically near zero in logs).

E4 reports canonical and local twin quotients side-by-side:

- canonical quotient (`rank, P_L, P_R`) is the hypothesis observable;
- local quotient (`+ child_L, child_R`) is a diagnostic floor.

The primary signal is `mean_log_aut_twin_canonical` over `N`.

### 8.4 Generator comparison summary (E8)

The E8 experiment compares observables across multiple Conway causet
generators: stochastic (random sampling, coupled-pool) and deterministic
(canonical Conway games). Three findings structure the negative result
about Conway extension's first iteration.

#### Finding 1: All canonical Conway games have |Aut| = 1

Eleven canonical Conway game constructors (`make_zero`, `make_star`,
`make_nimber(k)`, `make_integer(n)`, `make_up`, `make_down`,
`make_switch(a, 0)`, `make_balanced_binary_tree(d)`) all yield |Aut| = 1
in our framework. This is robust to relaxing the rank-preserving constraint
on automorphisms — even pure edge-color-preserving graph automorphisms give
|Aut| = 1.

This is a structural fact about ConwayCauset encoding: each event in a
canonical game tree is uniquely identifiable by its in-degree, out-degree,
and edge-coloring context. There is no abstract Conway-game-theoretic
symmetry preserved by graph automorphism. Game-theoretic richness lives
in **recursive outcome computation** and **canonical form equivalence**,
not in graph automorphism.

#### Finding 2: Stochastic builder mechanism dominates the signal

Across five stochastic generator families, mean log|Aut(C)| varies by 5×:

| Generator | mean log|Aut| @ N=15 | rate(δ_canonical_local > 0) @ N=15 |
|-----------|----------------------|-------------------------------------|
| `random_default` (independent sampling) | 0.383 | 0% |
| `random_overlap_05` (per-edge LR coin flip) | **1.692** | **17%** |
| `coupled_pool_00` (disjoint L/R pools) | 0.000 | 7% |
| `coupled_pool_05` (mixed pools) | 0.000 | 0% |
| `coupled_pool_10` (single shared pool) | 0.000 | 0% |

The dominant signal in `random_overlap_05` traces to **LR-edge density**.
A per-edge coin flip with `overlap_rate = 0.5` produces approximately 50%
LR-edges, whereas independent sampling within shared pools produces only
~9% (= sample_rate²). LR-edges weaken aut-group constraints: an LR-edge
maps to LR-edge under any color-preserving permutation, allowing the
permutation to act on its endpoints without fixing per-side color identity.

This signal is therefore **structural** (LR-edge density), not Conway
game-theoretic. It does not increase under coupled-pool overlap because
that builder's overlap is statistical, not per-edge enforced.

#### Finding 3: Asymptotic collapse for all generators

Even the strongest stochastic family (`random_overlap_05`) decays with N:
mean log|Aut| from 1.69 (N=15) to 0.05 (N=50), with rate(δ > 0) from 17%
(N=15) to 0% (N=30+). Other stochastic families are weaker but show the
same pattern. Deterministic generators are at zero throughout.

For N ≥ 70 we expect all generators to converge to |Aut| = 1, matching
bare-causet behavior. Conway extension is in this sense a **finite-size
phenomenon** dependent on builder design.

#### Open questions for next iteration

- **Does any builder design produce N-persistent signal?** Trying:
  scale-free LR-edge enforcement (rate proportional to log N), forced-swap
  causet construction (events explicitly placed in twin pairs), or
  Hackenbush-style explicit symmetric trees.
- **Game-equivalence quotient.** Counting Conway games modulo canonical
  form equivalence (rather than graph isomorphism) is the natural way to
  expose game-theoretic structure. Out of scope for current framework
  (needs canonicalization algorithm); flagged for future work.
- **Loopy / temperate / hot games.** Switches `{a|b}` with a > b are
  encoded but their thermography (Conway temperature) is not measured by
  our observables. Future iteration should add `temperature_estimate(c)`
  via mean field on game value computation.

(See `docs/MATH.md` §11.3 for the LR-edge fraction formalism and twin-lemma
limitations in terms of E8.)

## 9. Observability and reproducibility

The CLI entry points ``qf-run-e1``, ``qf-run-e2``, and ``qf-run-e3`` call
:func:`quantum_foundations.entropic_causets.configure_experiment_logging` so each
run writes a per-experiment **log file** next to the CSV/JSON under
``papers/entropic-causets/results/`` (stems ``e1_aut_scaling``, ``e2_gibbs``,
``e3_verlinde``). The file handler captures **DEBUG** (builder parameters, slow
``aut_order``, E3 placement stages, skip reasons). A **stream handler** on stdout
emits **INFO** and above so short runs are not silent. Child processes do not attach
their own file handlers: workers return :class:`TaskDiagnostic` objects; the main
process replays their ``(level, message)`` tuples so the log is single-process
ordered. Use the log to interpret NaNs in E3 and gate failures in E2.
