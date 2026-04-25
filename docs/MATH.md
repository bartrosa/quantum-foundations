# Entropic causets: mathematical notes

## 1. Stratified game causet

### Definition 1.1

A stratified game causet is a triple:

`C = (N, rank, past)`

- `N = {0, …, n-1}`,
- `rank: N → Z_{≥0}`,
- `past(i) ⊂ {j ∈ N : rank(j) < rank(i)}`.

The implementation stores only covering relations (Hasse edges), not the transitive closure.

### Lemma 1.2 (acyclicity / well-foundedness)

If every edge goes from a lower rank to a higher rank, there is no directed cycle.

*Sketch.* A cycle would require both strict increase and return to the same rank, contradicting rank monotonicity.

### Lemma 1.3 (no same-rank cover)

By construction, `past(i)` contains no vertex of the same rank as `i`.

## 2. Hasse DAG and automorphisms

### Definition 2.1

`Aut(C)` is the set of bijections `σ: N → N` such that:

- `rank(σ(i)) = rank(i)`,
- `(j ∈ past(i)) ⇔ (σ(j) ∈ past(σ(i)))`.

### Lemma 2.2

`Aut(C)` is a subgroup of `∏_r S_{|rank^{-1}(r)|}`.

*Sketch.* Rank preservation restricts permutations to layers; the `past` condition picks the subgroup that preserves the edge structure.

## 3. `|Aut|` as a counting problem

Computing the order of the automorphism group reduces to graph isomorphism with vertex colors given by rank.

- Fast backend: `pynauty` with `vertex_coloring`.
- Reference backend: `networkx.DiGraphMatcher` with a rank-based `node_match`.

The `networkx` path counts exact isomorphisms to self by iterating `isomorphisms_iter()`.

## 4. Disjoint union and the Gibbs statement

### Definition 4.1

`C1 ⊔ C2` has disjoint vertex sets and no new edges between components.

### Theorem 4.2 (Gibbs, two components)

- If `C1` and `C2` are not isomorphic: `|Aut(C1 ⊔ C2)| = |Aut(C1)| |Aut(C2)|`.
- If `C1 = C2 = C`: `|Aut(C ⊔ C)| = 2 |Aut(C)|^2`.

*Sketch.* In the isomorphic case, swapping the two isomorphic components contributes an extra `S2` factor.

### Corollary 4.3 (many copies)

For a decomposition into isomorphism classes `τ` with counts `N_τ`:

`|Aut(⊔_i C_i)| = (∏_τ N_τ!) (∏_i |Aut(C_i)|)`.

## 5. Entropy: three useful notions for causets (and a correction of an earlier sign draft)

**Honest acknowledgment.** An earlier draft of this document recommended a single “plus sign”
reading on the basis of a disjoint-union identity for `|Aut|`. That algebraic identity is
correct, but the resulting `S_B = ln|Aut|` is **super-additive** on disjoint unions of
identical causets, which is opposite to the standard Gibbs / indistinguishability accounting.
Here we **compute and report both** sign conventions; `S_A` is the thermodynamic default.

### 5.1 `S_A` (Boltzmann / labeled, minus in `S = ln n! - ln|Aut|`)

`S_A(C) = ln n! - ln|Aut(C)|` for the pos-level group (full rank+past-preserving
automorphisms). This is the form aligned with the usual “subtract symmetry degeneracy
from microstate counting” and with Bekenstein--Hawking-style entropy when one relates
`n!` to a labeled state count and `|Aut|` to unlabeled redundancy.

### 5.2 `S_B` (algebraic, plus: `S_B = ln|Aut|`)

`S_B` is operationally clean and composes in simple ways in products of *abstract*
symmetries, but for **disjoint unions of copies** the correct bookkeeping uses component
isomorphism types and their multiplicities (E2, `entropy_with_gibbs_correction`).

### 5.3 `S_C` (Conway / game levels)

Conway’s hierarchy (position, canonical form, value, outcome) is defined on a **larger
space of games** than a bare Hasse-labeled causet. **Conway’s full game-theoretic
hierarchy does not lift** to a stratified `GameCauset` with only `rank` and cover
`past` — we do not have `value` or `outcome` data on events as in full CGT.

For **poset** data alone, a strict analogue of coarse-graining is the chain

`rank → pos → twin`

(three **operational** symmetries). Here **twin** identifies events that share
the same **rank**, the same **Hasse-past** set, and the same **set of immediate
Hasse-children** (a strictly finer relation than *same rank and same past* when
out-neighbourhoods differ; that coarser relation does **not** in general
support the product identity in §5.5, because indistinguishability in the
automorphism **group** needs matching upper *and* lower covers in the
cover graph).

### 5.4 Inequalities (three poset levels)

Let `S_rank^B = Σ_r ln n_r!` with `n_r` events at rank `r` (the rank-only Young subgroup),
`S_pos^B = ln|Aut(C)|` (our DAG automorphisms), and `S_twin^B = ln|Aut(C / ~_twin)|` on the
twin quotient, with `~_twin` the equivalence “same *rank, past, and child
set* in the cover graph” (see ``twin_classes`` in the ``causet`` module). Then

`S_rank^B(C) ≥ S_pos^B(C) ≥ S_twin^B(C) ≥ 0`.

**Boltzmann:** `S_X^A = ln n! - S_X^B` for `X ∈ {pos, rank, twin}` (same `ln n!` anchor).

**Orbit losses:**

- `Δ_max_pos = S_rank^B - S_pos^B` (residual orbits inside rank-level groups after
  imposing the Hasse law).
- `Δ_pos_twin = S_pos^B - S_twin^B` (orbits collapsed by twin identification).

### 5.5 Twin-orbit factorization

Let twin classes be `τ` of sizes `k_τ`, and `Q = C / ~_twin`. Any permutation
that independently permutes events within each class preserves all covers and ranks, so
stabilizer fibering gives

`|Aut(C)| = |Aut(Q)| · ∏_τ k_τ!`,

hence

`S_pos^B - S_twin^B = Σ_τ ln(k_τ!) = Δ_pos_twin`.

*Sketch.* The quotient by twin classes forgets which representative was which; the
`∏ k_τ!` accounts for the lost inner permutations in the pos group.

### 5.6 Gibbs (disjoint union) for `S_A`

For **distinguishably labeled** unions of *non-isomorphic* `C_1, C_2` of sizes `n_1, n_2`,

`S_A(C_1 ⊔ C_2) - S_A(C_1) - S_A(C_2) = ln C(n_1 + n_2, n_1)` — only **binomial**
accounting, because `n!` cancels the automorphism part when components are in different
isomorphism types.

For **isomorphic** copies `C ⊔ C` with `|C| = n`, the standard identity gives

`S_A(C ⊔ C) - 2 S_A(C) = ln( (2n)! / (2 n!^2) ) = ln C(2n, n) - ln 2`,

i.e. the indistinguishability of two identical components. E2 uses exact integer `|Aut|`
as the primary gate and the **same** `S_A` relations as a secondary check.

## 6. Conway hierarchy *vs.* the poset twin hierarchy (what we are not doing)

In full combinatorial game theory, one may chain entropies across position → canonical
→ value → outcome. **Bare** stratified causets in this repository are **not** full games:
they are finite posets with a rank and cover relation only.

We therefore do **not** report `S_val` (value class) or `S_out` (outcome) entropies here:
`S_out` for minimal `{L|R}`-style “events” would be a trivial constant in a toy
assignment, and `S_val` would require a chosen surreal / value representation of events —
deliberate future work, not a stub.

The poset **rank–pos–twin** stack is the honest surrogate: it captures
coarse-graining and information loss **within** the available structural data, analogous
in spirit to `S_C` but strictly weaker and orthogonal to the `S_A` vs `S_B` labeling
convention.

## 7. Heuristics for `|Aut|` scaling

- **Linear (`O(N)`):** if the global object has product-like symmetry sectors.
- **Surface-like:** if “free” symmetries live mainly on a boundary of layers.
- **Log-like:** if typical objects are rigid and only local swaps remain.

E1 does not presuppose a scenario; it measures empirically.

## 8. Verlinde distance setup

### Definition 8.1

The combinatorial distance `r(m1, m2)` is the length of a shortest path in the Hasse skeleton
(suitably symmetrized with respect to edge directions) between representatives of the mass motifs.

### Definition 8.2

“Mass at infinity” means placing two motifs in sectors with no shared causal influence (in practice, maximal separation allowed by the background).

### Definition 8.3

`ΔS(r) = ln|Aut(C(r))| - ln|Aut(C(∞))|`.

In E3 this is a paired comparison for the same background seed.

## 9. Reproducibility

Each run is fixed by:

- a global seed,
- `SeedSequence.spawn` for per-task seeds,
- generator parameters (`n_target`, `max_rank`, `sample_rate`),
- deterministic build and relabeling rules.

Outputs are CSV + JSON (ISO 8601 UTC, pretty-printed JSON).

## 10. Open problems

1. A precise temperature definition in this class of models.
2. Continuum-limit criteria for the `ln|Aut|` term.
3. Links to CSG and causet growth dynamics.
4. Stability of exponents under changes of the mass definition.
5. The role of transfinite sectors in surreal/game-style constructions.

## 11. Observability

Fitted parameters (slopes, ``r²``) and trial tables remain in CSV/JSON. For **why** a
row is missing or a Gibbs check failed, read the co-located ``*.log`` in
``papers/entropic-causets/results/`` (see also ``docs/PHYSICS.md`` §8). E3
placement diagnostics record skip paths (e.g. no host rank) and
``STRUCTURAL DIFF EMPTY`` when the “at *r*” and “at infinity” labeled DAGs
coincide, which invalidates a naive ``|Aut|`` comparison.
