/-
Copyright (c) 2026 LMLExplore contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: LMLExplore contributors
-/
module

public import LMLExplore.ReinforcementLearning.Generative.Decoupling
public import LMLExplore.ReinforcementLearning.Generative.PolicyEvaluation
public import LMLExplore.MarkovDecisionProcess.Absorbing.StateAction

/-!
# Planning with a generative model: leave-one-out evaluation of the separated plug-in policy

Let `M` be an MDP with `S` states, `A` actions and rewards in `[0, C]`, `γ < 1`, and
`M̂ = M.empirical N ω` the empirical MDP of `N` samples per state-action pair under the generative
model `Bandits.streamMeasure M.P`. The plug-in policy `π̂` is the greedy optimal policy of `M̂`. It
depends on all the samples, so that the evaluation bound of
`ReinforcementLearning/Generative/PolicyEvaluation.lean` [LWCGC24, Lemma 1], which needs a policy
fixed independently of the data, does not apply to it directly.

Li, Wei, Chi, Gu and Chen [LWCGC24, Lemma 5] remove this dependence when the optimal action values
of `M̂` are *separated* by `η > 0`: at every state `s`, every action `a ≠ π̂ s` has
`Q̂* (s, a) + η ≤ Q̂* (s, π̂ s)` (the event `B_η` of [LWCGC24, (41)]). Their leave-one-out argument
runs through the state-action absorbing modifications `M̂.saAbsorbing p c`
(`MarkovDecisionProcess/Absorbing/StateAction.lean`):
* the modification at `p` is computed from the samples of the pairs other than `p`, so that every
  function of it, such as its greedy optimal policy, is independent of the samples of `p`
  (`Learning.MDP.indepFun_comp_saAbsorbing_empirical`), and Bernstein's inequality applies at `p`
  to every function of the modification;
* under the separation, a uniform grid of rewards `c`, fixed in advance, contains one whose
  modification at `p` has the greedy optimal policy `π̂` [LWCGC24, Lemma 4].

A union bound over the pairs `p`, the points of the grid and the levels of [LWCGC24, Lemma 2],
`ProbabilityTheory.Kernel.abs_integral_resolvent_sub_integral_resolvent_le_of_abs_sub_le`, then
gives the Bernstein-type deviations of the empirical state kernel of `π̂` that Lemma 2 needs.
With `K + 1` grid points, `8 (1 + γ) C / (1 - γ) ≤ (1 - γ) η K`, `δ > 0` and
`L = log (4 S A (K + 1) log (e / (1 - γ)) / δ)`, as soon as `18 L ≤ (1 - γ) N`, with probability
at least `1 - δ`, if `M̂` is separated by `η`, then

`|v̂^π̂ - v^π̂| ≤ 3 C √(2 L / (N (1 - γ) ^ 3))`.

Combined with [LWCGC24, Lemma 1] for the greedy optimal policy of `M`, which does not depend on the
samples, and with the optimality of `π̂` in `M̂`, this bounds the suboptimality of the plug-in
policy: with probability at least `1 - δ`, if `M̂` is separated by `η`, then
`v* - v^π̂ ≤ 6 C √(2 L' / (N (1 - γ) ^ 3))`, with `L' = log (8 S A (K + 1) log (e / (1 - γ)) / δ)`
the level of the first bound at `δ / 2`.

[LWCGC24, Lemma 5] assumes that the separation holds with probability at least `1 - δ` and
concludes with probability at least `1 - 3 δ`. The statements here hold on the separation event,
whatever its probability, so that they combine with any tie-breaking argument.

The grid size only enters through `log (K + 1)`: for `η` polynomially small in `S`, `A`,
`1 - γ` and `δ`, as produced by the random perturbation of the rewards of [LWCGC24, Lemma 6]
(`MarkovDecisionProcess/TieBreaking.lean`), the bound is of order `√(L / (N (1 - γ) ^ 3))` with a
logarithmic `L`, over the full range of sample sizes `N ≳ L / (1 - γ)`.

## Main statements

* `Learning.MDP.indepFun_comp_saAbsorbing_empirical`: every function of the `p`-absorbing
  modification of the empirical MDP is independent of the samples of `p`.
* `measureReal_sqrt_variance_add_le_abs_integral_sub_integral_empirical_saAbsorbing_le` (in
  `Learning.MDP`): Bernstein's inequality at `p` for the functions of the `p`-absorbing
  modifications of the empirical MDP.
* `Learning.MDP.le_streamMeasure_forall_abs_value_empirical_sub_value_le_of_separated`: the
  evaluation of the separated plug-in policy [LWCGC24, Lemma 5, first claim].
* `Learning.MDP.le_streamMeasure_forall_optValue_sub_value_le_of_separated`: the suboptimality of
  the separated plug-in policy [LWCGC24, Lemma 5, second claim].

## References

* [G. Li, Y. Wei, Y. Chi, Y. Gu and Y. Chen, *Breaking the sample size barrier in model-based
  reinforcement learning with a generative model*, Operations Research, 2024][LWCGC24], §6.3.2,
  Lemma 5 and its proof in Appendix B.4 (arXiv:2005.12900)
-/

public section

open MeasureTheory ProbabilityTheory LMLExplore Learning Bandits
open Real (log exp)
open scoped NNReal

namespace Learning.MDP

variable {𝓢 𝓐 𝓡 : Type*} {m𝓢 : MeasurableSpace 𝓢} {m𝓐 : MeasurableSpace 𝓐}
  {m𝓡 : MeasurableSpace 𝓡} [MeasurableSingletonClass 𝓢] [MeasurableSingletonClass 𝓐]

section Decoupling

variable [Finite 𝓢] [Finite 𝓐]

/-- **Leave-one-out decoupling** [LWCGC24, §6.3.2]. Under the generative model of `M`, every
function of the `p`-absorbing modification of the empirical MDP is independent of the samples of
the pair `p`: the modification is computed from the samples of the other pairs. -/
theorem indepFun_comp_saAbsorbing_empirical (M : MDP 𝓢 𝓐 𝓡) (N : ℕ) [NeZero N] {β : Type*}
    [MeasurableSpace β] (F : MDP 𝓢 𝓐 𝓡 → β) (p : 𝓢 × 𝓐) (c : 𝓡) :
    IndepFun (fun ω ↦ F ((M.empirical N ω).saAbsorbing p c)) (fun ω i ↦ ω i p)
      (streamMeasure M.P) := by
  simp_rw [saAbsorbing_def]
  exact indepFun_comp_redirect_empirical M N F _ p.1 c (Set.mem_singleton p)

/-- **Bernstein's inequality for the state-action absorbing modifications of the empirical MDP**
[LWCGC24, (92)]. Under the generative model of `M`, let `G M'` be a function on the states
computed from an MDP `M'` (for instance a value of its greedy optimal policy), with values in
`[l, u]`, `l < u`, at the `p`-absorbing modifications of the empirical MDPs with reward `c`. For
`N ≠ 0` and every `x`, the one-step prediction error `|∫ v dP p - ∫ v dP̂ p|` of
`v = G ((M.empirical N ω).saAbsorbing p c)` is at least
`√(2 Var_{P p}(v) x / N) + (u - l) x / (3 N)` with probability at most `2 exp (-x)`. -/
theorem measureReal_sqrt_variance_add_le_abs_integral_sub_integral_empirical_saAbsorbing_le
    (M : MDP 𝓢 𝓐 𝓡) (N : ℕ) [NeZero N] (G : MDP 𝓢 𝓐 𝓡 → 𝓢 → ℝ) {l u : ℝ} (hlu : l < u)
    (p : 𝓢 × 𝓐) (c : 𝓡) (hG : ∀ ω s', G ((M.empirical N ω).saAbsorbing p c) s' ∈ Set.Icc l u)
    (x : ℝ) :
    (streamMeasure M.P).real {ω |
      √(2 * Var[G ((M.empirical N ω).saAbsorbing p c); M.P p] * x / N) +
        (u - l) * x / (3 * N) ≤
      |∫ s', G ((M.empirical N ω).saAbsorbing p c) s' ∂(M.P p) -
        ∫ s', G ((M.empirical N ω).saAbsorbing p c) s' ∂((M.empirical N ω).P p)|} ≤
      2 * exp (-x) := by
  simpa only [empirical_P] using
    measureReal_sqrt_variance_add_le_abs_integral_sub_integral_empiricalKernel_le_of_indepFun M.P
      (measurable_comp_empirical M N fun M' ↦ G (M'.saAbsorbing p c))
      (indepFun_comp_saAbsorbing_empirical M N G p c) hlu hG N x

end Decoupling

/-- The confidence count of [LWCGC24, (92)]: for `m ≤ 2 ℓ`, the probability `2 exp (-L)` of each
bad event of Bernstein's inequality at the level `L = log (4 S A K ℓ / δ)` is at most `δ` divided
by the number `S A K m` of pairs, points of the grid and levels. -/
private lemma two_mul_exp_neg_log_le {S A K ℓ m δ : ℝ} (hS : 0 < S) (hA : 0 < A) (hK : 0 < K)
    (hℓ : 0 < ℓ) (hm : 0 < m) (hδ : 0 < δ) (hmℓ : m ≤ 2 * ℓ) :
    2 * exp (-log (4 * S * A * K * ℓ / δ)) ≤ δ / (S * A * (K * m)) := by
  rw [Real.exp_neg, Real.exp_log (by positivity), inv_div, le_div_iff₀ (by positivity)]
  calc 2 * (δ / (4 * S * A * K * ℓ)) * (S * A * (K * m)) = δ * m / (2 * ℓ) := by
        field_simp
        ring
    _ ≤ δ * (2 * ℓ) / (2 * ℓ) := by gcongr
    _ = δ := by field_simp

section Generative

variable [Fintype 𝓢] [Fintype 𝓐] [Nonempty 𝓢] [Nonempty 𝓐]

/-- **Evaluation of the separated plug-in policy** [LWCGC24, Lemma 5, first claim]. Under the
generative model of an MDP `M` with `S` states, `A` actions and rewards in `[0, C]`, let `γ < 1`,
`η > 0`, `K > 0` with `8 (1 + γ) C / (1 - γ) ≤ (1 - γ) η K`, `δ > 0` and
`L = log (4 S A (K + 1) log (e / (1 - γ)) / δ)`. If `18 L ≤ (1 - γ) N`, then with probability at
least `1 - δ`, whenever the optimal action values of the empirical MDP `M̂ = M.empirical N ω` are
separated by `η` at its greedy optimal policy `π̂` (`Q̂* (s, a) + η ≤ Q̂* (s, π̂ s)` for every
`a ≠ π̂ s`), the value of `π̂` in `M̂` satisfies `|v̂^π̂ - v^π̂| ≤ 3 C √(2 L / (N (1 - γ) ^ 3))` at
every state.

For rewards in `[0, 1]`, [LWCGC24] uses the grid of spacing `(1 - γ) η / 4` on
`[-1 / (1 - γ), 1 / (1 - γ)]`, with `K ≤ 8 / ((1 - γ) ^ 2 η)`, assumes
`N ≥ 16 e ^ 2 β₁ / (1 - γ)` for the corresponding `β₁ = 2 L`, and proves the bound with the
constant `6` in place of `3`; the constant `3` comes from the constant `1` of
`ProbabilityTheory.Kernel.integral_sqrt_variance_integral_resolvent_le`. -/
theorem le_streamMeasure_forall_abs_value_empirical_sub_value_le_of_separated {M : MDP 𝓢 𝓐 ℝ}
    {γ : ℝ≥0} (hγ : γ < 1) {C : ℝ} (hR : ∀ p, ∀ᵐ x ∂(M.R p), x ∈ Set.Icc 0 C) {η : ℝ}
    (hη : 0 < η) {K : ℕ} (hK : 0 < K) (hKη : 8 * ((1 + γ) * C / (1 - γ)) ≤ (1 - γ) * η * K)
    (N : ℕ) [NeZero N] {δ : ℝ} (hδ : 0 < δ)
    (hN : 18 * log (4 * Fintype.card 𝓢 * Fintype.card 𝓐 * (K + 1) * log (exp 1 / (1 - γ)) / δ) ≤
      (1 - γ) * N) :
    1 - δ ≤ (streamMeasure M.P).real {ω |
      (∀ s a, a ≠ (M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ) s →
        (M.empirical N ω).optQValue γ (s, a) + η ≤
          (M.empirical N ω).optQValue γ
            (s, (M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ) s)) →
      ∀ s, |(M.empirical N ω).value γ (Kernel.deterministic
            ((M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ))
            (measurable_of_finite _)) s -
          M.value γ (Kernel.deterministic
            ((M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ))
            (measurable_of_finite _)) s| ≤
        3 * C * √(2 * log (4 * Fintype.card 𝓢 * Fintype.card 𝓐 * (K + 1) *
          log (exp 1 / (1 - γ)) / δ) / (N * (1 - γ) ^ 3))} := by
  -- For every pair `p`, point `k` of the grid and level `l ≤ m = ⌈log (1 - γ)⁻¹⌉₊`, Bernstein's
  -- inequality at `p` holds for the `l`-th level `V d l` of [LWCGC24, Lemma 2] of the greedy
  -- optimal policy `d` of the `p`-absorbing modification with the `k`-th reward of the grid, which
  -- is independent of the samples of `p`. Under the separation, at `p = (s, π̂ s)` one of these
  -- policies is `π̂` [LWCGC24, Lemma 4], so that the deviations of the empirical state kernel of
  -- `π̂` needed by [LWCGC24, Lemma 2] hold at every state.
  rcases le_or_gt 1 δ with hδ1 | hδ1
  · exact (sub_nonpos.2 hδ1).trans measureReal_nonneg
  have hN₀ : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hγ₁ : (γ : ℝ) < 1 := hγ
  have hu : 0 < 1 - (γ : ℝ) := sub_pos.2 hγ₁
  have hRabs (p : 𝓢 × 𝓐) : ∀ᵐ x ∂(M.R p), |x| ≤ C :=
    (hR p).mono fun x hx ↦ abs_le.2 ⟨by linarith [hx.1, hx.2], hx.2⟩
  have hRe (ω : ℕ → 𝓢 × 𝓐 → 𝓢) : ∀ p, ∀ᵐ x ∂((M.empirical N ω).R p), |x| ≤ C := by
    simpa using hRabs
  -- the reward of a deterministic policy `d` is in `[0, C]`
  have hf₀ (d : 𝓢 → 𝓐) (s : 𝓢) :
      0 ≤ M.policyReward (Kernel.deterministic d (measurable_of_finite d)) s := by
    rw [policyReward_deterministic, meanReward_def]
    exact integral_nonneg_of_ae ((hR _).mono fun x hx ↦ hx.1)
  have hfC (d : 𝓢 → 𝓐) (s : 𝓢) :
      M.policyReward (Kernel.deterministic d (measurable_of_finite d)) s ≤ C :=
    (le_abs_self _).trans (abs_policyReward_le hRabs s)
  -- both values of `d` are resolvents of the same reward
  have hv (ω : ℕ → 𝓢 × 𝓐 → 𝓢) (d : 𝓢 → 𝓐) (s : 𝓢) :
      (M.empirical N ω).value γ (Kernel.deterministic d (measurable_of_finite d)) s -
          M.value γ (Kernel.deterministic d (measurable_of_finite d)) s =
        ∫ y, M.policyReward (Kernel.deterministic d (measurable_of_finite d)) y
            ∂(Kernel.resolvent γ
              ((M.empirical N ω).stateKernel (Kernel.deterministic d (measurable_of_finite d))) s) -
          ∫ y, M.policyReward (Kernel.deterministic d (measurable_of_finite d)) y
            ∂(Kernel.resolvent γ (M.stateKernel (Kernel.deterministic d (measurable_of_finite d)))
              s) := by
    have hfe : (M.empirical N ω).policyReward (Kernel.deterministic d (measurable_of_finite d)) =
        M.policyReward (Kernel.deterministic d (measurable_of_finite d)) :=
      funext fun s ↦ by rw [policyReward_def, empirical_R, policyReward_def]
    rw [value_eq_integral_resolvent hγ (hRe ω), value_eq_integral_resolvent hγ hRabs, hfe]
  have hC : 0 ≤ C := by
    obtain ⟨x, hx⟩ := (hR (Classical.arbitrary (𝓢 × 𝓐))).exists
    exact hx.1.trans hx.2
  -- for `C = 0` all the values vanish, and for `γ = 0` both values are the reward
  rcases hC.eq_or_lt with rfl | hC₀
  · refine le_trans ?_ (measureReal_mono (s₁ := Set.univ) (fun ω _ _ s ↦ ?_) (measure_ne_top _ _))
    · rw [probReal_univ]
      linarith
    · have h₁ := abs_value_le (π := Kernel.deterministic
        ((M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ)) (measurable_of_finite _))
        hγ (hRe ω) s
      have h₂ := abs_value_le (π := Kernel.deterministic
        ((M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ)) (measurable_of_finite _))
        hγ hRabs s
      rw [zero_div] at h₁ h₂
      rw [mul_zero, zero_mul]
      exact (abs_sub _ _).trans (by linarith)
  rcases eq_or_ne γ 0 with rfl | hγ₀
  · refine le_trans ?_ (measureReal_mono (s₁ := Set.univ) (fun ω _ _ s ↦ ?_) (measure_ne_top _ _))
    · rw [probReal_univ]
      linarith
    · rw [hv, Kernel.resolvent_zero, Kernel.resolvent_zero, Kernel.id_apply, sub_self, abs_zero]
      positivity
  have hγ₀' : 0 < γ := pos_iff_ne_zero.2 hγ₀
  -- the confidence level `L`, the accuracy `ε = 2 L / N` and the number `m` of levels
  set S : ℝ := (Fintype.card 𝓢 : ℝ)
  set A : ℝ := (Fintype.card 𝓐 : ℝ)
  set lg : ℝ := log (exp 1 / (1 - γ)) with hlg_def
  have hlg : lg = 1 + log (1 - γ)⁻¹ := by
    rw [hlg_def, Real.log_div (Real.exp_pos 1).ne' hu.ne', Real.log_exp, Real.log_inv]
    ring
  have hy : 0 ≤ log (1 - (γ : ℝ))⁻¹ :=
    Real.log_nonneg (one_le_inv₀ hu |>.2 (by linarith [γ.coe_nonneg]))
  set L := log (4 * S * A * (K + 1) * lg / δ)
  set ε := 2 * L / N with hε_def
  set m := ⌈log (1 - (γ : ℝ))⁻¹⌉₊
  have hm : log (1 - (γ : ℝ))⁻¹ ≤ m := Nat.le_ceil _
  have hm' : (m : ℝ) + 1 ≤ 2 * lg := by
    have := Nat.ceil_lt_add_one hy
    rw [hlg]
    linarith
  have hS : 1 ≤ S := Nat.one_le_cast.2 Fintype.card_pos
  have hA : 1 ≤ A := Nat.one_le_cast.2 Fintype.card_pos
  have hK₁ : (1 : ℝ) ≤ K + 1 := by linarith [(Nat.cast_nonneg K : (0 : ℝ) ≤ K)]
  have hlg1 : 1 ≤ lg := by rw [hlg]; linarith
  have hL : 0 ≤ L := Real.log_nonneg (by
    rw [le_div_iff₀ hδ, one_mul]
    have h₁ : 1 ≤ S * A := one_le_mul_of_one_le_of_one_le hS hA
    have h₂ : 1 ≤ S * A * (K + 1) := one_le_mul_of_one_le_of_one_le h₁ hK₁
    have h₃ : 1 ≤ S * A * (K + 1) * lg := one_le_mul_of_one_le_of_one_le h₂ hlg1
    nlinarith)
  -- the levels `V d l` of a deterministic policy `d` and their bounds `b l`
  obtain ⟨V, hV₀, hV⟩ : ∃ V : (𝓢 → 𝓐) → ℕ → 𝓢 → ℝ,
      (∀ d x, V d 0 x = ∫ y, M.policyReward (Kernel.deterministic d (measurable_of_finite d)) y
        ∂(Kernel.resolvent γ (M.stateKernel (Kernel.deterministic d (measurable_of_finite d))) x)) ∧
      ∀ d l x, V d (l + 1) x =
        ∫ y, √Var[V d l; M.stateKernel (Kernel.deterministic d (measurable_of_finite d)) y]
          ∂(Kernel.resolvent γ (M.stateKernel (Kernel.deterministic d (measurable_of_finite d)))
            x) :=
    ⟨fun d l ↦ Nat.rec
      (fun x ↦ ∫ y, M.policyReward (Kernel.deterministic d (measurable_of_finite d)) y
        ∂(Kernel.resolvent γ (M.stateKernel (Kernel.deterministic d (measurable_of_finite d))) x))
      (fun _ W x ↦ ∫ y, √Var[W; M.stateKernel (Kernel.deterministic d (measurable_of_finite d)) y]
        ∂(Kernel.resolvent γ (M.stateKernel (Kernel.deterministic d (measurable_of_finite d))) x))
      l, fun _ _ ↦ rfl, fun _ _ _ ↦ rfl⟩
  set b : ℕ → ℝ := fun l ↦ C / ((1 - γ) * (γ * √(1 - γ)) ^ l)
  have hbpos (l : ℕ) : 0 < b l :=
    div_pos hC₀ (mul_pos hu (pow_pos (mul_pos (NNReal.coe_pos.2 hγ₀') (Real.sqrt_pos.2 hu)) _))
  have hVb (d : 𝓢 → 𝓐) : ∀ l y, V d l y ∈ Set.Icc 0 (b l) :=
    Kernel.mem_Icc_of_eq_integral_resolvent_sqrt_variance hγ₀' hγ (measurable_of_finite _)
      (hf₀ d) (hfC d) (hV₀ d) (hV d)
  -- the grid of rewards on `[-B, B]`, with `B` bounding `Q̂* (s, a) - γ v̂* s`
  set B := (1 + γ) * C / (1 - γ)
  -- the levels of the greedy optimal policies of the absorbing modifications on the grid
  set W : (𝓢 × 𝓐) × Fin (K + 1) × Fin (m + 1) → MDP 𝓢 𝓐 ℝ → 𝓢 → ℝ := fun i M' ↦
    V ((M'.saAbsorbing i.1 (-B + 2 * B * (i.2.1 : ℕ) / K)).greedy γ
      ((M'.saAbsorbing i.1 (-B + 2 * B * (i.2.1 : ℕ) / K)).optValue γ)) i.2.2 with hW_def
  -- the bad events: Bernstein's inequality fails for one of them
  set bad : (𝓢 × 𝓐) × Fin (K + 1) × Fin (m + 1) → Set (ℕ → 𝓢 × 𝓐 → 𝓢) := fun i ↦ {ω |
    √(2 * Var[W i (M.empirical N ω); M.P i.1] * L / N) + (b i.2.2 - 0) * L / (3 * N) ≤
      |∫ s', W i (M.empirical N ω) s' ∂(M.P i.1) -
        ∫ s', W i (M.empirical N ω) s' ∂((M.empirical N ω).P i.1)|} with hbad_def
  have h_bad (i : (𝓢 × 𝓐) × Fin (K + 1) × Fin (m + 1)) :
      (streamMeasure M.P).real (bad i) ≤
        δ / Fintype.card ((𝓢 × 𝓐) × Fin (K + 1) × Fin (m + 1)) := by
    refine (measureReal_sqrt_variance_add_le_abs_integral_sub_integral_empirical_saAbsorbing_le M N
      (fun M'' ↦ V (M''.greedy γ (M''.optValue γ)) i.2.2) (hbpos i.2.2) i.1 _
      (fun ω s' ↦ hVb _ _ _) L).trans ?_
    simp only [Fintype.card_prod, Fintype.card_fin, Nat.cast_mul, Nat.cast_add, Nat.cast_one]
    exact two_mul_exp_neg_log_le (by positivity) (by positivity) (by positivity) (by positivity)
      (by positivity) hδ hm'
  -- outside the bad events, the deterministic comparison of resolvents applies to `π̂`
  refine (one_sub_le_measureReal_compl_iUnion hδ.le h_bad).trans
    (measureReal_mono fun ω hω ↦ ?_)
  simp only [Set.mem_compl_iff, Set.mem_iUnion, not_exists] at hω
  intro hsep x
  set g := (M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ)
  have hε₀ : 0 ≤ ε := by positivity
  have hε9 : 9 * ε ≤ 1 - γ := by
    rw [hε_def, ← mul_div_assoc, div_le_iff₀ hN₀]
    linarith
  rw [hv]
  refine (Kernel.abs_integral_resolvent_sub_integral_resolvent_le_of_abs_sub_le
    ((M.empirical N ω).stateKernel (Kernel.deterministic g (measurable_of_finite g))) hγ₀' hγ
    (measurable_of_finite _) (hf₀ g) (hfC g) (hV₀ g) (hV g) hm hε₀ hε9 (fun l hl y ↦ ?_)
    x).trans_eq ?_
  · -- `g` is the greedy optimal policy of a modification at `(y, g y)` on the grid
    obtain ⟨k, hk⟩ := exists_greedy_optValue_saAbsorbing_eq (p := (y, g y)) hγ (hRe ω) hη hsep
      (abs_optQValue_sub_mul_optValue_le hγ (hRe ω) _) hK hKη
    have h := hω ((y, g y), k, ⟨l, Nat.lt_succ_of_le hl⟩)
    simp only [hbad_def, hW_def, Set.mem_ofPred_eq, not_le] at h
    rw [hk] at h
    rw [stateKernel_deterministic_apply, stateKernel_deterministic_apply, abs_sub_comm]
    refine h.le.trans (add_le_add (le_of_eq ?_) ?_)
    · rw [hε_def]
      congr 1
      ring
    · have hbL : 0 ≤ b l * L / N := by have := hbpos l; positivity
      calc (b l - 0) * L / (3 * N) = b l * L / N / 3 := by
            rw [sub_zero, div_div, mul_comm (3 : ℝ)]
        _ ≤ b l * L / N * 2 := by linarith
        _ = ε * b l := by rw [hε_def]; ring
  · rw [hε_def, div_div]

/-- **Suboptimality of the separated plug-in policy** [LWCGC24, Lemma 5, second claim]. Under the
generative model of an MDP `M` with `S` states, `A` actions and rewards in `[0, C]`, let `γ < 1`,
`η > 0`, `K > 0` with `8 (1 + γ) C / (1 - γ) ≤ (1 - γ) η K`, `δ > 0` and
`L = log (8 S A (K + 1) log (e / (1 - γ)) / δ)`. If `18 L ≤ (1 - γ) N`, then with probability at
least `1 - δ`, whenever the optimal action values of the empirical MDP `M̂ = M.empirical N ω` are
separated by `η` at its greedy optimal policy `π̂` (`Q̂* (s, a) + η ≤ Q̂* (s, π̂ s)` for every
`a ≠ π̂ s`), the value of `π̂` in `M` satisfies `v* - v^π̂ ≤ 6 C √(2 L / (N (1 - γ) ^ 3))` at every
state.

[LWCGC24] proves the bound with the constant `12` in place of `6`, for rewards in `[0, 1]` and
under its condition on `N`. -/
theorem le_streamMeasure_forall_optValue_sub_value_le_of_separated {M : MDP 𝓢 𝓐 ℝ}
    {γ : ℝ≥0} (hγ : γ < 1) {C : ℝ} (hR : ∀ p, ∀ᵐ x ∂(M.R p), x ∈ Set.Icc 0 C) {η : ℝ}
    (hη : 0 < η) {K : ℕ} (hK : 0 < K) (hKη : 8 * ((1 + γ) * C / (1 - γ)) ≤ (1 - γ) * η * K)
    (N : ℕ) [NeZero N] {δ : ℝ} (hδ : 0 < δ)
    (hN : 18 * log (8 * Fintype.card 𝓢 * Fintype.card 𝓐 * (K + 1) * log (exp 1 / (1 - γ)) / δ) ≤
      (1 - γ) * N) :
    1 - δ ≤ (streamMeasure M.P).real {ω |
      (∀ s a, a ≠ (M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ) s →
        (M.empirical N ω).optQValue γ (s, a) + η ≤
          (M.empirical N ω).optQValue γ
            (s, (M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ) s)) →
      ∀ s, M.optValue γ s - M.value γ (Kernel.deterministic
            ((M.empirical N ω).greedy γ ((M.empirical N ω).optValue γ))
            (measurable_of_finite _)) s ≤
        6 * C * √(2 * log (8 * Fintype.card 𝓢 * Fintype.card 𝓐 * (K + 1) *
          log (exp 1 / (1 - γ)) / δ) / (N * (1 - γ) ^ 3))} := by
  -- `v* - v^π̂ = (v^π* - v̂^π*) + (v̂^π* - v̂^π̂) + (v̂^π̂ - v^π̂)` for the greedy optimal policy
  -- `π*` of `M`: the first term is bounded by [LWCGC24, Lemma 1], since `π*` does not depend on
  -- the samples, the second one is nonpositive by optimality of `π̂` in `M̂`, and the third one is
  -- bounded by the first claim. Each bound holds with probability at least `1 - δ / 2`.
  set μ := streamMeasure M.P
  have hRabs (p : 𝓢 × 𝓐) : ∀ᵐ x ∂(M.R p), |x| ≤ C :=
    (hR p).mono fun x hx ↦ abs_le.2 ⟨by linarith [hx.1, hx.2], hx.2⟩
  have hRe (ω : ℕ → 𝓢 × 𝓐 → 𝓢) : ∀ p, ∀ᵐ x ∂((M.empirical N ω).R p), |x| ≤ C := by
    simpa using hRabs
  set S : ℝ := (Fintype.card 𝓢 : ℝ)
  set A : ℝ := (Fintype.card 𝓐 : ℝ)
  set lg : ℝ := log (exp 1 / (1 - γ)) with hlg_def
  set L := log (8 * S * A * (K + 1) * lg / δ)
  have hS : 1 ≤ S := Nat.one_le_cast.2 Fintype.card_pos
  have hA : 1 ≤ A := Nat.one_le_cast.2 Fintype.card_pos
  have hK₁ : (1 : ℝ) ≤ K + 1 := by linarith [(Nat.cast_nonneg K : (0 : ℝ) ≤ K)]
  have hlg₀ : 0 < lg := by
    rw [hlg_def]
    refine Real.log_pos ((one_lt_div (sub_pos.2 (NNReal.coe_lt_one.2 hγ))).2 ?_)
    linarith [Real.add_one_le_exp (1 : ℝ), γ.coe_nonneg]
  have h8 : 4 * S * A * (K + 1) * lg / (δ / 2) = 8 * S * A * (K + 1) * lg / δ := by
    field_simp
    ring
  -- the level of [LWCGC24, Lemma 1] at `δ / 2` is at most `L`
  have hL₀ : log (4 * S * lg / (δ / 2)) ≤ L := by
    refine Real.log_le_log (by positivity) ?_
    rw [show 4 * S * lg / (δ / 2) = 8 * S * lg / δ by field_simp; ring]
    refine div_le_div_of_nonneg_right ?_ hδ.le
    have h₁ : 1 ≤ A * (K + 1) := one_le_mul_of_one_le_of_one_le hA hK₁
    calc 8 * S * lg = 8 * S * lg * 1 := (mul_one _).symm
      _ ≤ 8 * S * lg * (A * (K + 1)) := by gcongr
      _ = 8 * S * A * (K + 1) * lg := by ring
  -- the two events, each of probability at least `1 - δ / 2`
  set g : MDP 𝓢 𝓐 ℝ → 𝓢 → 𝓐 := fun M' ↦ M'.greedy γ (M'.optValue γ)
  have hg := measurable_greedy (M := M) (γ := γ) (measurable_optValue hγ hRabs)
  set E₁ := {ω | (∀ s a, a ≠ g (M.empirical N ω) s →
      (M.empirical N ω).optQValue γ (s, a) + η ≤ (M.empirical N ω).optQValue γ
        (s, g (M.empirical N ω) s)) →
    ∀ s, |(M.empirical N ω).value γ (Kernel.deterministic (g (M.empirical N ω))
        (measurable_of_finite _)) s -
      M.value γ (Kernel.deterministic (g (M.empirical N ω)) (measurable_of_finite _)) s| ≤
      3 * C * √(2 * L / (N * (1 - γ) ^ 3))}
  set E₂ := {ω | ∀ s, |(M.empirical N ω).value γ (Kernel.deterministic (g M) hg) s -
      M.value γ (Kernel.deterministic (g M) hg) s| ≤
      3 * C * √(2 * log (4 * S * lg / (δ / 2)) / (N * (1 - γ) ^ 3))}
  have h₁ : 1 - δ / 2 ≤ μ.real E₁ := by
    have h := le_streamMeasure_forall_abs_value_empirical_sub_value_le_of_separated hγ hR hη hK
      hKη N (half_pos hδ) (by rwa [h8])
    rwa [h8] at h
  have h₂ : 1 - δ / 2 ≤ μ.real E₂ :=
    le_streamMeasure_forall_abs_value_empirical_sub_value_le hγ hR _ N (half_pos hδ)
      ((mul_le_mul_of_nonneg_left hL₀ (by norm_num)).trans hN)
  have hE₂ : MeasurableSet E₂ :=
    measurableSet_setOfPred.2 (measurable_comp_empirical M N fun M' ↦ ∀ s,
      |M'.value γ (Kernel.deterministic (g M) hg) s - M.value γ (Kernel.deterministic (g M) hg) s| ≤
        3 * C * √(2 * log (4 * S * lg / (δ / 2)) / (N * (1 - γ) ^ 3)))
  have h_inter : 1 - δ ≤ μ.real (E₁ ∩ E₂) := by
    have h := measureReal_union_add_inter (μ := μ) (s := E₁) hE₂
    have h₃ : μ.real (E₁ ∪ E₂) ≤ 1 := measureReal_le_one
    linarith
  refine h_inter.trans (measureReal_mono (fun ω ⟨hω₁, hω₂⟩ hsep s ↦ ?_) (measure_ne_top _ _))
  have hv₁ := abs_le.1 (hω₁ hsep s)
  have hv₂ := abs_le.1 (hω₂ s)
  -- `π*` is optimal in `M`, and `π̂` is optimal in `M̂`
  have hopt : M.value γ (Kernel.deterministic (g M) hg) = M.optValue γ :=
    value_greedy_optValue hγ hRabs hg
  have hle : (M.empirical N ω).value γ (Kernel.deterministic (g M) hg) s ≤
      (M.empirical N ω).value γ (Kernel.deterministic (g (M.empirical N ω))
        (measurable_of_finite _)) s := by
    exact (value_le_optValue hγ (hRe ω) s).trans_eq
      (congrFun (value_greedy_optValue hγ (hRe ω) (measurable_of_finite _)) s).symm
  have hC : 0 ≤ C := by
    obtain ⟨x, hx⟩ := (hR (s, Classical.arbitrary 𝓐)).exists
    exact hx.1.trans hx.2
  have hsq : 3 * C * √(2 * log (4 * S * lg / (δ / 2)) / (N * (1 - γ) ^ 3)) ≤
      3 * C * √(2 * L / (N * (1 - γ) ^ 3)) := by
    have : 0 ≤ (N : ℝ) * (1 - γ) ^ 3 := by
      have := sub_pos.2 (NNReal.coe_lt_one.2 hγ)
      positivity
    gcongr
  rw [← hopt]
  linarith

end Generative

end Learning.MDP
