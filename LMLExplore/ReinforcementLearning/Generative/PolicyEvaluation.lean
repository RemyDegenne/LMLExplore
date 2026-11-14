/-
Copyright (c) 2026 LMLExplore contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: LMLExplore contributors
-/
module

public import LMLExplore.ReinforcementLearning.Generative.Basic
import LMLExplore.Algebra.Order.Ring.Abs
import LMLExplore.Analysis.SpecialFunctions.Log.Basic
import LMLExplore.Analysis.Convex.Integral
import LMLExplore.Analysis.Convex.Simplex

/-!
# Planning with a generative model: policy evaluation over the full range of sample sizes

Let `M` be an MDP with `S` states, finitely many actions and rewards in `[0, C]`, `γ < 1`, and `π`
a stationary policy fixed independently of the data. Under the generative model
`Bandits.streamMeasure M.P`, the value `v̂^π` of `π` in the empirical MDP `M.empirical N ω` of `N`
samples per state-action pair estimates the value `v^π` of `π` in `M` with accuracy
[LWCGC24, Lemma 1]: for `δ > 0` and `L = log (4 S log (e / (1 - γ)) / δ)`, as soon as
`N ≥ 18 L / (1 - γ)`, with probability at least `1 - δ`,

`|v̂^π - v^π| ≤ 3 C √(2 L / (N (1 - γ) ^ 3))`.

The accuracy is of order `√(L / (N (1 - γ) ^ 3))` over the whole range of sample sizes
`N ≥ 18 L / (1 - γ)`: at `N = 18 L / (1 - γ)`, the bound is the trivial bound `C / (1 - γ)`.

The bound has a deterministic part and a probabilistic part.
* The **deterministic part** [LWCGC24, Lemma 2],
  `ProbabilityTheory.Kernel.abs_integral_resolvent_sub_integral_resolvent_le_of_abs_sub_le` in
  `Probability/Kernel/Resolvent.lean`, in its MDP form
  `Learning.MDP.abs_value_sub_value_le_of_abs_integral_stateKernel_sub_le` in
  `MarkovDecisionProcess/ReturnVariance.lean`, compares the resolvents `v^π = R_γ(P_π) r_π` and
  `v̂^π = R_γ(P̂_π) r_π` of two state kernels when `P̂_π` deviates from `P_π` by a Bernstein-type
  bound on the first `m ≥ log (1 - γ)⁻¹` levels `V = M.valueLevel γ π`, with `V 0 = v^π` and
  `V (l + 1) = R_γ(P_π) √Var_{P_π}(V l)`.
* The **probabilistic part** is Bernstein's inequality for the empirical state kernel of `π`
  (`measureReal_sqrt_variance_add_le_abs_integral_sub_integral_stateKernel_empirical_le`).
  The functions `V l` depend on `M` and `π` only, so that Bernstein's inequality applies to each of
  them, and a union bound over the `m + 1 ≤ 2 log (e / (1 - γ))` levels and the `S` states gives
  the Bernstein-type deviations with `ε = 2 L / N`.

For a stochastic policy, the empirical state kernel at `s` averages over the actions `a` the means
of the samples of `(s, a)`: its `i`-th summand `∑ a, π(a | s) V (ω i (s, a))` is a mixture of
independent samples, whose variance is at most the variance `Var_{P_π s}(V)` of the mixture law
by Jensen's inequality. [LWCGC24] states the lemma for deterministic policies.

## Main statements

* `measureReal_sqrt_variance_add_le_abs_integral_sub_integral_stateKernel_empirical_le`: Bernstein's
  inequality for the empirical state kernel of a stationary policy.
* `Learning.MDP.le_streamMeasure_forall_abs_value_empirical_sub_value_le`: model-based policy
  evaluation with a generative model [LWCGC24, Lemma 1].

## References

* [G. Li, Y. Wei, Y. Chi, Y. Gu and Y. Chen, *Breaking the sample size barrier in model-based
  reinforcement learning with a generative model*, Operations Research, 2024][LWCGC24], Lemmas 1
  and 2 (arXiv:2005.12900)
-/

public section

open MeasureTheory ProbabilityTheory LMLExplore Learning Bandits Finset
open Real (log exp)
open scoped NNReal

namespace Learning.MDP

variable {𝓢 𝓐 𝓡 : Type*} {m𝓢 : MeasurableSpace 𝓢} {m𝓐 : MeasurableSpace 𝓐}
  {m𝓡 : MeasurableSpace 𝓡} [MeasurableSingletonClass 𝓢] [MeasurableSingletonClass 𝓐]

/-- **Bernstein's inequality for the empirical state kernel of a stationary policy.** Under the
generative model `streamMeasure M.P` of an MDP `M` with finitely many states and actions, let `π`
be a stationary policy and `f` a function with values in `[l, u]`, `l < u`, `M.P (s, a)`-almost
everywhere for every action `a`. For `N ≠ 0` and every `x`, the integral of `f` against the state
kernel of `π` in the empirical MDP `M.empirical N ω` deviates from its integral against the state
kernel of `π` in `M` by at least `√(2 Var[f; P_π s] x / N) + (u - l) x / (3 N)` with probability at
most `2 exp (-x)`. -/
theorem measureReal_sqrt_variance_add_le_abs_integral_sub_integral_stateKernel_empirical_le
    [Finite 𝓢] [Finite 𝓐] (M : MDP 𝓢 𝓐 𝓡) (π : Kernel 𝓢 𝓐) [IsMarkovKernel π]
    {f : 𝓢 → ℝ} {l u : ℝ} (hlu : l < u) {s : 𝓢}
    (hfb : ∀ a, ∀ᵐ y ∂(M.P (s, a)), f y ∈ Set.Icc l u) (N : ℕ) [NeZero N] (x : ℝ) :
    (streamMeasure M.P).real {ω | √(2 * Var[f; M.stateKernel π s] * x / N) +
      (u - l) * x / (3 * N) ≤
        |∫ y, f y ∂(M.stateKernel π s) - ∫ y, f y ∂((M.empirical N ω).stateKernel π s)|} ≤
      2 * exp (-x) := by
  -- The empirical integral is the mean of the `N` independent variables
  -- `Z (ω i) = ∑ a, π(a | s) f (ω i (s, a))`, `i < N`, with values in `[l, u]` and mean
  -- `∫ f d(P_π s)`. By Jensen's inequality, their variance is at most the variance `Var[f; P_π s]`
  -- of the mixture `P_π s = ∑ a, π(a | s) M.P (s, a)`, and the bound is Bernstein's inequality for
  -- their mean.
  have := Fintype.ofFinite 𝓐
  have hf : Measurable f := .of_discrete
  -- the weights `w a = π(a | s)` of the actions
  set w : 𝓐 → ℝ := fun a ↦ (π s).real {a} with hw_def
  have hw₀ (a : 𝓐) : 0 ≤ w a := measureReal_nonneg
  have hw₁ : ∑ a, w a = 1 := by simp [hw_def]
  have h_comb {y : 𝓐 → ℝ} (hy : ∀ a, y a ∈ Set.Icc l u) : ∑ a, w a * y a ∈ Set.Icc l u := by
    simpa only [smul_eq_mul] using (convex_Icc l u).sum_mem (fun a _ ↦ hw₀ a) hw₁ fun a _ ↦ hy a
  -- the mean `m = ∫ f d(P_π s)` and the variable `Z`
  set m := ∫ y, f y ∂(M.stateKernel π s) with hm_def
  have hm : m = ∑ a, w a * ∫ y, f y ∂(M.P (s, a)) := integral_stateKernel_eq_sum .of_finite
  have hm_mem : m ∈ Set.Icc l u :=
    hm ▸ h_comb fun a ↦ integral_mem_Icc_of_ae_mem_Icc hf.aemeasurable (hfb a)
  set Z : (𝓢 × 𝓐 → 𝓢) → ℝ := fun ρ ↦ ∑ a, w a * f (ρ (s, a)) with hZ_def
  have hZ : Measurable Z :=
    Finset.measurable_sum _ fun a _ ↦ (hf.comp (measurable_pi_apply (s, a))).const_mul _
  -- the law of the samples `ρ (s, a)` under `Measure.infinitePi M.P` is `M.P (s, a)`
  set μ := Measure.infinitePi M.P with hμ_def
  have h_law (a : 𝓐) := hasLaw_eval_infinitePi M.P (s, a)
  have h_ae : ∀ᵐ ρ ∂μ, ∀ a, f (ρ (s, a)) ∈ Set.Icc l u :=
    ae_all_iff.2 fun a ↦ ae_of_ae_map (h_law a).aemeasurable ((h_law a).map_eq ▸ hfb a)
  have h_int (a : 𝓐) {g : ℝ → ℝ} (hg : Measurable g) {c d : ℝ}
      (hgb : ∀ y ∈ Set.Icc l u, g y ∈ Set.Icc c d) :
      Integrable (fun ρ : 𝓢 × 𝓐 → 𝓢 ↦ g (f (ρ (s, a)))) μ :=
    Integrable.of_mem_Icc c d ((hg.comp hf).comp_aemeasurable (h_law a).aemeasurable)
      (h_ae.mono fun ρ hρ ↦ hgb _ (hρ a))
  have hZm : ∫ ρ, Z ρ ∂μ = m := by
    have h₁ (a : 𝓐) : ∫ ρ, f (ρ (s, a)) ∂μ = ∫ y, f y ∂(M.P (s, a)) :=
      (h_law a).integral_comp hf.aestronglyMeasurable
    have hi (a : 𝓐) : Integrable (fun ρ : 𝓢 × 𝓐 → 𝓢 ↦ f (ρ (s, a))) μ :=
      h_int a (g := fun y ↦ y) measurable_id' fun y hy ↦ hy
    simp only [hZ_def]
    rw [integral_finsetSum _ fun a _ ↦ (hi a).const_mul _]
    simp_rw [integral_const_mul, h₁, ← hm]
  -- the centred variable is bounded by `u - l`
  have hZb : ∀ᵐ ρ ∂μ, |Z ρ - m| ≤ u - l := by
    filter_upwards [h_ae] with ρ hρ
    have hZρ := h_comb hρ
    rw [abs_le]
    constructor <;> simp only [hZ_def] <;> linarith [hZρ.1, hZρ.2, hm_mem.1, hm_mem.2]
  -- by Jensen's inequality, the variance of `Z` is at most `Var[f; P_π s]`
  have hZσ : ∫ ρ, (Z ρ - m) ^ 2 ∂μ ≤ Var[f; M.stateKernel π s] := by
    have h_pt (ρ : 𝓢 × 𝓐 → 𝓢) : (Z ρ - m) ^ 2 ≤ ∑ a, w a * (f (ρ (s, a)) - m) ^ 2 := by
      have hZρ : Z ρ - m = ∑ a, w a * (f (ρ (s, a)) - m) := by
        simp only [hZ_def, mul_sub, sum_sub_distrib, ← sum_mul, hw₁, one_mul]
      rw [hZρ]
      exact sq_dotProduct_le_dotProduct_sq_of_mem_simplex (mem_simplex_iff.2 ⟨hw₀, hw₁⟩)
        fun a ↦ f (ρ (s, a)) - m
    have h_int_sq (a : 𝓐) := h_int a (g := fun y ↦ (y - m) ^ 2) (by fun_prop)
      fun y hy ↦ sq_sub_mem_Icc_of_mem_Icc hy hm_mem
    have h_int_Z : Integrable (fun ρ ↦ (Z ρ - m) ^ 2) μ :=
      Integrable.of_mem_Icc 0 ((u - l) ^ 2) ((hZ.sub_const m).pow_const 2).aemeasurable
        (hZb.mono fun ρ hρ ↦
          ⟨sq_nonneg _, by rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) hρ 2⟩)
    calc ∫ ρ, (Z ρ - m) ^ 2 ∂μ
        ≤ ∫ ρ, ∑ a, w a * (f (ρ (s, a)) - m) ^ 2 ∂μ :=
          integral_mono h_int_Z (integrable_finsetSum _ fun a _ ↦ (h_int_sq a).const_mul _) h_pt
      _ = ∑ a, w a * ∫ y, (f y - m) ^ 2 ∂(M.P (s, a)) := by
          rw [integral_finsetSum _ fun a _ ↦ (h_int_sq a).const_mul _]
          refine sum_congr rfl fun a _ ↦ ?_
          rw [integral_const_mul]
          congr 1
          exact (h_law a).integral_comp (f := fun y ↦ (f y - m) ^ 2)
            ((hf.sub_const m).pow_const 2).aestronglyMeasurable
      _ = Var[f; M.stateKernel π s] := by
          rw [← integral_stateKernel_eq_sum (v := fun y ↦ (f y - m) ^ 2) .of_finite,
            variance_eq_integral hf.aemeasurable]
  refine le_trans (measureReal_mono fun ω (hω : _ ≤ _) ↦ ?_)
    (measureReal_sqrt_add_le_abs_sub_inv_mul_sum_le M.P hZ (sub_pos.2 hlu) hZm hZb hZσ N x)
  -- the empirical integral is the mean of the `Z (ω i)`, `i < N`
  have h_emp : ∫ y, f y ∂((M.empirical N ω).stateKernel π s) =
      (N : ℝ)⁻¹ * ∑ i ∈ range N, Z (ω i) := by
    rw [integral_stateKernel_eq_sum .of_finite, mul_sum]
    simp_rw [empirical_P, integral_empiricalKernel N ω _ hf.stronglyMeasurable, smul_eq_mul,
      hZ_def, mul_sum]
    rw [sum_comm]
    exact sum_congr rfl fun i _ ↦ sum_congr rfl fun a _ ↦ by ring
  rw [Set.mem_ofPred_eq, ← h_emp]
  exact hω

/-- **Model-based policy evaluation with a generative model** [LWCGC24, Lemma 1]. Under the
generative model of an MDP `M` with `S` states, finitely many actions and rewards in `[0, C]`, let
`γ < 1`, `π` a stationary policy, `δ > 0` and `L = log (4 S log (e / (1 - γ)) / δ)`. If
`18 L ≤ (1 - γ) N`, then with probability at least `1 - δ`, the value `v̂^π` of `π` in the empirical
MDP `M.empirical N ω` satisfies `|v̂^π - v^π| ≤ 3 C √(2 L / (N (1 - γ) ^ 3))` at every state.

[LWCGC24] assumes `N ≥ 32 e ^ 2 L / (1 - γ)`, which implies `18 L ≤ (1 - γ) N`, and proves the
bound `6 √(2 L / (N (1 - γ) ^ 3))` for rewards in `[0, 1]` and deterministic policies; the constant
`3` comes from the constant `1` of
`ProbabilityTheory.Kernel.integral_sqrt_variance_integral_resolvent_le`, which improves the
constant `4` of [LWCGC24, Lemma 11]. -/
theorem le_streamMeasure_forall_abs_value_empirical_sub_value_le [Fintype 𝓢] [Finite 𝓐]
    {M : MDP 𝓢 𝓐 ℝ} {γ : ℝ≥0}
    (hγ : γ < 1) {C : ℝ} (hR : ∀ p, ∀ᵐ x ∂(M.R p), x ∈ Set.Icc 0 C) (π : Kernel 𝓢 𝓐)
    [IsMarkovKernel π] (N : ℕ) [NeZero N] {δ : ℝ} (hδ : 0 < δ)
    (hN : 18 * log (4 * Fintype.card 𝓢 * log (exp 1 / (1 - γ)) / δ) ≤ (1 - γ) * N) :
    1 - δ ≤ (streamMeasure M.P).real {ω | ∀ s,
      |(M.empirical N ω).value γ π s - M.value γ π s| ≤
        3 * C * √(2 * log (4 * Fintype.card 𝓢 * log (exp 1 / (1 - γ)) / δ) /
          (N * (1 - γ) ^ 3))} := by
  -- We apply the deterministic part `abs_value_sub_value_le_of_abs_integral_stateKernel_sub_le`
  -- with `m = ⌈log (1 - γ)⁻¹⌉₊` levels. The functions `V l` of the levels only depend on `M` and
  -- `π`, so that Bernstein's inequality for the empirical state kernel of `π`, with a union bound
  -- over the `(m + 1) S ≤ 2 S log (e / (1 - γ))` levels and states, gives the deviations it needs.
  rcases le_or_gt 1 δ with hδ1 | hδ1
  · exact (sub_nonpos.2 hδ1).trans measureReal_nonneg
  have hN₀ : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have hRabs (p : 𝓢 × 𝓐) : ∀ᵐ x ∂(M.R p), |x| ≤ C :=
    (hR p).mono fun x hx ↦ (abs_of_nonneg hx.1).trans_le hx.2
  set P := M.stateKernel π with hP_def
  -- for `γ = 0`, both values are the reward
  rcases eq_or_ne γ 0 with rfl | hγ₀
  · calc 1 - δ ≤ (streamMeasure M.P).real Set.univ := by simp [hδ.le]
      _ ≤ _ := measureReal_mono fun ω _ ↦ by
        simp only [Set.mem_ofPred_eq]
        intro s
        have hf := policyReward_mem_Icc (π := π) hR s
        have hC : 0 ≤ C := hf.1.trans hf.2
        rw [value_empirical_zero, sub_self, abs_zero]
        positivity
  have hγ₀' : 0 < γ := pos_iff_ne_zero.2 hγ₀
  -- the confidence level `L` and the number `m` of levels
  set S : ℝ := (Fintype.card 𝓢 : ℝ) with hS_def
  set lg : ℝ := log (exp 1 / (1 - γ)) with hlg_def
  set L := log (4 * S * lg / δ) with hL_def
  set m := ⌈log (1 - (γ : ℝ))⁻¹⌉₊ with hm_def
  have hm' : (m : ℝ) + 1 ≤ 2 * lg := Real.natCeil_log_inv_one_sub_add_one_le γ.coe_nonneg hγ
  have hS (s : 𝓢) : 1 ≤ S := Nat.one_le_cast.2 (Fintype.card_pos_iff.2 ⟨s⟩)
  have hlg1 : 1 ≤ 2 * lg := by linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  have hlg₀ : 0 < lg := by linarith
  have hL (s : 𝓢) : 0 ≤ L := Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith [hS s])
  -- the levels `V l` and their bounds `b l`
  set V : ℕ → 𝓢 → ℝ := M.valueLevel γ π with hV_def
  set b : ℕ → ℝ := fun l ↦ C / ((1 - γ) * (γ * √(1 - γ)) ^ l) with hb_def
  have hVb : ∀ l y, V l y ∈ Set.Icc 0 (b l) := valueLevel_mem_Icc hγ₀' hγ hR π
  -- the bad events: a Bernstein deviation fails at some level `l ≤ m` and state `s`
  set bad : Fin (m + 1) × 𝓢 → Set (ℕ → 𝓢 × 𝓐 → 𝓢) := fun i ↦ {ω |
    √(2 * Var[V i.1; P i.2] * L / N) + b i.1 * L / (3 * N) <
      |∫ y, V i.1 y ∂((M.empirical N ω).stateKernel π i.2) - ∫ y, V i.1 y ∂(P i.2)|}
    with hbad_def
  have h_bad (i : Fin (m + 1) × 𝓢) :
      (streamMeasure M.P).real (bad i) ≤ δ / Fintype.card (Fin (m + 1) × 𝓢) := by
    have hS := hS i.2
    have hcard : (Fintype.card (Fin (m + 1) × 𝓢) : ℝ) = (m + 1) * S := by
      simp [hS_def, Fintype.card_prod]
    have hb₀ : 0 ≤ b i.1 := (hVb i.1 i.2).1.trans (hVb i.1 i.2).2
    rcases hb₀.eq_or_lt with hb0 | hbpos
    · -- if `b l = 0`, then `V l = 0` and the bad event is empty
      have hV0 : V i.1 = 0 :=
        funext fun y ↦ le_antisymm ((hVb i.1 y).2.trans_eq hb0.symm) (hVb i.1 y).1
      have h_empty : bad i = ∅ := by
        ext ω
        simp only [hbad_def, Set.mem_ofPred_eq, hV0, Pi.zero_apply, integral_zero, sub_self,
          abs_zero, ← hb0, zero_mul, zero_div, add_zero, Set.mem_empty_iff_false, iff_false,
          not_lt]
        exact Real.sqrt_nonneg _
      rw [h_empty, measureReal_empty]
      positivity
    -- Bernstein's inequality at `x = L`, with `2 exp (-L) ≤ δ / ((m + 1) S)`
    refine le_trans (measureReal_mono (fun ω (hω : _ < _) ↦ ?_) (measure_ne_top _ _))
      ((measureReal_sqrt_variance_add_le_abs_integral_sub_integral_stateKernel_empirical_le M π
        (l := 0) (u := b i.1) (s := i.2) hbpos (fun a ↦ .of_forall (hVb i.1)) N L).trans ?_)
    · rw [Set.mem_ofPred_eq, abs_sub_comm, sub_zero]
      exact hω.le
    · rw [hcard, hL_def]
      exact Real.mul_exp_neg_log_div_le two_pos (by positivity) hδ (by nlinarith)
  -- outside the bad events, the deterministic part applies
  refine (one_sub_le_measureReal_compl_iUnion hδ.le h_bad).trans
    (measureReal_mono fun ω hω ↦ ?_)
  simp only [Set.mem_compl_iff, Set.mem_iUnion, not_exists] at hω
  simp only [Set.mem_ofPred_eq]
  intro s
  refine abs_value_sub_value_le_of_abs_integral_stateKernel_sub_le (empirical_R M N ω) hγ₀' hγ hR
    π (Nat.le_ceil _) hN₀ (hL s) hN (fun l hl x ↦ ?_) s
  have h := hω (⟨l, Nat.lt_succ_of_le hl⟩, x)
  simp only [hbad_def, Set.mem_ofPred_eq, not_lt] at h
  exact h

end Learning.MDP
