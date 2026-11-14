/-
Copyright (c) 2026 LMLExplore contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: LMLExplore contributors
-/
module

public import LMLExplore.Probability.Martingale.RateFunction
public import LMLExplore.Probability.Moments.KLChernoff
import LMLExplore.Probability.Distributions.Bernoulli

/-!
# Conditional Bernoulli domination and time-uniform Chernoff bounds in relative-entropy form

A random variable `X` with values in `[0, 1]` whose conditional mean given a σ-algebra `m` is at
most `q` is *conditionally dominated above* by the Bernoulli law `Ber((1 : ℝ), 0, q)`: for
`t ≥ 0`, `E[exp (t X) | m] ≤ 1 - q + q exp t`, the moment generating function of that law
[GC11, Lemma 9; LS, Note 10.3.1]. The proof is the unconditional one
(`LMLExplore.mgfDominatedOn_Ici_bernoulliMeasure`, by convexity of `exp` on `[0, 1]`), applied to
the conditional law of `X` given `m`, the conditional expectation kernel. Symmetrically, if the
conditional mean is at least `q`, then `X` is conditionally dominated below by `Ber((1 : ℝ), 0, q)`,
for `t ≤ 0`. In the language of `LMLExplore.HasCondMGFBound`, the conditional moment generating
function of `X` is bounded by that of the Bernoulli law on `[0, ∞)` (resp. `(-∞, 0]`).

Along a filtration `ℱ`, this applies to rewards `X k ∈ [0, 1]` that are `ℱ (k + 1)`-measurable
with conditional means given `ℱ k` at most (resp. at least) `m`, for instance the rewards of a
bandit algorithm whose arms have means at most (resp. at least) `m`, with no independence. The
self-normalized deviations of `LMLExplore.Probability.Martingale.RateFunction`, for the Bernoulli
law of mean `m`, whose rate function is the binary relative entropy `d(·, m)` on `[0, 1]`
(`LMLExplore.rateFunction_id_bernoulliMeasure`), then give [GC11, Theorem 10] for such rewards:
for selection indicators `w k ∈ {0, 1}` adapted to `ℱ`, `N t = ∑ k < t, w k` selected rewards
with mean `μ̂ t` after `t` rounds, `x > 0` and `n ≥ 2`,
`P(∃ t, 1 ≤ N t ≤ n, m < μ̂ t, x ≤ N t d(μ̂ t, m)) ≤ e ⌈x log n⌉ exp (-x)`, and the same below
`m`. Equivalently, the upper confidence level `U(μ̂ t, x / N t)` of the KL-UCB algorithm is below
`m` at some round with `1 ≤ N t ≤ n` with probability at most `e ⌈x log n⌉ exp (-x)` if the
conditional means are at least `m`, and the lower confidence level `L(μ̂ t, x / N t)` is above
`m` with the same probability if they are at most `m`. For `w k = 1` and independent rewards these
are the bounds of `LMLExplore.Probability.Moments.KLChernoff` on the first `s` variables.

## Main statements

* `LMLExplore.hasCondMGFBound_Ici_cgf_bernoulliMeasure`,
  `LMLExplore.hasCondMGFBound_Iic_cgf_bernoulliMeasure`: [GC11, Lemma 9], a random variable with
  values in `[0, 1]` and conditional mean at most (resp. at least) `q` has its conditional moment
  generating function bounded by that of `Ber((1 : ℝ), 0, q)` on `Set.Ici 0` (resp. `Set.Iic 0`).
* `LMLExplore.measure_exists_gt_and_le_mul_klBer_le_of_condExp_le`,
  `LMLExplore.measure_exists_lt_and_le_mul_klBer_le_of_le_condExp`: [GC11, Theorem 10] along a
  filtration, the bound `e ⌈x log n⌉ exp (-x)` on the deviations of the selected means above
  (resp. below) `m` measured by `d(·, m)`.
* `LMLExplore.measure_exists_klUpper_lt_le_of_le_condExp`,
  `LMLExplore.measure_exists_lt_klLower_le_of_condExp_le`: the time-uniform coverage of the upper
  (resp. lower) confidence levels along a filtration.

## References

* [GC11] A. Garivier and O. Cappé, *The KL-UCB algorithm for bounded stochastic bandits and
  beyond*, COLT 2011, Lemma 9 and Theorem 10.
* T. Lattimore and Cs. Szepesvári, *Bandit Algorithms*, Cambridge University Press, 2020,
  Note 10.3.1.
-/

public section

open MeasureTheory InformationTheory ProbabilityTheory Filter Real Finset
open scoped ENNReal unitInterval

namespace LMLExplore

/-! ### Conditional domination by a Bernoulli law -/

section Domination

variable {Ω : Type*} {m mΩ : MeasurableSpace Ω} {hm : m ≤ mΩ} [StandardBorelSpace Ω]
  {μ : Measure Ω} [IsFiniteMeasure μ] {X : Ω → ℝ}

/-- If `X ∈ [0, 1]` almost surely, then for almost every `ω'` (for the restriction of `μ` to `m`),
`X ∈ [0, 1]` almost surely under the conditional expectation kernel at `ω'`, and the mean of `X`
under that kernel is `E[X | m] ω'`. -/
private lemma ae_trim_ae_mem_Icc_and_condExp_eq (hm : m ≤ mΩ) (hX : Measurable X)
    (hX01 : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc 0 1) :
    ∀ᵐ ω' ∂(μ.trim hm), (∀ᵐ ω ∂(condExpKernel μ m ω'), X ω ∈ Set.Icc 0 1) ∧
      μ[X | m] ω' = ∫ ω, X ω ∂(condExpKernel μ m ω') := by
  have h01 : ∀ᵐ ω' ∂(μ.trim hm), ∀ᵐ ω ∂(condExpKernel μ m ω'), X ω ∈ Set.Icc 0 1 :=
    Measure.ae_ae_of_ae_comp (by rwa [condExpKernel_comp_trim hm])
  filter_upwards [h01, condExp_ae_eq_trim_integral_condExpKernel hm
    (.of_mem_Icc 0 1 hX.aemeasurable hX01)] with ω' h01 h_eq using ⟨h01, h_eq⟩

/-- **Conditional domination above by a Bernoulli law** [GC11, Lemma 9]. If `X` is measurable with
values in `[0, 1]` almost surely and `E[X | m] ≤ q` almost surely, then the conditional moment
generating function of `X` given `m` is bounded by that of `Ber((1 : ℝ), 0, q)` on the nonnegative
parameters: `E[exp (t X) | m] ≤ 1 - q + q exp t` for `t ≥ 0`. -/
theorem hasCondMGFBound_Ici_cgf_bernoulliMeasure (hX : Measurable X)
    (hX01 : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc 0 1) {q : I} (hmean : μ[X | m] ≤ᵐ[μ] fun _ ↦ (q : ℝ)) :
    HasCondMGFBound m hm X (cgf id Ber((1 : ℝ), 0, q)) (Set.Ici 0) μ := by
  have hmean' := stronglyMeasurable_condExp.ae_le_trim_of_stronglyMeasurable hm
    stronglyMeasurable_const hmean
  refine hasCondMGFBound_iff_kernel.2 ⟨fun t _ ↦ ?_, ?_, ?_⟩
  · rw [condExpKernel_comp_trim hm]
    exact integrable_exp_mul_of_mem_Icc hX.aemeasurable hX01
  · filter_upwards [ae_trim_ae_mem_Icc_and_condExp_eq hm hX hX01] with ω' h t _
    exact integrable_exp_mul_of_mem_Icc hX.aemeasurable h.1
  · filter_upwards [ae_trim_ae_mem_Icc_and_condExp_eq hm hX hX01, hmean'] with ω' h hq t ht
    have h_int := integrable_bernoulliMeasure (1 : ℝ) 0 q fun y ↦ exp (t * id y)
    rw [exp_cgf h_int]
    exact (mgfDominatedOn_Ici_bernoulliMeasure hX.aemeasurable h.1 (h.2 ▸ hq)).mgf_le ht h_int

/-- **Conditional domination below by a Bernoulli law** [GC11, Lemma 9]. If `X` is measurable with
values in `[0, 1]` almost surely and `q ≤ E[X | m]` almost surely, then the conditional moment
generating function of `X` given `m` is bounded by that of `Ber((1 : ℝ), 0, q)` on the nonpositive
parameters: `E[exp (t X) | m] ≤ 1 - q + q exp t` for `t ≤ 0`. -/
theorem hasCondMGFBound_Iic_cgf_bernoulliMeasure (hX : Measurable X)
    (hX01 : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc 0 1) {q : I} (hmean : (fun _ ↦ (q : ℝ)) ≤ᵐ[μ] μ[X | m]) :
    HasCondMGFBound m hm X (cgf id Ber((1 : ℝ), 0, q)) (Set.Iic 0) μ := by
  have hmean' := stronglyMeasurable_const.ae_le_trim_of_stronglyMeasurable hm
    stronglyMeasurable_condExp hmean
  refine hasCondMGFBound_iff_kernel.2 ⟨fun t _ ↦ ?_, ?_, ?_⟩
  · rw [condExpKernel_comp_trim hm]
    exact integrable_exp_mul_of_mem_Icc hX.aemeasurable hX01
  · filter_upwards [ae_trim_ae_mem_Icc_and_condExp_eq hm hX hX01] with ω' h t _
    exact integrable_exp_mul_of_mem_Icc hX.aemeasurable h.1
  · filter_upwards [ae_trim_ae_mem_Icc_and_condExp_eq hm hX hX01, hmean'] with ω' h hq t ht
    have h_int := integrable_bernoulliMeasure (1 : ℝ) 0 q fun y ↦ exp (t * id y)
    rw [exp_cgf h_int]
    exact (mgfDominatedOn_Iic_bernoulliMeasure hX.aemeasurable h.1 (h.2 ▸ hq)).mgf_le ht h_int

end Domination

section CondExpBounds

variable {Ω : Type*} {m mΩ : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]
  {X : Ω → ℝ} {c : ℝ}

/-- A bound `E[X | m] ≤ c` on the conditional mean of a variable `X ∈ [0, 1]` forces `0 ≤ c`. -/
private lemma nonneg_of_condExp_le (hX01 : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc 0 1)
    (hmean : μ[X | m] ≤ᵐ[μ] fun _ ↦ c) : 0 ≤ c := by
  obtain ⟨ω, h0, hc⟩ := ((condExp_nonneg (hX01.mono fun ω hω ↦ hω.1)).and hmean).exists
  exact h0.trans hc

/-- A bound `c ≤ E[X | m]` on the conditional mean of a variable `X ∈ [0, 1]`, for a
sub-σ-algebra `m`, forces `c ≤ 1`. -/
private lemma le_one_of_le_condExp (hm : m ≤ mΩ) (hX : Measurable X)
    (hX01 : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc 0 1) (hmean : (fun _ ↦ c) ≤ᵐ[μ] μ[X | m]) : c ≤ 1 := by
  have h_le : μ[X | m] ≤ᵐ[μ] fun _ ↦ (1 : ℝ) := by
    have := condExp_mono (m := m) (.of_mem_Icc 0 1 hX.aemeasurable hX01)
      (integrable_const (1 : ℝ)) (hX01.mono fun ω hω ↦ hω.2)
    rwa [condExp_const hm] at this
  obtain ⟨ω, hc, h1⟩ := (hmean.and h_le).exists
  exact hc.trans h1

end CondExpBounds

/-! ### Self-normalized deviations along a filtration -/

section SelfNormalized

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} [StandardBorelSpace Ω] {P : Measure Ω}
  [IsProbabilityMeasure P] {ℱ : Filtration ℕ mΩ} {X w : ℕ → Ω → ℝ} {m : ℝ}

/-- The mean `(∑ k < t, w k x k) / ∑ k < t, w k` of numbers `x k ∈ [0, 1]` with nonnegative
weights of positive sum lies in `[0, 1]`. -/
private lemma sum_mul_div_sum_mem_Icc {w x : ℕ → ℝ} (hw : ∀ k, 0 ≤ w k)
    (hx : ∀ k, x k ∈ Set.Icc 0 1) {t : ℕ} (ht : 0 < ∑ k ∈ range t, w k) :
    (∑ k ∈ range t, w k * x k) / ∑ k ∈ range t, w k ∈ Set.Icc 0 1 :=
  ⟨div_nonneg (sum_nonneg fun k _ ↦ mul_nonneg (hw k) (hx k).1) ht.le,
    (div_le_one ht).2 (sum_le_sum fun k _ ↦ mul_le_of_le_one_right (hw k) (hx k).2)⟩

/-- On a run of rewards in `[0, 1]` with selection indicators in `{0, 1}`, the mean of the
selected rewards lies in `[0, 1]` as soon as one is selected. -/
private lemma mean_mem_Icc_of_one_le (hw01 : ∀ n ω, w n ω = 0 ∨ w n ω = 1) {ω : Ω}
    (hω : ∀ k, X k ω ∈ Set.Icc 0 1) {t : ℕ} (ht : 1 ≤ ∑ k ∈ range t, w k ω) :
    (∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω ∈ Set.Icc 0 1 :=
  sum_mul_div_sum_mem_Icc (fun k ↦ (hw01 k ω).elim (·.ge) fun h ↦ h ▸ zero_le_one) hω
    (zero_lt_one.trans_le ht)

/-- **Self-normalized deviations above the mean in relative-entropy form, along a filtration**
[GC11, Theorem 10]. Let `X k ∈ [0, 1]` be `ℱ (k + 1)`-measurable with `E[X k | ℱ k] ≤ m`, and let
the selection indicators `w` be adapted with values in `{0, 1}`. For `N t = ∑ k < t, w k`
selected variables with mean `μ̂ t` after `t` rounds, `x > 0` and `n ≥ 2`, the probability that
`1 ≤ N t ≤ n`, `m < μ̂ t` and `x ≤ N t d(μ̂ t, m)` at some round `t` is at most
`e ⌈x log n⌉ exp (-x)`, where `d` is the binary relative entropy. -/
theorem measure_exists_gt_and_le_mul_klBer_le_of_condExp_le
    (hX : ∀ n, Measurable[ℱ (n + 1)] (X n)) (hX01 : ∀ n, ∀ᵐ ω ∂P, X n ω ∈ Set.Icc 0 1)
    (hmean : ∀ n, P[X n | ℱ n] ≤ᵐ[P] fun _ ↦ m) (hw : Adapted ℱ w)
    (hw01 : ∀ n ω, w n ω = 0 ∨ w n ω = 1) {x : ℝ} (hx : 0 < x) {n : ℕ} (hn : 1 < n) :
    P {ω | ∃ t, 1 ≤ ∑ k ∈ range t, w k ω ∧ ∑ k ∈ range t, w k ω ≤ n ∧
      m < (∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω ∧
      ENNReal.ofReal x ≤ ENNReal.ofReal (∑ k ∈ range t, w k ω) *
        klBer ((∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω) m} ≤
      ENNReal.ofReal (exp 1 * ⌈x * log n⌉₊ * exp (-x)) := by
  have hXm n : Measurable (X n) := (hX n).mono (ℱ.le (n + 1)) le_rfl
  have hX' : ∀ᵐ ω ∂P, ∀ k, X k ω ∈ Set.Icc 0 1 := ae_all_iff.2 hX01
  have hm0 : 0 ≤ m := nonneg_of_condExp_le (hX01 0) (hmean 0)
  rcases le_or_gt m 1 with hm1 | hm1
  swap
  · -- for `m > 1`, no selected mean is above `m`
    refine le_of_eq_of_le (measure_eq_zero_iff_ae_notMem.2 ?_) zero_le
    filter_upwards [hX'] with ω hω ⟨t, ht, _, hlt, _⟩
    linarith [(mean_mem_Icc_of_one_le hw01 hω ht).2]
  -- the variables are conditionally dominated above by the Bernoulli law of mean `m`, whose rate
  -- function is `d(·, m)` on `[0, 1]`
  set q : I := ⟨m, hm0, hm1⟩
  have h := measure_exists_gt_and_le_mul_rateFunction_le hX
    (fun k ↦ (hasCondMGFBound_Ici_cgf_bernoulliMeasure (q := q) (hXm k) (hX01 k)
      (hmean k)).mono Set.inter_subset_left fun _ _ ↦ le_rfl)
    hw hw01 (integrable_bernoulliMeasure (1 : ℝ) 0 q id) hx hn
  refine (measure_mono_ae ?_).trans h
  filter_upwards [hX'] with ω hω ⟨t, ht, htn, hlt, hle⟩
  have h_mean : ∫ y, id y ∂Ber((1 : ℝ), 0, q) = m :=
    TauCeti.Probability.integral_id_bernoulliMeasure q
  refine ⟨t, ht, htn, h_mean ▸ hlt, ?_⟩
  rwa [rateFunction_id_bernoulliMeasure q (mean_mem_Icc_of_one_le hw01 hω ht)]

/-- **Self-normalized deviations below the mean in relative-entropy form, along a filtration**
[GC11, Theorem 10]. Let `X k ∈ [0, 1]` be `ℱ (k + 1)`-measurable with `m ≤ E[X k | ℱ k]`, and let
the selection indicators `w` be adapted with values in `{0, 1}`. For `N t = ∑ k < t, w k`
selected variables with mean `μ̂ t` after `t` rounds, `x > 0` and `n ≥ 2`, the probability that
`1 ≤ N t ≤ n`, `μ̂ t < m` and `x ≤ N t d(μ̂ t, m)` at some round `t` is at most
`e ⌈x log n⌉ exp (-x)`, where `d` is the binary relative entropy. -/
theorem measure_exists_lt_and_le_mul_klBer_le_of_le_condExp
    (hX : ∀ n, Measurable[ℱ (n + 1)] (X n)) (hX01 : ∀ n, ∀ᵐ ω ∂P, X n ω ∈ Set.Icc 0 1)
    (hmean : ∀ n, (fun _ ↦ m) ≤ᵐ[P] P[X n | ℱ n]) (hw : Adapted ℱ w)
    (hw01 : ∀ n ω, w n ω = 0 ∨ w n ω = 1) {x : ℝ} (hx : 0 < x) {n : ℕ} (hn : 1 < n) :
    P {ω | ∃ t, 1 ≤ ∑ k ∈ range t, w k ω ∧ ∑ k ∈ range t, w k ω ≤ n ∧
      (∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω < m ∧
      ENNReal.ofReal x ≤ ENNReal.ofReal (∑ k ∈ range t, w k ω) *
        klBer ((∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω) m} ≤
      ENNReal.ofReal (exp 1 * ⌈x * log n⌉₊ * exp (-x)) := by
  have hXm n : Measurable (X n) := (hX n).mono (ℱ.le (n + 1)) le_rfl
  have hX' : ∀ᵐ ω ∂P, ∀ k, X k ω ∈ Set.Icc 0 1 := ae_all_iff.2 hX01
  have hm1 : m ≤ 1 := le_one_of_le_condExp (ℱ.le 0) (hXm 0) (hX01 0) (hmean 0)
  rcases le_or_gt 0 m with hm0 | hm0
  swap
  · -- for `m < 0`, no selected mean is below `m`
    refine le_of_eq_of_le (measure_eq_zero_iff_ae_notMem.2 ?_) zero_le
    filter_upwards [hX'] with ω hω ⟨t, ht, _, hlt, _⟩
    linarith [(mean_mem_Icc_of_one_le hw01 hω ht).1]
  -- the variables are conditionally dominated below by the Bernoulli law of mean `m`, whose rate
  -- function is `d(·, m)` on `[0, 1]`
  set q : I := ⟨m, hm0, hm1⟩
  have h := measure_exists_lt_and_le_mul_rateFunction_le hX
    (fun k ↦ (hasCondMGFBound_Iic_cgf_bernoulliMeasure (q := q) (hXm k) (hX01 k)
      (hmean k)).mono Set.inter_subset_left fun _ _ ↦ le_rfl)
    hw hw01 (integrable_bernoulliMeasure (1 : ℝ) 0 q id) hx hn
  refine (measure_mono_ae ?_).trans h
  filter_upwards [hX'] with ω hω ⟨t, ht, htn, hlt, hle⟩
  have h_mean : ∫ y, id y ∂Ber((1 : ℝ), 0, q) = m :=
    TauCeti.Probability.integral_id_bernoulliMeasure q
  refine ⟨t, ht, htn, h_mean ▸ hlt, ?_⟩
  rwa [rateFunction_id_bernoulliMeasure q (mean_mem_Icc_of_one_le hw01 hω ht)]

/-- `x = N * (x / N)` in `ℝ≥0∞` for a positive real `N`. -/
private lemma ofReal_eq_ofReal_mul_ofReal_div {x N : ℝ} (hN : 0 < N) :
    ENNReal.ofReal x = ENNReal.ofReal N * ENNReal.ofReal (x / N) := by
  rw [← ENNReal.ofReal_mul hN.le, mul_div_cancel₀ x hN.ne']

/-- **Time-uniform coverage of the upper confidence level, along a filtration**
[GC11, Theorem 10]. Let `X k ∈ [0, 1]` be `ℱ (k + 1)`-measurable with `m ≤ E[X k | ℱ k]`, and let
the selection indicators `w` be adapted with values in `{0, 1}`. For `N t = ∑ k < t, w k`
selected variables with mean `μ̂ t` after `t` rounds, `x > 0` and `n ≥ 2`, the probability that
`1 ≤ N t ≤ n` and the upper confidence level `U(μ̂ t, x / N t)` is less than `m` at some round
`t` is at most `e ⌈x log n⌉ exp (-x)`. -/
theorem measure_exists_klUpper_lt_le_of_le_condExp
    (hX : ∀ n, Measurable[ℱ (n + 1)] (X n)) (hX01 : ∀ n, ∀ᵐ ω ∂P, X n ω ∈ Set.Icc 0 1)
    (hmean : ∀ n, (fun _ ↦ m) ≤ᵐ[P] P[X n | ℱ n]) (hw : Adapted ℱ w)
    (hw01 : ∀ n ω, w n ω = 0 ∨ w n ω = 1) {x : ℝ} (hx : 0 < x) {n : ℕ} (hn : 1 < n) :
    P {ω | ∃ t, 1 ≤ ∑ k ∈ range t, w k ω ∧ ∑ k ∈ range t, w k ω ≤ n ∧
      klUpper ((∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω)
        (ENNReal.ofReal (x / ∑ k ∈ range t, w k ω)) < m} ≤
      ENNReal.ofReal (exp 1 * ⌈x * log n⌉₊ * exp (-x)) := by
  refine (measure_mono_ae ?_).trans
    (measure_exists_lt_and_le_mul_klBer_le_of_le_condExp hX hX01 hmean hw hw01 hx hn)
  filter_upwards [ae_all_iff.2 hX01] with ω hω ⟨t, ht, htn, hU⟩
  -- `μ̂ ≤ U(μ̂, x / N) < m`, and `m ≤ U(μ̂, x / N)` unless `x / N < d(μ̂, m)`
  have hp := mean_mem_Icc_of_one_le hw01 hω ht
  have hpU := (klUpper_mem_Icc hp.2 (ENNReal.ofReal (x / ∑ k ∈ range t, w k ω))).1
  have hm1 : m ≤ 1 :=
    le_one_of_le_condExp (ℱ.le 0) ((hX 0).mono (ℱ.le 1) le_rfl) (hX01 0) (hmean 0)
  have hd : ENNReal.ofReal (x / ∑ k ∈ range t, w k ω) <
      klBer ((∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω) m := by
    by_contra! h
    exact hU.not_ge ((le_klUpper_iff ⟨hpU.trans hU.le, hm1⟩).2 h)
  refine ⟨t, ht, htn, hpU.trans_lt hU, ?_⟩
  rw [ofReal_eq_ofReal_mul_ofReal_div (zero_lt_one.trans_le ht)]
  gcongr

/-- **Time-uniform coverage of the lower confidence level, along a filtration**
[GC11, Theorem 10]. Let `X k ∈ [0, 1]` be `ℱ (k + 1)`-measurable with `E[X k | ℱ k] ≤ m`, and let
the selection indicators `w` be adapted with values in `{0, 1}`. For `N t = ∑ k < t, w k`
selected variables with mean `μ̂ t` after `t` rounds, `x > 0` and `n ≥ 2`, the probability that
`1 ≤ N t ≤ n` and the lower confidence level `L(μ̂ t, x / N t)` is more than `m` at some round
`t` is at most `e ⌈x log n⌉ exp (-x)`. -/
theorem measure_exists_lt_klLower_le_of_condExp_le
    (hX : ∀ n, Measurable[ℱ (n + 1)] (X n)) (hX01 : ∀ n, ∀ᵐ ω ∂P, X n ω ∈ Set.Icc 0 1)
    (hmean : ∀ n, P[X n | ℱ n] ≤ᵐ[P] fun _ ↦ m) (hw : Adapted ℱ w)
    (hw01 : ∀ n ω, w n ω = 0 ∨ w n ω = 1) {x : ℝ} (hx : 0 < x) {n : ℕ} (hn : 1 < n) :
    P {ω | ∃ t, 1 ≤ ∑ k ∈ range t, w k ω ∧ ∑ k ∈ range t, w k ω ≤ n ∧
      m < klLower ((∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω)
        (ENNReal.ofReal (x / ∑ k ∈ range t, w k ω))} ≤
      ENNReal.ofReal (exp 1 * ⌈x * log n⌉₊ * exp (-x)) := by
  refine (measure_mono_ae ?_).trans
    (measure_exists_gt_and_le_mul_klBer_le_of_condExp_le hX hX01 hmean hw hw01 hx hn)
  filter_upwards [ae_all_iff.2 hX01] with ω hω ⟨t, ht, htn, hL⟩
  -- `m < L(μ̂, x / N) ≤ μ̂`, and `L(μ̂, x / N) ≤ m` unless `x / N < d(μ̂, m)`
  have hp := mean_mem_Icc_of_one_le hw01 hω ht
  have hLp := klLower_mem_Icc hp.1 (ENNReal.ofReal (x / ∑ k ∈ range t, w k ω))
  have hm0 : 0 ≤ m := nonneg_of_condExp_le (hX01 0) (hmean 0)
  have hd : ENNReal.ofReal (x / ∑ k ∈ range t, w k ω) <
      klBer ((∑ k ∈ range t, w k ω * X k ω) / ∑ k ∈ range t, w k ω) m := by
    by_contra! h
    exact hL.not_ge ((klLower_le_iff ⟨hm0, hL.le.trans hLp.2⟩).2 h)
  refine ⟨t, ht, htn, hL.trans_le hLp.2, ?_⟩
  rw [ofReal_eq_ofReal_mul_ofReal_div (zero_lt_one.trans_le ht)]
  gcongr

end SelfNormalized

end LMLExplore
