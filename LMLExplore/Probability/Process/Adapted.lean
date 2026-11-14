/-
Copyright (c) 2026 LMLExplore contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: LMLExplore contributors
-/
module

public import Mathlib.MeasureTheory.Measure.Prod
public import Mathlib.Probability.Process.Adapted

/-!
# Adapted products and mixtures

A product `M₀ * ∏ k < n, F k` is adapted to a filtration `ℱ` indexed by `ℕ` when `M₀` is
`ℱ 0`-measurable and each factor `F k` is `ℱ (k + 1)`-measurable. Such products are the wealth
processes of a gambler who reinvests at every round. A mixture `∫⁻ θ, M θ i ω ∂π` of a family of
`ℝ≥0∞`-valued processes over an s-finite measure is adapted when the family is jointly measurable
in `(θ, ω)` for `ℱ i` at every time `i`.

## Main statements

* `MeasureTheory.adapted_mul_prod_range`: the process `n ↦ M₀ * ∏ k < n, F k` is adapted.
* `MeasureTheory.adapted_lintegral`: the mixture `i ↦ ∫⁻ θ, M θ i · ∂π` is adapted.
-/

public section

open Finset
open scoped ENNReal

namespace MeasureTheory

variable {Ω β : Type*} {mΩ : MeasurableSpace Ω} [MeasurableSpace β] [CommMonoid β]
  [MeasurableMul₂ β] {ℱ : Filtration ℕ mΩ}

/-- The process `n ↦ M₀ * ∏ k < n, F k` is adapted when `M₀` is `ℱ 0`-measurable and each factor
`F k` is `ℱ (k + 1)`-measurable. -/
theorem adapted_mul_prod_range {M₀ : Ω → β} (hM₀ : Measurable[ℱ 0] M₀) {F : ℕ → Ω → β}
    (hF : ∀ k, Measurable[ℱ (k + 1)] (F k)) :
    Adapted ℱ (fun n ω ↦ M₀ ω * ∏ k ∈ range n, F k ω) := fun n ↦
  (hM₀.mono (ℱ.mono n.zero_le) le_rfl).mul <| Finset.measurable_fun_prod _ fun k hk ↦
    (hF k).mono (ℱ.mono (mem_range.1 hk)) le_rfl

/-- The mixture `i ↦ ∫⁻ θ, M θ i · ∂π` over an s-finite measure `π` of a family of
`ℝ≥0∞`-valued processes is adapted when, at every time `i`, the family is jointly measurable in
`(θ, ω)` for the product of the σ-algebra of `Θ` and `f i`. -/
theorem adapted_lintegral {ι Θ : Type*} [Preorder ι] {mΘ : MeasurableSpace Θ} {π : Measure Θ}
    [SFinite π] {f : Filtration ι mΩ} {M : Θ → ι → Ω → ℝ≥0∞}
    (hM : ∀ i, Measurable[mΘ.prod (f i)] fun p : Θ × Ω ↦ M p.1 i p.2) :
    Adapted f (fun i ω ↦ ∫⁻ θ, M θ i ω ∂π) := fun i ↦
  -- the measurable space on `Ω` is `f i`, not the ambient `mΩ`, hence the explicit instances
  @Measurable.lintegral_prod_left' Θ Ω mΘ (f i) π _ _ (hM i)

end MeasureTheory
