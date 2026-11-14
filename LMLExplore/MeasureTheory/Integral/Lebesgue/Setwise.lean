/-
Copyright (c) 2026 LMLExplore contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: LMLExplore contributors
-/
module

public import Mathlib.MeasureTheory.Integral.Lebesgue.Basic
public import Mathlib.Order.Filter.ENNReal
public import TauCeti.Topology.Instances.ENNReal

/-!
# Fatou's lemma for setwise lower bounds on measures

Let `μs i` be a family of measures and `μ` a measure such that `μ s ≤ liminf (fun i ↦ μs i s) l`
for every measurable set `s`; this holds in particular when `μs i` converges setwise to `μ` along
`l`, that is, when `μs i s` tends to `μ s` for every measurable set `s`. Then the lower Lebesgue
integral is lower semicontinuous: `∫⁻ f ∂μ ≤ liminf (fun i ↦ ∫⁻ f ∂μs i) l` for every function
`f : α → ℝ≥0∞`, with no measurability, finiteness or integrability assumption. This is Fatou's
lemma with the measures, rather than the functions, varying, and the measurable-set analogue of
the Portmanteau bound
`MeasureTheory.lintegral_le_liminf_lintegral_of_forall_isOpen_measure_le_liminf_measure`. It
shows, for instance, that a set of measures defined by upper bounds on the integrals of a family
of functions is closed under setwise convergence.

## Main statements

* `MeasureTheory.lintegral_le_liminf_lintegral_of_forall_measure_le_liminf_measure`: the lower
  Lebesgue integral is lower semicontinuous.
* `MeasureTheory.lintegral_le_of_frequently_of_forall_measure_le_liminf_measure`: if
  `∫⁻ f ∂μs i ≤ c` frequently along `l`, then `∫⁻ f ∂μ ≤ c`.
-/

public section

open Filter
open scoped ENNReal NNReal

namespace MeasureTheory

variable {α ι : Type*} {mα : MeasurableSpace α} {l : Filter ι} {μs : ι → Measure α}
  {μ : Measure α}

/-- **Fatou's lemma for varying measures**: if `μ s ≤ liminf (fun i ↦ μs i s) l` for every
measurable set `s`, for instance if `μs i` converges setwise to `μ` along `l`, then the lower
Lebesgue integral of any function under `μ` is at most the inferior limit of its integrals under
`μs i`. -/
theorem lintegral_le_liminf_lintegral_of_forall_measure_le_liminf_measure
    (h : ∀ s, MeasurableSet s → μ s ≤ liminf (fun i ↦ μs i s) l) (f : α → ℝ≥0∞) :
    ∫⁻ x, f x ∂μ ≤ liminf (fun i ↦ ∫⁻ x, f x ∂μs i) l := by
  rcases l.eq_or_neBot with rfl | hl
  · simp
  rw [lintegral_eq_nnreal]
  refine iSup₂_le fun φ hφ ↦ ?_
  -- The integral of a simple function with finite values is a finite combination of the
  -- measures of measurable sets, and the inferior limit is superadditive.
  have hφ_le : (φ.map ((↑) : ℝ≥0 → ℝ≥0∞)).lintegral μ ≤
      liminf (fun i ↦ (φ.map ((↑) : ℝ≥0 → ℝ≥0∞)).lintegral (μs i)) l := by
    simp only [SimpleFunc.map_lintegral]
    induction φ.range using Finset.induction_on with
    | empty => simp
    | insert c t hc ih =>
      simp only [Finset.sum_insert hc]
      refine (add_le_add ?_ ih).trans (ENNReal.le_liminf_add _ _)
      rw [ENNReal.liminf_const_mul_of_ne_top ENNReal.coe_ne_top]
      gcongr
      exact h _ (φ.measurableSet_preimage _)
  refine hφ_le.trans (liminf_le_liminf (.of_forall fun i ↦ ?_))
  rw [← SimpleFunc.lintegral_eq_lintegral]
  exact lintegral_mono fun x ↦ by simpa using hφ x

/-- If `μ s ≤ liminf (fun i ↦ μs i s) l` for every measurable set `s`, for instance if `μs i`
converges setwise to `μ` along `l`, and `∫⁻ f ∂μs i ≤ c` frequently along `l`, then
`∫⁻ f ∂μ ≤ c`. -/
theorem lintegral_le_of_frequently_of_forall_measure_le_liminf_measure
    (h : ∀ s, MeasurableSet s → μ s ≤ liminf (fun i ↦ μs i s) l) {f : α → ℝ≥0∞}
    {c : ℝ≥0∞} (hc : ∃ᶠ i in l, ∫⁻ x, f x ∂μs i ≤ c) :
    ∫⁻ x, f x ∂μ ≤ c :=
  (lintegral_le_liminf_lintegral_of_forall_measure_le_liminf_measure h f).trans
    (liminf_le_of_frequently_le' hc)

end MeasureTheory
