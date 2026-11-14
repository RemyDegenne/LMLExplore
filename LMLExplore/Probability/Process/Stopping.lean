/-
Copyright (c) 2026 LMLExplore contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: LMLExplore contributors
-/
module

public import Mathlib.MeasureTheory.MeasurableSpace.Instances
public import Mathlib.Probability.Process.Stopping

/-!
# Stopping times valued in `ℕ∞` and stopped adapted processes

A stopping time of a filtration indexed by `ℕ` takes values in `WithTop ℕ`, which Mathlib equips
with its Borel σ-algebra, and `IsStoppingTime.measurable'` is measurability for that σ-algebra.
The same type seen as `ℕ∞` carries the discrete σ-algebra `⊤`; the two instances are not
syntactically equal. This file states measurability of such a stopping time for the discrete
σ-algebra of `ℕ∞`.

A process that is adapted, in the sense of measurability, to a filtration indexed by a discrete
order stays adapted when it is stopped at a stopping time; Mathlib states this for strongly adapted
processes.

## Main statements

* `MeasureTheory.IsStoppingTime.measurable_enat`: a stopping time `τ : Ω → ℕ∞` of a filtration
  indexed by `ℕ` is measurable for the discrete σ-algebra of `ℕ∞`.
* `MeasureTheory.Adapted.stoppedProcess_of_discrete`: a stopped adapted process is adapted.
-/

public section

open TopologicalSpace

namespace MeasureTheory.IsStoppingTime

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {f : Filtration ℕ mΩ} {τ : Ω → ℕ∞}

/-- A stopping time `τ : Ω → ℕ∞` of a filtration indexed by `ℕ` is measurable for the discrete
σ-algebra of `ℕ∞`.

`IsStoppingTime.measurable'` gives measurability for the Borel σ-algebra of `WithTop ℕ`, a
different instance on the same type. -/
lemma measurable_enat (hτ : IsStoppingTime f τ) : Measurable τ :=
  -- the preimage of each (Borel-measurable) singleton is measurable
  measurable_to_countable' fun x ↦ hτ.measurable' (measurableSet_singleton (α := WithTop ℕ) x)

end MeasureTheory.IsStoppingTime

namespace MeasureTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω}

/-- If the indexing order has the discrete topology, then the stopped process of an adapted
process with values in a second-countable pseudo-metrizable Borel space is adapted. This is the
measurable form of Mathlib's `StronglyAdapted.stoppedProcess_of_discrete`. -/
theorem Adapted.stoppedProcess_of_discrete {ι γ : Type*} [Nonempty ι] [LinearOrder ι]
    [TopologicalSpace ι] [SecondCountableTopology ι] [OrderTopology ι] [MeasurableSpace ι]
    [BorelSpace ι] [DiscreteTopology ι] [TopologicalSpace γ] [PseudoMetrizableSpace γ]
    [SecondCountableTopology γ] [MeasurableSpace γ] [BorelSpace γ] {f : Filtration ι mΩ}
    {u : ι → Ω → γ} (hu : Adapted f u) {τ : Ω → WithTop ι} (hτ : IsStoppingTime f τ) :
    Adapted f (stoppedProcess u τ) :=
  (hu.stronglyAdapted.stoppedProcess_of_discrete hτ).adapted

end MeasureTheory
