/-
Copyright (c) 2026 LMLExplore contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: LMLExplore contributors
-/
module

public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Ring.Defs

/-!
# Sums of products of nonnegative and nonpositive terms

A sum `∑ i ∈ s, c i * v i` with `0 ≤ c i` and `v i ≤ 0` for `i ∈ s` is nonpositive, so it is
nonnegative only when every product `c i * v i` vanishes. With `c` a vector of Lagrange multipliers
and `v` the values of the constraints at a feasible point, the second statement is complementary
slackness.

## Main statements

* `Finset.sum_mul_nonpos`: `∑ i ∈ s, c i * v i ≤ 0`.
* `Finset.mul_eq_zero_of_sum_mul_nonneg`: if moreover `0 ≤ ∑ i ∈ s, c i * v i`, then
  `c i * v i = 0` for every `i ∈ s`.
-/

public section

namespace Finset

variable {ι R : Type*} [NonUnitalNonAssocSemiring R] [PartialOrder R] [IsOrderedAddMonoid R]
  [PosMulMono R] {s : Finset ι} {c v : ι → R}

/-- A sum of products of nonnegative and nonpositive terms is nonpositive. -/
theorem sum_mul_nonpos (hc : ∀ i ∈ s, 0 ≤ c i) (hv : ∀ i ∈ s, v i ≤ 0) :
    ∑ i ∈ s, c i * v i ≤ 0 :=
  sum_nonpos fun i hi ↦ mul_nonpos_of_nonneg_of_nonpos (hc i hi) (hv i hi)

/-- Products of nonnegative and nonpositive terms with a nonnegative sum all vanish. -/
theorem mul_eq_zero_of_sum_mul_nonneg (hc : ∀ i ∈ s, 0 ≤ c i) (hv : ∀ i ∈ s, v i ≤ 0)
    (hs : 0 ≤ ∑ i ∈ s, c i * v i) {i : ι} (hi : i ∈ s) : c i * v i = 0 :=
  (sum_eq_zero_iff_of_nonpos fun i hi ↦ mul_nonpos_of_nonneg_of_nonpos (hc i hi) (hv i hi)).1
    ((sum_mul_nonpos hc hv).antisymm hs) i hi

end Finset
