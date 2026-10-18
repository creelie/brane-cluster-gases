/-
  Finite and elementary statements behind "Ghosts in Poisson brane gases".

  Masses squared of a frozen realization are t : Fin n → ℝ, strictly increasing
  and positive.  The residue of 1/(z ∏ (1 + z/t j)) at z = -t k is
  r k = -1 / ∏_{j ≠ k} (1 - t k / t j).
-/
import Mathlib

open Finset Polynomial

namespace BraneGhosts

/-- The product entering the massive residues. -/
noncomputable def resProd {n : ℕ} (t : Fin n → ℝ) (k : Fin n) : ℝ :=
  ∏ j ∈ univ.erase k, (1 - t k / t j)

/-- The residue at the pole z = -t k. -/
noncomputable def residue {n : ℕ} (t : Fin n → ℝ) (k : Fin n) : ℝ :=
  -1 / resProd t k

/-- Each factor of the product is nonzero and its sign is fixed by the order. -/
lemma factor_neg {n : ℕ} {t : Fin n → ℝ} (ht : StrictMono t) (hpos : ∀ j, 0 < t j)
    {j k : Fin n} (hjk : j < k) : 1 - t k / t j < 0 := by
  have h1 : t j < t k := ht hjk
  have h2 : 1 < t k / t j := by rw [lt_div_iff₀ (hpos j)]; linarith
  linarith

lemma factor_pos {n : ℕ} {t : Fin n → ℝ} (ht : StrictMono t) (hpos : ∀ j, 0 < t j)
    {j k : Fin n} (hjk : k < j) : 0 < 1 - t k / t j := by
  have h1 : t k < t j := ht hjk
  have h2 : t k / t j < 1 := by rw [div_lt_one (hpos j)]; exact h1
  linarith

/-- Sign of the residue product: `(-1)^k * resProd t k > 0`, so with masses
listed in increasing order the residue at the `k`-th massive pole (counted from
`0`) has sign `(-1)^(k+1)`; the lightest massive pole is a ghost. -/
theorem resProd_sign {n : ℕ} (t : Fin n → ℝ) (ht : StrictMono t) (hpos : ∀ j, 0 < t j)
    (k : Fin n) : 0 < (-1 : ℝ) ^ (k : ℕ) * resProd t k := by
  classical
  unfold resProd
  -- split the factors with j < k from those with j > k
  have hsplit : (univ.erase k) = (univ.filter (fun j => j < k)) ∪
      (univ.filter (fun j => k < j)) := by
    ext j
    simp only [mem_erase, mem_univ, and_true, mem_union, mem_filter, true_and, ne_eq]
    constructor
    · intro h; rcases lt_or_gt_of_ne h with h | h
      · exact Or.inl h
      · exact Or.inr h
    · rintro (h | h)
      · exact ne_of_lt h
      · exact ne_of_gt h
  have hdisj : Disjoint (univ.filter (fun j : Fin n => j < k))
      (univ.filter (fun j => k < j)) := by
    rw [disjoint_filter]; intro j _ h1 h2; exact lt_asymm h1 h2
  rw [hsplit, prod_union hdisj]
  have hlow : ∏ j ∈ univ.filter (fun j => j < k), (1 - t k / t j) =
      (-1 : ℝ) ^ (univ.filter (fun j : Fin n => j < k)).card *
        ∏ j ∈ univ.filter (fun j => j < k), (t k / t j - 1) := by
    rw [← prod_const, ← prod_mul_distrib]
    apply prod_congr rfl; intro j _; ring
  have hcard : (univ.filter (fun j : Fin n => j < k)).card = (k : ℕ) := by
    rw [Finset.filter_gt_eq_Iio, Fin.card_Iio]
  rw [hlow, hcard]
  have hA : 0 < ∏ j ∈ univ.filter (fun j => j < k), (t k / t j - 1) := by
    apply prod_pos; intro j hj
    have := factor_neg ht hpos (mem_filter.mp hj).2; linarith
  have hB : 0 < ∏ j ∈ univ.filter (fun j => k < j), (1 - t k / t j) := by
    apply prod_pos; intro j hj; exact factor_pos ht hpos (mem_filter.mp hj).2
  have hsq : ((-1 : ℝ) ^ (k : ℕ)) * ((-1 : ℝ) ^ (k : ℕ)) = 1 := by
    rw [← mul_pow]; norm_num
  calc 0 < (∏ j ∈ univ.filter (fun j => j < k), (t k / t j - 1)) *
        ∏ j ∈ univ.filter (fun j => k < j), (1 - t k / t j) := mul_pos hA hB
    _ = (-1 : ℝ) ^ (k : ℕ) * ((-1 : ℝ) ^ (k : ℕ) *
          (∏ j ∈ univ.filter (fun j => j < k), (t k / t j - 1)) *
          ∏ j ∈ univ.filter (fun j => k < j), (1 - t k / t j)) := by
        rw [← mul_assoc, ← mul_assoc, hsq, one_mul]

/-- The lightest massive pole has a negative residue. -/
theorem lightest_is_ghost {n : ℕ} (t : Fin (n + 1) → ℝ) (ht : StrictMono t)
    (hpos : ∀ j, 0 < t j) : residue t 0 < 0 := by
  have h := resProd_sign t ht hpos 0
  simp only [Fin.val_zero, pow_zero, one_mul] at h
  unfold residue
  exact div_neg_of_neg_of_pos (by norm_num) h

/-- Evaluating a Lagrange basis polynomial at `0`. -/
lemma eval_basis_zero {n : ℕ} (t : Fin n → ℝ) (k : Fin n) :
    (Lagrange.basis univ t k).eval 0 = ∏ j ∈ univ.erase k, (t j / (t j - t k)) := by
  classical
  unfold Lagrange.basis
  rw [eval_prod]
  apply prod_congr rfl
  intro j _
  unfold Lagrange.basisDivisor
  simp only [eval_mul, eval_C, eval_sub, eval_X, zero_sub]
  rw [div_eq_mul_inv, mul_comm]
  have : t j - t k = -(t k - t j) := by ring
  rw [this, inv_neg]; ring

/-- Finite sum rule: for distinct positive masses the massive residues of
`1/(z ∏ (1 + z / t j))` add up to `-1`. -/
theorem sum_rule {n : ℕ} (t : Fin (n + 1) → ℝ) (hinj : Function.Injective t)
    (hpos : ∀ j, 0 < t j) : ∑ k, residue t k = -1 := by
  classical
  have hL := Lagrange.sum_basis (s := (univ : Finset (Fin (n + 1)))) (v := t)
    (hinj.injOn) univ_nonempty
  have h0 := congrArg (fun p : ℝ[X] => p.eval 0) hL
  simp only [eval_finsetSum, eval_one] at h0
  have key : ∀ k : Fin (n + 1), residue t k =
      -((Lagrange.basis univ t k).eval 0) := by
    intro k
    rw [eval_basis_zero, residue, resProd, neg_div, one_div, ← prod_inv_distrib]
    congr 1
    apply prod_congr rfl
    intro j hj
    have hne : j ≠ k := ne_of_mem_erase hj
    have htj : t j ≠ 0 := (hpos j).ne'
    have hd : t j - t k ≠ 0 := sub_ne_zero.mpr (fun h => hne (hinj h))
    rw [one_sub_div htj, inv_div]
  simp_rw [key, sum_neg_distrib, h0]

/-- Truncating a frozen gas to its lightest masses shrinks every residue in
modulus: each discarded factor lies in `(0,1)`. -/
theorem truncation_factor {s t : ℝ} (hs : 0 < s) (hst : s < t) :
    0 < 1 - s / t ∧ 1 - s / t < 1 := by
  have ht : 0 < t := lt_trans hs hst
  constructor
  · have : s / t < 1 := by rw [div_lt_one ht]; exact hst
    linarith
  · have : 0 < s / t := div_pos hs ht
    linarith

/-- `1 - e^{-d} ≥ d/(1+d)` for `d ≥ 0`, used to bound the near-neighbour
factors of the residues. -/
theorem one_sub_exp_neg_ge {d : ℝ} (hd : 0 ≤ d) : d / (1 + d) ≤ 1 - Real.exp (-d) := by
  have h1 : 0 < 1 + d := by linarith
  have h2 : d + 1 ≤ Real.exp d := Real.add_one_le_exp d
  have h3 : Real.exp (-d) ≤ 1 / (1 + d) := by
    rw [Real.exp_neg, inv_eq_one_div]
    apply one_div_le_one_div_of_le h1; linarith
  have h4 : d / (1 + d) = 1 - 1 / (1 + d) := by field_simp; ring
  linarith

/-- The half-angle bound `|t + z|² ≥ cos²(φ/2) (t + |z|)²` in the form used in
the text: with `c = cos φ`, `t² + r² + 2 t r c ≥ (t + r)² (1 + c) / 2`. -/
theorem half_angle_bound (t r c : ℝ) (hc : c ≤ 1) :
    (t + r) ^ 2 * (1 + c) / 2 ≤ t ^ 2 + r ^ 2 + 2 * t * r * c := by
  nlinarith [sq_nonneg (t - r), mul_nonneg (sub_nonneg.mpr hc) (sq_nonneg (t - r))]

/-- If `0 ≤ Im w ≤ π` then `Im (1 - e^{-w}) ≥ 0`: the map `w ↦ 1 - e^{-w}`
sends the strip of height `π` into the closed upper half plane. -/
theorem im_one_sub_exp_neg_nonneg (w : ℂ) (h0 : 0 ≤ w.im) (h1 : w.im ≤ Real.pi) :
    0 ≤ (1 - Complex.exp (-w)).im := by
  rw [Complex.sub_im, Complex.one_im, Complex.exp_im, Complex.neg_re, Complex.neg_im,
    Real.sin_neg]
  have : 0 ≤ Real.sin w.im := Real.sin_nonneg_of_nonneg_of_le_pi h0 h1
  have : 0 < Real.exp (-w.re) := Real.exp_pos _
  nlinarith

/-- Sign of the averaged spectral density `-e^{-Re φ} sin(πλ)/(π s)`: negative
(ghost) for `0 < λ < 1`. -/
theorem density_sign_ghost {lam : ℝ} (h0 : 0 < lam) (h1 : lam < 1) :
    0 < Real.sin (Real.pi * lam) := by
  apply Real.sin_pos_of_pos_of_lt_pi
  · exact mul_pos Real.pi_pos h0
  · nlinarith [Real.pi_pos]

/-- ... and positive (normal) for `1 < λ < 2`. -/
theorem density_sign_normal {lam : ℝ} (h1 : 1 < lam) (h2 : lam < 2) :
    Real.sin (Real.pi * lam) < 0 := by
  have : Real.sin (Real.pi * lam) = -Real.sin (Real.pi * (lam - 1)) := by
    rw [show Real.pi * lam = Real.pi * (lam - 1) + Real.pi by ring, Real.sin_add_pi]
  rw [this, neg_lt_zero]
  apply Real.sin_pos_of_pos_of_lt_pi
  · exact mul_pos Real.pi_pos (by linarith)
  · nlinarith [Real.pi_pos]

/-- The scale-free gas with one brane per e-fold (`θ = 1`): the averaged
propagator is the graviton minus a single ghost of unit weight at threshold. -/
theorem theta_one_partial_fraction (z t₀ : ℂ) (hz : z ≠ 0) (ht : t₀ ≠ 0)
    (hzt : z + t₀ ≠ 0) :
    1 / (z * (1 + z / t₀)) = 1 / z - 1 / (z + t₀) := by
  have h1 : 1 + z / t₀ = (z + t₀) / t₀ := by field_simp; ring
  rw [h1]
  field_simp
  ring

section PoissonKernel
open MeasureTheory Real Set
/-- The Poisson kernel of the upper half plane has total mass `π`. -/
theorem poisson_kernel_total (x y : ℝ) (hy : 0 < y) :
    ∫ t : ℝ, y / ((t + x) ^ 2 + y ^ 2) = π := by
  have h1 : ∀ t : ℝ, y / ((t + x) ^ 2 + y ^ 2) = y⁻¹ * (1 + ((t + x) / y) ^ 2)⁻¹ := by
    intro t
    have : y ≠ 0 := hy.ne'
    field_simp
    ring
  simp_rw [h1]
  rw [integral_const_mul]
  rw [integral_add_right_eq_self (fun t => (1 + (t / y) ^ 2)⁻¹) x]
  rw [Measure.integral_comp_div (fun t => (1 + t ^ 2)⁻¹) y, integral_univ_inv_one_add_sq]
  rw [abs_of_pos hy, smul_eq_mul]
  field_simp

theorem poisson_kernel_integrable (x y : ℝ) (hy : 0 < y) :
    Integrable (fun t : ℝ => y / ((t + x) ^ 2 + y ^ 2)) := by
  have h1 : (fun t : ℝ => y / ((t + x) ^ 2 + y ^ 2)) =
      fun t => y⁻¹ * (1 + ((t + x) / y) ^ 2)⁻¹ := by
    funext t
    have : y ≠ 0 := hy.ne'
    field_simp
    ring
  rw [h1]
  apply Integrable.const_mul
  have := (integrable_inv_one_add_sq.comp_div hy.ne').comp_add_right x
  simpa using this

/-- Im φ ≤ π when the log-intensity λ is at most one. -/
theorem im_phi_le_pi (lam : ℝ → ℝ) (hm : Measurable lam) (h0 : ∀ t, 0 ≤ lam t)
    (h1 : ∀ t, lam t ≤ 1) (x y : ℝ) (hy : 0 < y) :
    ∫ t in Ioi (0 : ℝ), lam t * (y / ((t + x) ^ 2 + y ^ 2)) ≤ π := by
  have hP := poisson_kernel_integrable x y hy
  have hpos : ∀ t, 0 ≤ y / ((t + x) ^ 2 + y ^ 2) := fun t => by positivity
  have hI : Integrable (fun t => lam t * (y / ((t + x) ^ 2 + y ^ 2))) := by
    refine hP.mono' (hm.aestronglyMeasurable.mul hP.aestronglyMeasurable) ?_
    refine Filter.Eventually.of_forall (fun t => ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (h0 t) (hpos t))]
    exact mul_le_of_le_one_left (hpos t) (h1 t)
  calc ∫ t in Ioi (0 : ℝ), lam t * (y / ((t + x) ^ 2 + y ^ 2))
      ≤ ∫ t, lam t * (y / ((t + x) ^ 2 + y ^ 2)) :=
        setIntegral_le_integral hI (Filter.Eventually.of_forall
          (fun t => mul_nonneg (h0 t) (hpos t)))
    _ ≤ ∫ t, y / ((t + x) ^ 2 + y ^ 2) := by
        apply integral_mono hI hP
        intro t; exact mul_le_of_le_one_left (hpos t) (h1 t)
    _ = π := poisson_kernel_total x y hy

end PoissonKernel

end BraneGhosts
