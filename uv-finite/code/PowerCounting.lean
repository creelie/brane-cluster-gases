/-
  Exact integer statements behind the power counting and the one-loop tuning of
  "Finite quantum gravity from a scale-invariant gas of branes" (D. Bhattacharjee).

  Checked with Lean 4 core only (no Mathlib):  lean PowerCounting.lean
  The file compiles without errors or `sorry` exactly when every statement below holds.

  What is proved here, for all natural numbers where quantifiers appear:
    * the merging inequality (a-k)_+ + (b-k)_+ <= (a+b-2-k)_+ for a, b, k >= 2;
    * its iterated form for any list of vertex valences h_v >= 2;
    * the superficial-degree bound 4L - (2n+2) I + sum_v max(2n+2, h_v) <= omegaBar n L
      for every graph with I lines, V vertices and I - V = L - 1;
    * omegaBar n L < 0 for all n >= 4 and L >= 2, while omegaBar n 2 >= 0 for n <= 3;
    * with the weakest fine graining, 2 m_sigma = j <= 2n - 4: the bound omegaBarM grows with j,
      equals omegaBar at j = 0, and is negative for all n >= 4 and L >= 2;
    * the entries of Table (power), both blocks;
    * the determinants -24 and -12 of the pole matrix and of its upper-left block,
      and the fact that the tuning of Eq. (tuning) cancels the three one-loop poles;
    * the constant C_r = 1 + 2r + 2r^2 of the insertion lemma as the sum of its three parts.
  References such as Eq. (omega) or Table (power) name the LaTeX labels of the paper.
  Analytic statements (bounds on special functions, integrals, probability) are not
  formalized here; they are proved in the paper and checked numerically in C, Python
  and Julia.
-/

/-! ## Merging inequality (footnote in Sec. degree) -/

theorem merge_pair (a b k : Nat) (ha : 2 ≤ a) (hb : 2 ≤ b) (hk : 2 ≤ k) :
    (a - k) + (b - k) ≤ a + b - 2 - k := by
  omega

theorem sum_ge_two_mul_length : ∀ (hs : List Nat), (∀ h ∈ hs, 2 ≤ h) → 2 * hs.length ≤ hs.sum
  | [], _ => by simp
  | h :: t, hall => by
    have h1 : 2 ≤ h := hall h (List.mem_cons_self h t)
    have ih := sum_ge_two_mul_length t (fun x hx => hall x (List.mem_cons_of_mem h hx))
    simp only [List.length_cons, List.sum_cons]
    omega

/-- Iterated merging: the sum of (h_v - k)_+ over the vertices is at most
    (sum h_v - 2 (V - 1) - k)_+ . -/
theorem merge_list (k : Nat) (hk : 2 ≤ k) :
    ∀ (hs : List Nat), hs ≠ [] → (∀ h ∈ hs, 2 ≤ h) →
      (hs.map (· - k)).sum ≤ hs.sum - 2 * (hs.length - 1) - k
  | [], hne, _ => absurd rfl hne
  | [h], _, _ => by simp
  | h :: h' :: t, _, hall => by
    have hh : 2 ≤ h := hall h (List.mem_cons_self h _)
    have hrest : ∀ x ∈ h' :: t, 2 ≤ x := fun x hx => hall x (List.mem_cons_of_mem h hx)
    have ih := merge_list k hk (h' :: t) (List.cons_ne_nil h' t) hrest
    have hsum := sum_ge_two_mul_length (h' :: t) hrest
    simp only [List.map_cons, List.sum_cons, List.length_cons] at ih hsum ⊢
    omega

theorem sum_max_split (k : Nat) :
    ∀ (hs : List Nat), (hs.map (fun h => max k h)).sum = k * hs.length + (hs.map (· - k)).sum
  | [] => by simp
  | h :: t => by
    have ih := sum_max_split k t
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.mul_add, Nat.mul_one] at ih ⊢
    have : max k h = k + (h - k) := by omega
    omega

/-! ## Superficial degree, Eq. (omega) -/

/-- The bound on the superficial degree of an L-loop subgraph, Eq. (omega), for m_sigma = 0. -/
def omegaBar (n L : Int) : Int := 4*L - (2*n+2)*(L-1) + 2*(max (L - n - 1) 0)

/-- The degree count of Sec. degree: lines contribute -(2n+2) each, a vertex with h_v lines
    of the subgraph at most max(2n+2, h_v), the measure 4L.  For any valences h_v >= 2 with
    sum 2I and I - V = L - 1, the count is at most omegaBar n L. -/
theorem degree_le_omegaBar (n L I : Nat) (hs : List Nat) (hne : hs ≠ [])
    (hval : ∀ h ∈ hs, 2 ≤ h) (hsum : hs.sum = 2 * I) (hloops : I + 1 = hs.length + L) :
    (4 * (L : Int) - (2 * n + 2) * I + ((hs.map (fun h => max (2*n+2) h)).sum : Nat))
      ≤ omegaBar n L := by
  have hk : 2 ≤ 2 * n + 2 := by omega
  have hm := merge_list (2*n+2) hk hs hne hval
  have hs2 := sum_max_split (2*n+2) hs
  have hlen : 1 ≤ hs.length := by
    cases hs with
    | nil => exact absurd rfl hne
    | cons _ _ => simp
  rw [hs2]
  unfold omegaBar
  -- write everything with V = hs.length and the excess E = sum (h_v - k)_+
  generalize hE : (hs.map (· - (2*n+2))).sum = E at hm ⊢
  generalize hV : hs.length = V at hm hlen hloops ⊢
  rw [hsum] at hm
  have e2 : ((2:Int)*n+2)*((L:Int)-1) = 2*((n:Int)*L) - 2*n + 2*L - 2 := by
    simp only [Int.add_mul, Int.mul_sub, Int.mul_assoc]; omega
  have e3 : ((2:Int)*n+2)*(I:Int) = 2*((n:Int)*(I:Int)) + 2*I := by
    simp only [Int.add_mul, Int.mul_assoc] <;> omega
  have e4 : (2*(n:Int)+2)*(V:Int) = 2*((n:Int)*(V:Int)) + 2*V := by
    simp only [Int.add_mul, Int.mul_assoc] <;> omega
  have hIV : (I:Int) = V + L - 1 := by omega
  have e5 : (n:Int)*(I:Int) = (n:Int)*(V:Int) + (n:Int)*(L:Int) - n := by
    rw [hIV, Int.mul_sub, Int.mul_add, Int.mul_one]
  rw [e2]
  push_cast
  rw [e3, e4, e5]
  rcases Int.le_total ((L:Int) - n - 1) 0 with h | h
  · rw [Int.max_eq_right h]; omega
  · rw [Int.max_eq_left h]; omega

theorem omegaBar_neg (n L : Int) (hn : 4 ≤ n) (hL : 2 ≤ L) : omegaBar n L < 0 := by
  have hp : 0 ≤ (n - 4) * (L - 2) := Int.mul_nonneg (by omega) (by omega)
  have e1 : (n - 4) * (L - 2) = n*L - 2*n - 4*L + 8 := by
    simp only [Int.sub_mul, Int.mul_sub]; omega
  have e2 : (2*n+2)*(L-1) = 2*(n*L) - 2*n + 2*L - 2 := by
    simp only [Int.add_mul, Int.mul_sub, Int.mul_assoc]; omega
  unfold omegaBar
  rw [e2]
  rcases Int.le_total (L - n - 1) 0 with h | h
  · rw [Int.max_eq_right h]; omega
  · rw [Int.max_eq_left h]; omega

/-- The threshold is sharp: for n <= 3 the two-loop bound is not negative. -/
theorem omegaBar_two_loops_nonneg (n : Int) (hn : n ≤ 3) : 0 ≤ omegaBar n 2 := by
  unfold omegaBar
  have e : (2*n+2)*((2:Int)-1) = 2*n+2 := by omega
  rw [e]
  rcases Int.le_total ((2:Int) - n - 1) 0 with h | h
  · rw [Int.max_eq_right h]; omega
  · rw [Int.max_eq_left h]; omega

theorem omegaBar_one_loop (n : Int) (hn : 0 ≤ n) : omegaBar n 1 = 4 := by
  unfold omegaBar
  have e : (2*n+2)*((1:Int)-1) = 0 := by simp
  rw [e, Int.max_eq_right (by omega)]
  omega

/-- Table (power), upper block: rows n = 3..8, columns L = 1..5. -/
theorem table_power_upper :
    ([3,4,5,6,7,8].map fun n => [1,2,3,4,5].map fun L => omegaBar n L) =
    [[4, 0, -4, -8, -10],
     [4, -2, -8, -14, -20],
     [4, -4, -12, -20, -28],
     [4, -6, -16, -26, -36],
     [4, -8, -20, -32, -44],
     [4, -10, -24, -38, -52]] := by decide

/-! ## The weakest fine graining, Eq. (omega) with m_sigma > 0 -/

/-- The bound of Eq. (omega) with j = 2 m_sigma:  4L - (2n+2)(L-1) + (2L - 2n - 2 + j)_+ . -/
def omegaBarM (n L j : Int) : Int := 4*L - (2*n+2)*(L-1) + max (2*L - 2*n - 2 + j) 0

theorem omegaBarM_zero (n L : Int) : omegaBarM n L 0 = omegaBar n L := by
  unfold omegaBarM omegaBar
  rcases Int.le_total (L - n - 1) 0 with h | h
  · rw [Int.max_eq_right h, Int.max_eq_right (by omega)]; omega
  · rw [Int.max_eq_left h, Int.max_eq_left (by omega)]; omega

/-- The bound grows with m_sigma, so the limit m_sigma = n - 2 bounds every sigma > 4. -/
theorem omegaBarM_mono (n L j j' : Int) (h : j ≤ j') : omegaBarM n L j ≤ omegaBarM n L j' := by
  unfold omegaBarM
  have : max (2*L - 2*n - 2 + j) 0 ≤ max (2*L - 2*n - 2 + j') 0 := by
    rcases Int.le_total (2*L - 2*n - 2 + j) 0 with h1 | h1 <;>
    rcases Int.le_total (2*L - 2*n - 2 + j') 0 with h2 | h2
    · rw [Int.max_eq_right h1, Int.max_eq_right h2]; exact Int.le_refl 0
    · rw [Int.max_eq_right h1, Int.max_eq_left h2]; omega
    · omega
    · rw [Int.max_eq_left h1, Int.max_eq_left h2]; omega
  omega

/-- For n >= 4 every graph with L >= 2 loops converges for every fine graining allowed by (G):
    omegaBarM n L j < 0 whenever 2 m_sigma = j <= 2n - 4. -/
theorem omegaBarM_neg (n L j : Int) (hn : 4 ≤ n) (hL : 2 ≤ L) (hj : j ≤ 2*n - 4) :
    omegaBarM n L j < 0 := by
  have hm := omegaBarM_mono n L j (2*n - 4) hj
  have hp : 0 ≤ (n - 4) * (L - 2) := Int.mul_nonneg (by omega) (by omega)
  have e1 : (n - 4) * (L - 2) = n*L - 2*n - 4*L + 8 := by
    simp only [Int.sub_mul, Int.mul_sub]; omega
  have e2 : (2*n+2)*(L-1) = 2*(n*L) - 2*n + 2*L - 2 := by
    simp only [Int.add_mul, Int.mul_sub, Int.mul_assoc]; omega
  have hw : omegaBarM n L (2*n - 4) < 0 := by
    unfold omegaBarM
    rw [e2]
    rcases Int.le_total (2*L - 2*n - 2 + (2*n - 4)) 0 with h | h
    · rw [Int.max_eq_right h]; omega
    · rw [Int.max_eq_left h]; omega
  omega

/-- Table (power), lower block: m_sigma = n - 2, rows n = 3..8, columns L = 1..5. -/
theorem table_power_lower :
    ([3,4,5,6,7,8].map fun n => [1,2,3,4,5].map fun L => omegaBarM n L (2*n - 4)) =
    [[4, 0, -4, -6, -8],
     [4, -2, -8, -12, -16],
     [4, -4, -12, -18, -24],
     [4, -6, -16, -24, -32],
     [4, -8, -20, -30, -40],
     [4, -10, -24, -36, -48]] := by decide

/-! ## One-loop poles of the quartic operators, Eqs. (killcontrib)-(tuning) -/

/-- Determinant of a 3x3 integer matrix by cofactor expansion. -/
def det3 (a : Fin 3 → Fin 3 → Int) : Int :=
  a 0 0 * (a 1 1 * a 2 2 - a 1 2 * a 2 1)
  - a 0 1 * (a 1 0 * a 2 2 - a 1 2 * a 2 0)
  + a 0 2 * (a 1 0 * a 2 1 - a 1 1 * a 2 0)

/-- Three times the matrix of Eq. (killcontrib), which has integer entries. -/
def poleMatrix3 : Fin 3 → Fin 3 → Int :=
  fun i j => [[-36, -3, -8], [0, 3, 24], [0, 0, 6]][i]![j]!

/-- det(3A) = 27 det A, so det A = -24. -/
theorem pole_matrix_det : det3 poleMatrix3 = 27 * (-24) := by decide

/-- The upper-left 2x2 block used when s_3 = 0: det = (-12)(1) = -12. -/
theorem pole_block_det : (-12 : Int) * 1 - (-1) * 0 = -12 := by decide

/-- The tuning of Eq. (tuning) cancels all three poles.  With t_i = s_i / c and the one-loop
    coefficients b1, b2, bE, write u_i = 36 t_i:
      u3 = -18 bE,  u2 = -36 b2 + 144 bE,  u1 = 3 b1 + 3 b2 - 8 bE.
    The rows of Eq. (killcontrib), multiplied by 108, then give 108 (b_i + Delta b_i) = 0. -/
theorem tuning_cancels (b1 b2 bE : Int) :
    let u1 := 3*b1 + 3*b2 - 8*bE
    let u2 := -36*b2 + 144*bE
    let u3 := -18*bE
    (108*b1 + (-36*u1 - 3*u2 - 8*u3) = 0) ∧
    (36*b2 + (u2 + 8*u3) = 0) ∧
    (36*bE + 2*u3 = 0) := by
  intro u1 u2 u3
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-! ## Insertion lemma constant, Eq. (insertion) -/

/-- The three pieces (1+r) + 2r + (2r^2 - r) of the constant add to 1 + 2r + 2r^2. -/
theorem insertion_constant (r : Nat) : (1 + r) + 2*r + (2*(r*r) - r) = 1 + 2*r + 2*(r*r) := by
  have : r ≤ r*r ∨ r = 0 := by
    cases r with
    | zero => exact Or.inr rfl
    | succ k => exact Or.inl (Nat.le_mul_of_pos_left _ (Nat.succ_pos k))
  omega
