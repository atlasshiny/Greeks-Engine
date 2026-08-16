import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.SplitIfs

namespace Proofs.Greeks

def call_payoff (S K : ℝ) : ℝ := max (S - K) 0
def put_payoff (S K : ℝ) : ℝ := max (K - S) 0

theorem put_call_parity_at_expiry (S K : ℝ) :
    call_payoff S K - put_payoff S K = S - K := by
  unfold call_payoff put_payoff
  rw [max_def, max_def]
  split
  · -- Case 1: S - K ≤ 0
    split
    · linarith
    · linarith
  · -- Case 2: ¬(S - K ≤ 0)
    split
    · linarith
    · linarith

end Proofs.Greeks
