import Mathlib.Tactic.Linarith
import Mathlib.Data.Real.Basic

namespace Proofs.Greeks

def delta_call (cdf_d1 : ℝ) : ℝ := cdf_d1
def delta_put (cdf_d1 : ℝ) : ℝ := cdf_d1 - 1

theorem bsm_delta_parity (cdf_d1 : ℝ) :
  delta_call cdf_d1 - delta_put cdf_d1 = 1 := by
  unfold delta_call delta_put
  linarith
end Proofs.Greeks
