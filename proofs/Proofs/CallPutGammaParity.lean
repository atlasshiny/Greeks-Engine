import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.sqrt

namespace Proofs.Greeks

noncomputable def CallGamma (pdf_d1 : ℝ) (S σ T : ℝ) : ℝ :=
  pdf_d1 / (S * σ * Real.sqrt T)
noncomputable def PutGamma (pdf_d1 : ℝ) (S σ T : ℝ) : ℝ :=
  pdf_d1 / (S * σ * Real.sqrt T)

theorem call_put_gamma_parity (pdf_d1 S σ T : ℝ) :
  CallGamma pdf_d1 S σ T - PutGamma pdf_d1 S σ T = 0 := by
  unfold CallGamma PutGamma
  linarith

end Proofs.Greeks
