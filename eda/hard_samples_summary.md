# Per-setting hard sample summary

### ID
- N=491, MAE=0.273, bias=-0.038 → no strong global bias.
- |residual| by y_true quartile: Q1=0.36, Q2=0.23, Q3=0.21, Q4=0.29
- signed residual by quartile: Q1=+0.22, Q2=-0.01, Q3=-0.11, Q4=-0.25
- top hardness drivers (permutation): FpDensityMorgan3, FpDensityMorgan1, FpDensityMorgan2, MolLogP, Monomer_Length_in_Main_Chain
- hard-10% means (vs rest):
  • FpDensityMorgan3: 1.707 (hard) vs 1.710 (rest), Δ=-0.003
  • FpDensityMorgan1: 0.632 (hard) vs 0.631 (rest), Δ=+0.001
  • FpDensityMorgan2: 1.148 (hard) vs 1.151 (rest), Δ=-0.002
  • MolLogP: 2.841 (hard) vs 2.982 (rest), Δ=-0.141
  • Monomer_Length_in_Main_Chain: 6.800 (hard) vs 6.812 (rest), Δ=-0.012

### OD
- N=488, MAE=0.377, bias=+0.193 → over-prediction (model says more permeable).
- |residual| by y_true quartile: Q1=0.74, Q2=0.30, Q3=0.20, Q4=0.27
- signed residual by quartile: Q1=+0.73, Q2=+0.22, Q3=-0.00, Q4=-0.18
- top hardness drivers (permutation): MinPartialCharge, MolLogP, FpDensityMorgan1, LabuteASA, NumAromaticRings
- hard-10% means (vs rest):
  • MinPartialCharge: -0.429 (hard) vs -0.351 (rest), Δ=-0.077
  • MolLogP: 1.853 (hard) vs 3.121 (rest), Δ=-1.268
  • FpDensityMorgan1: 0.629 (hard) vs 0.620 (rest), Δ=+0.009
  • LabuteASA: 306.581 (hard) vs 330.078 (rest), Δ=-23.497
  • NumAromaticRings: 1.918 (hard) vs 2.210 (rest), Δ=-0.291

### Cliff_ratio
- N=493, MAE=0.284, bias=-0.018 → no strong global bias.
- |residual| by y_true quartile: Q1=0.39, Q2=0.27, Q3=0.19, Q4=0.28
- signed residual by quartile: Q1=+0.24, Q2=-0.01, Q3=-0.07, Q4=-0.24
- top hardness drivers (permutation): FpDensityMorgan3, FpDensityMorgan2, MolLogP, MinPartialCharge, FpDensityMorgan1
- hard-10% means (vs rest):
  • FpDensityMorgan3: 1.707 (hard) vs 1.721 (rest), Δ=-0.014
  • FpDensityMorgan2: 1.149 (hard) vs 1.158 (rest), Δ=-0.009
  • MolLogP: 2.508 (hard) vs 2.922 (rest), Δ=-0.414
  • MinPartialCharge: -0.385 (hard) vs -0.382 (rest), Δ=-0.003
  • FpDensityMorgan1: 0.634 (hard) vs 0.635 (rest), Δ=-0.001

