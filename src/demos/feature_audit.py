#!/usr/bin/env python3
from pathlib import Path
import json
features={
 'primitive_xi_resolvent':True,'positive_moment_gates':True,'hankel_and_shifted_hankel_tests':True,'positive_s_fraction_tests':True,'canonical_stieltjes_recursion':True,'positive_jacobi_off_diagonals':True,'self_adjoint_finite_operator':True,'nested_prefix_consistency':True,'zero_alignment_demo':True,'precision_radius_sample_stability':True,'direct_derivative_crosscheck':True,'carleman_diagnostic':True,'control_survival_tests':True,'gue_spacing_tests':True,'spacing_ratio_tests':True,'multiple_unfoldings':True,'bootstrap_gue_validation':True,'preregistered_holdout_depth64':True,'global_positivity_proof':False,'rh_proof':False}
Path('outputs').mkdir(exist_ok=True); Path('outputs/feature_audit.json').write_text(json.dumps(features,indent=2)); print(json.dumps(features,indent=2))
