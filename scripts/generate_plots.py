#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/reported'; USER=ROOT/'data/user_reported'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)

def save(name): plt.tight_layout(); plt.savefig(FIG/name,dpi=180); plt.close()
# depth convergence
df=pd.read_csv(DATA/'rot_rh_primitive_resolvent_validation_stable_depth_flow.csv'); plt.figure(figsize=(6,4)); plt.semilogy(df.depth,df.penalized_relative_rmse,marker='o'); plt.xlabel('Jacobi depth'); plt.ylabel('Relative RMSE'); plt.title('Zero reconstruction convergence'); plt.grid(True,alpha=.3); save('depth_convergence.png')
# zero alignment depth24
p=pd.read_csv(DATA/'rot_rh_primitive_resolvent_validation_stable_predictions.csv'); p=p[p.depth==24]; plt.figure(figsize=(7,4)); plt.plot(p['index'],p.target,marker='o',label='Target'); plt.plot(p['index'],p.predicted,marker='x',label='Jacobi reconstruction'); plt.xlabel('Zero index'); plt.ylabel('Ordinate'); plt.title('Depth-24 zero alignment'); plt.legend(); plt.grid(True,alpha=.3); save('zero_alignment_depth24.png')
plt.figure(figsize=(7,4)); plt.semilogy(p['index'],np.maximum(p.abs_error,1e-18),marker='o'); plt.xlabel('Zero index'); plt.ylabel('Absolute error'); plt.title('Depth-24 zero alignment error'); plt.grid(True,alpha=.3); save('zero_alignment_error_depth24.png')
# controls
c=pd.read_csv(DATA/'rot_rh_primitive_resolvent_validation_stable_control_audit.csv'); plt.figure(figsize=(7,4)); plt.bar(c.control,np.maximum(c.penalized_relative_rmse,1e-18)); plt.yscale('log'); plt.ylabel('Penalized relative RMSE'); plt.title('Signal survival against controls'); plt.xticks(rotation=20); save('control_survival.png')
# moments
m=pd.read_csv(DATA/'rot_rh_primitive_resolvent_generator_moments.csv'); plt.figure(figsize=(7,4)); plt.semilogy(m['index'],np.abs(m.moment.astype(float)),marker='.',linestyle='none'); plt.xlabel('Moment index'); plt.ylabel('|r_n|'); plt.title('Primitive resolvent moments'); plt.grid(True,alpha=.3); save('resolvent_moments.png')
# jacobi coeffs depth24
j=pd.read_csv(DATA/'rot_rh_primitive_resolvent_validation_stable_jacobi_coefficients.csv'); j=j[j.depth==24]; plt.figure(figsize=(7,4));
for kind,g in j.groupby('kind'): plt.plot(g['index'],g.value,marker='o',label=kind)
plt.xlabel('Index'); plt.ylabel('Coefficient'); plt.title('Canonical Jacobi coefficients at depth 24'); plt.legend(); plt.grid(True,alpha=.3); save('jacobi_coefficients_depth24.png')
# Carleman
car=pd.read_csv(DATA/'rot_rh_primitive_resolvent_validation_stable_carleman.csv'); plt.figure(figsize=(7,4)); plt.plot(car.n,car.partial_sum,marker='o'); plt.xlabel('n'); plt.ylabel('Partial sum'); plt.title('Carleman determinacy diagnostic'); plt.grid(True,alpha=.3); save('carleman_partial_sum.png')
# preregistered summary
u=pd.read_csv(USER/'preregistered_depth64_summary.csv'); plt.figure(figsize=(7,4)); plt.bar(u.depth,u.stable_spacings); plt.axhline(28,linestyle='--',label='Preregistered minimum'); plt.xlabel('Depth'); plt.ylabel('Stable spacings'); plt.title('Preregistered GUE holdout summary'); plt.legend(); save('preregistered_depth64_spacings.png')
print(f'Wrote figures to {FIG}')
