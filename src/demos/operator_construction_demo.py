#!/usr/bin/env python3
"""Construct the finite canonical Jacobi operator and export its matrix."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import mpmath as mp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import xi_taylor_cauchy, phi_from_xi_coeffs, primitive_resolvent, stieltjes_jacobi, jacobi_matrix, predict_gamma

def main():
 p=argparse.ArgumentParser(); p.add_argument('--depth',type=int,default=24); p.add_argument('--dps',type=int,default=180); p.add_argument('--radius',type=float,default=4); p.add_argument('--samples',type=int,default=2048); p.add_argument('--out-prefix',default='outputs/operator_demo'); a=p.parse_args()
 mp.mp.dps=a.dps; need=2*a.depth+8; phi_order=need+2; xi_order=2*phi_order+2
 coeff=xi_taylor_cauchy(xi_order,mp.mpf(str(a.radius)),a.samples); phi=phi_from_xi_coeffs(coeff,phi_order); moments=[mp.re(x) for x in primitive_resolvent(phi,need)]
 result=stieltjes_jacobi(moments,a.depth,mp.power(10,-a.dps//2)); J=jacobi_matrix(result); gamma=predict_gamma(result)
 prefix=Path(a.out_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
 np.savetxt(str(prefix)+'_matrix.csv',J,delimiter=',')
 with open(str(prefix)+'_coefficients.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['kind','index','value']); [w.writerow(['alpha',i,float(v)]) for i,v in enumerate(result.alpha)]; [w.writerow(['beta',i+1,float(v)]) for i,v in enumerate(result.beta)]
 with open(str(prefix)+'_spectrum.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['index','gamma']); [w.writerow([i+1,float(v)]) for i,v in enumerate(gamma)]
 report={'depth':a.depth,'self_adjoint_defect':float(np.linalg.norm(J-J.T)/max(np.linalg.norm(J),1e-300)),'orthogonality_defect':float(result.orthogonality_defect),'breakdown_index':result.breakdown_index,'positive_beta':bool(all(v>0 for v in result.beta))}
 Path(str(prefix)+'_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
