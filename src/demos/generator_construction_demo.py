#!/usr/bin/env python3
"""Demonstrate the same Jacobi operator as a generator of unitary evolution."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import xi_taylor_cauchy, phi_from_xi_coeffs, primitive_resolvent, stieltjes_jacobi, jacobi_matrix

def main():
 p=argparse.ArgumentParser(); p.add_argument('--depth',type=int,default=20); p.add_argument('--time',type=float,default=1.0); p.add_argument('--dps',type=int,default=140); p.add_argument('--radius',type=float,default=4); p.add_argument('--samples',type=int,default=1024); p.add_argument('--out-prefix',default='outputs/generator_demo'); a=p.parse_args()
 mp.mp.dps=a.dps; need=2*a.depth+8; po=need+2; xo=2*po+2
 c=xi_taylor_cauchy(xo,mp.mpf(str(a.radius)),a.samples); phi=phi_from_xi_coeffs(c,po); moments=[mp.re(x) for x in primitive_resolvent(phi,need)]
 r=stieltjes_jacobi(moments,a.depth,mp.power(10,-a.dps//2)); J=jacobi_matrix(r); U=expm(-1j*a.time*J)
 psi=np.zeros(a.depth,dtype=complex); psi[0]=1; evolved=U@psi
 prefix=Path(a.out_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
 np.savetxt(str(prefix)+'_unitary_real.csv',U.real,delimiter=','); np.savetxt(str(prefix)+'_unitary_imag.csv',U.imag,delimiter=',')
 report={'depth':a.depth,'time':a.time,'unitarity_defect':float(np.linalg.norm(U.conj().T@U-np.eye(a.depth))/a.depth),'initial_norm':float(np.linalg.norm(psi)),'evolved_norm':float(np.linalg.norm(evolved)),'interpretation':'J is the operator; when used in exp(-itJ), it is the generator of unitary evolution.'}
 Path(str(prefix)+'_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
