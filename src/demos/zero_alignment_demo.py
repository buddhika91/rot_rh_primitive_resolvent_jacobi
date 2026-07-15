#!/usr/bin/env python3
"""Summarize zero alignment from the reported canonical generator run."""
from pathlib import Path
import argparse, pandas as pd, numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument('--predictions',default='data/reported/rot_rh_primitive_resolvent_generator_predictions.csv'); p.add_argument('--control',default='signal'); p.add_argument('--depth',type=int,default=20); p.add_argument('--out',default='outputs/zero_alignment_summary.csv'); a=p.parse_args()
 df=pd.read_csv(a.predictions); df=df[(df.control==a.control)&(df.depth==a.depth)].copy(); df['relative_error']=df.abs_error/df.target
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.out,index=False)
 print(df[['index','predicted','target','abs_error','relative_error']].to_string(index=False)); print('RMSE=',np.sqrt(np.mean((df.predicted-df.target)**2)))
if __name__=='__main__': main()
