#!/usr/bin/env python3
"""Summarize survival against signflip, permutation, and Gaussian controls."""
import argparse, pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',default='data/reported/rot_rh_primitive_resolvent_validation_stable_control_audit.csv'); a=p.parse_args(); df=pd.read_csv(a.input); print(df[['control','count','coverage','penalized_relative_rmse','breakdown_index','min_beta']].to_string(index=False))
if __name__=='__main__': main()
