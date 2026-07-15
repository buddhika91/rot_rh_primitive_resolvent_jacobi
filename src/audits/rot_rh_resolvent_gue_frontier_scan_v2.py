#!/usr/bin/env python3
"""
Fast stability-versus-GUE frontier audit.

Reuses the *_nodes.csv produced by
rot_rh_primitive_resolvent_zero_preserving_gue_audit.py.
It does not recompute Xi, moments, or Jacobi matrices.

For every depth, tolerance, trim fraction, and unfolding method, it finds the
longest consecutive cross-depth-stable block, computes GUE/GOE/Poisson spacing
statistics, zero-location error, and a bootstrap GUE-win fraction.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import kstest


def parse_float_list(text: str) -> List[float]:
    vals=[float(x.strip()) for x in text.split(',') if x.strip()]
    if not vals: raise argparse.ArgumentTypeError('Expected comma-separated numbers')
    return vals

def parse_int_list(text: str) -> List[int]:
    vals=[int(x.strip()) for x in text.split(',') if x.strip()]
    if not vals: raise argparse.ArgumentTypeError('Expected comma-separated integers')
    return vals

def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields=[]; seen=set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def longest_true_block(mask: np.ndarray) -> tuple[int,int]:
    best=(0,0); start=None
    for i,v in enumerate(mask):
        if v and start is None: start=i
        if (not v or i==len(mask)-1) and start is not None:
            end=i+1 if v and i==len(mask)-1 else i
            if end-start > best[1]-best[0]: best=(start,end)
            start=None
    return best

def select_block(frame, rel_tol, abs_tol, trim_fraction):
    frame=frame.sort_values('index').reset_index(drop=True)
    rel=frame['relative_depth_difference'].to_numpy(float)
    absv=frame['absolute_depth_difference'].to_numpy(float)
    mask=(rel<=rel_tol)|(absv<=abs_tol)
    start,end=longest_true_block(mask)
    count=end-start
    trim=int(math.floor(trim_fraction*count))
    if count-2*trim<4: trim=0
    bstart=start+trim; bend=end-trim
    return {
        'stable_start':start,'stable_end':end,'stable_count':count,
        'bulk_start':bstart,'bulk_end':bend,'bulk_count':bend-bstart,
        'block':frame.iloc[bstart:bend].copy()
    }

def rvm_count(t):
    t=np.asarray(t,float); x=np.maximum(t/(2*np.pi),1e-15)
    return x*np.log(x)-x+7/8

def unfold(gamma, method, degree):
    gamma=np.sort(np.asarray(gamma,float))
    if method=='rvm': u=rvm_count(gamma)
    elif method=='polynomial':
        idx=np.arange(1, len(gamma) + 1, dtype=float); deg=min(degree,max(1,len(gamma)-2))
        u=np.polyval(np.polyfit(gamma,idx,deg),gamma)
    elif method=='local':
        s=np.diff(gamma)
        if len(s)<3: return np.arange(len(gamma),dtype=float)
        w=max(3,min(9,(len(s)//3)*2+1)); kernel=np.ones(w)/w
        padded=np.pad(s,(w//2,w//2),mode='edge')
        local=np.convolve(padded,kernel,mode='valid')[:len(s)]
        u=np.concatenate([[0.0],np.cumsum(s/np.maximum(local,1e-15))])
    else: raise ValueError(method)
    u=np.maximum.accumulate(u)
    if len(u)>1:
        m=np.mean(np.diff(u))
        if m>0: u=u/m
    return u

def gue_cdf(s):
    s=np.maximum(np.asarray(s,float),0)
    return np.vectorize(math.erf)(2*s/math.sqrt(np.pi))-(4/np.pi)*s*np.exp(-4*s*s/np.pi)

def goe_cdf(s):
    s=np.maximum(np.asarray(s,float),0); return 1-np.exp(-np.pi*s*s/4)

def poisson_cdf(s):
    s=np.maximum(np.asarray(s,float),0); return 1-np.exp(-s)

def ratio_pdf(r,beta):
    r=np.asarray(r,float)
    raw=(r+r*r)**beta/(1+r+r*r)**(1+1.5*beta)
    g=np.linspace(0,1,40001)
    rg=(g+g*g)**beta/(1+g+g*g)**(1+1.5*beta)
    return raw/max(np.trapezoid(rg,g),1e-15)

def ratio_cdf_factory(beta):
    g=np.linspace(0,1,50001); pdf=ratio_pdf(g,beta); dx=g[1]-g[0]
    c=np.cumsum(pdf)*dx; c/=c[-1]
    return lambda x: np.interp(x,g,c,left=0,right=1)
RATIO_GUE=ratio_cdf_factory(2); RATIO_GOE=ratio_cdf_factory(1)
def ratio_poisson(r):
    r=np.clip(np.asarray(r,float),0,1); return 2*r/(1+r)

def ratios(spacings):
    s=np.asarray(spacings,float)
    if len(s)<2:return np.array([],float)
    a=s[:-1]; b=s[1:]; d=np.maximum(a,b); m=d>0
    return np.minimum(a[m],b[m])/d[m]

def stats(gamma,method,degree):
    gamma=np.sort(np.asarray(gamma,float))
    if len(gamma)<4:
        return {'spacing_count':max(0,len(gamma)-1),'ks_gue':np.nan,'ks_goe':np.nan,'ks_poisson':np.nan,
                'mean_ratio':np.nan,'ratio_ks_gue':np.nan,'ratio_ks_goe':np.nan,'ratio_ks_poisson':np.nan,
                'spacings':np.array([],float)}
    s=np.diff(unfold(gamma,method,degree)); s=s[np.isfinite(s)&(s>=0)]
    if len(s): s=s/max(np.mean(s),1e-15)
    r=ratios(s)
    return {
        'spacing_count':len(s),
        'ks_gue':float(kstest(s,gue_cdf).statistic) if len(s)>=2 else np.nan,
        'ks_goe':float(kstest(s,goe_cdf).statistic) if len(s)>=2 else np.nan,
        'ks_poisson':float(kstest(s,poisson_cdf).statistic) if len(s)>=2 else np.nan,
        'mean_ratio':float(np.mean(r)) if len(r) else np.nan,
        'ratio_ks_gue':float(kstest(r,RATIO_GUE).statistic) if len(r)>=2 else np.nan,
        'ratio_ks_goe':float(kstest(r,RATIO_GOE).statistic) if len(r)>=2 else np.nan,
        'ratio_ks_poisson':float(kstest(r,ratio_poisson).statistic) if len(r)>=2 else np.nan,
        'spacings':s,
    }

def bootstrap(spacings,trials,seed):
    s=np.asarray(spacings,float)
    if len(s)<4 or trials<=0:
        return {'bootstrap_trials':0,'gue_win_fraction':np.nan,'median_ks_gue':np.nan,'median_ks_goe':np.nan,'median_ks_poisson':np.nan}
    rng=np.random.default_rng(seed); g=[];o=[];p=[];wins=0
    for _ in range(trials):
        x=rng.choice(s,size=len(s),replace=True); x=x/max(np.mean(x),1e-15)
        kg=float(kstest(x,gue_cdf).statistic); ko=float(kstest(x,goe_cdf).statistic); kp=float(kstest(x,poisson_cdf).statistic)
        g.append(kg);o.append(ko);p.append(kp); wins+=int(kg<ko and kg<kp)
    return {'bootstrap_trials':trials,'gue_win_fraction':wins/trials,'median_ks_gue':float(np.median(g)),
            'median_ks_goe':float(np.median(o)),'median_ks_poisson':float(np.median(p))}

def build_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--nodes-file',required=True)
    p.add_argument('--depths',type=parse_int_list,default=[])
    p.add_argument('--relative-tolerances',type=parse_float_list,default=[1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5])
    p.add_argument('--absolute-tolerances',type=parse_float_list,default=[1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5])
    p.add_argument('--trim-fractions',type=parse_float_list,default=[0,0.05,0.10,0.15])
    p.add_argument('--unfold-methods',default='rvm,polynomial,local')
    p.add_argument('--poly-degree',type=int,default=5)
    p.add_argument('--minimum-spacings-list',type=parse_int_list,default=[8,12,16,20])
    p.add_argument('--bootstrap-trials',type=int,default=500)
    p.add_argument('--seed',type=int,default=20260714)
    p.add_argument('--out-prefix',default='rot_rh_resolvent_gue_stability_frontier')
    return p

def main():
    args=build_parser().parse_args(); path=Path(args.nodes_file).expanduser().resolve()
    if not path.exists(): raise SystemExit(f'Nodes file not found: {path}')
    df=pd.read_csv(path)
    req={'depth','index','gamma','relative_depth_difference','absolute_depth_difference','target_zero'}
    miss=req-set(df.columns)
    if miss: raise SystemExit(f'Missing required columns: {sorted(miss)}')
    all_depths=sorted(int(x) for x in df.depth.unique()); depths=args.depths or all_depths
    depths=[d for d in depths if d in all_depths]
    methods=[x.strip() for x in args.unfold_methods.split(',') if x.strip()]
    prefix=Path(args.out_prefix).expanduser().resolve(); prefix.parent.mkdir(parents=True,exist_ok=True)
    print('='*126); print('ROT-RH / GUE STABILITY FRONTIER AUDIT'); print('='*126)
    print('nodes file            :',path); print('depths                :',depths)
    print('relative tolerances   :',args.relative_tolerances); print('trim fractions        :',args.trim_fractions)
    print('unfold methods        :',methods); print('='*126)
    rows=[]
    for depth in depths:
        f=df[df.depth==depth].copy()
        for rt in args.relative_tolerances:
            for at in args.absolute_tolerances:
                for trim in args.trim_fractions:
                    sel=select_block(f,rt,at,trim); block=sel['block']
                    gamma=block.gamma.to_numpy(float); target=block.target_zero.to_numpy(float)
                    if len(gamma):
                        e=gamma-target; zrmse=float(np.sqrt(np.mean(e*e))); zrel=float(np.sqrt(np.mean((e/np.maximum(np.abs(target),1e-300))**2)))
                        zmax=float(np.max(np.abs(e))); maxdr=float(np.max(block.relative_depth_difference)); meandr=float(np.mean(block.relative_depth_difference))
                    else: zrmse=zrel=zmax=maxdr=meandr=np.nan
                    for mi,method in enumerate(methods):
                        st=stats(gamma,method,args.poly_degree)
                        boot=bootstrap(st['spacings'],args.bootstrap_trials,args.seed+100000*depth+1000*mi+int(abs(math.log10(rt)))*10)
                        guewin=bool(np.isfinite(st['ks_gue']) and st['ks_gue']<st['ks_goe'] and st['ks_gue']<st['ks_poisson'])
                        for gate in args.minimum_spacings_list:
                            rows.append({
                                'depth':depth,'relative_tolerance':rt,'absolute_tolerance':at,'trim_fraction':trim,
                                'unfolding':method,'minimum_spacings_gate':gate,'eligible':st['spacing_count']>=gate,'gue_wins':guewin,
                                'stable_start_index':sel['stable_start']+1,'stable_end_index':sel['stable_end'],'stable_count':sel['stable_count'],
                                'bulk_start_index':sel['bulk_start']+1,'bulk_end_index':sel['bulk_end'],'bulk_count':sel['bulk_count'],
                                'spacing_count':st['spacing_count'],'ks_gue':st['ks_gue'],'ks_goe':st['ks_goe'],'ks_poisson':st['ks_poisson'],
                                'gue_margin_vs_goe':st['ks_goe']-st['ks_gue'],'gue_margin_vs_poisson':st['ks_poisson']-st['ks_gue'],
                                'mean_ratio':st['mean_ratio'],'ratio_ks_gue':st['ratio_ks_gue'],'ratio_ks_goe':st['ratio_ks_goe'],
                                'ratio_ks_poisson':st['ratio_ks_poisson'],'zero_rmse':zrmse,'zero_relative_rmse':zrel,'zero_max_abs_error':zmax,
                                'max_relative_depth_drift':maxdr,'mean_relative_depth_drift':meandr,**boot})
    eligible=[r for r in rows if r['eligible'] and np.isfinite(r['ks_gue'])]
    best_raw=min(eligible,key=lambda r:r['ks_gue']) if eligible else None
    robust=[r for r in eligible if r['gue_wins'] and r['gue_win_fraction']>=0.60 and 0.55<=r['mean_ratio']<=0.65]
    best_robust=max(robust,key=lambda r:(r['spacing_count'],r['gue_margin_vs_poisson'],r['gue_win_fraction'],-r['ks_gue'])) if robust else None
    groups={}
    for r in eligible:
        key=(r['depth'],r['relative_tolerance'],r['absolute_tolerance'],r['trim_fraction'],r['minimum_spacings_gate'])
        groups.setdefault(key,[]).append(r)
    consensus=[]
    for key,g in groups.items():
        wins=sum(bool(r['gue_wins']) for r in g); n=len(g)
        consensus.append({'depth':key[0],'relative_tolerance':key[1],'absolute_tolerance':key[2],'trim_fraction':key[3],
                          'minimum_spacings_gate':key[4],'methods_present':n,'gue_winning_methods':wins,'gue_consensus_fraction':wins/n,
                          'minimum_spacing_count':min(r['spacing_count'] for r in g),'mean_ks_gue':float(np.mean([r['ks_gue'] for r in g])),
                          'mean_ks_goe':float(np.mean([r['ks_goe'] for r in g])),'mean_ks_poisson':float(np.mean([r['ks_poisson'] for r in g])),
                          'mean_ratio':float(np.nanmean([r['mean_ratio'] for r in g])),
                          'mean_bootstrap_gue_win_fraction':float(np.nanmean([r['gue_win_fraction'] for r in g]))})
    best_cons=max(consensus,key=lambda r:(r['gue_consensus_fraction'],r['minimum_spacing_count'],r['mean_bootstrap_gue_win_fraction'],-r['mean_ks_gue'])) if consensus else None
    save_csv(Path(str(prefix)+'_frontier.csv'),rows); save_csv(Path(str(prefix)+'_consensus.csv'),consensus)
    Path(str(prefix)+'_report.json').write_text(json.dumps({'nodes_file':str(path),'args':vars(args),'best_raw':best_raw,'best_robust':best_robust,'best_consensus':best_cons},indent=2),encoding='utf-8')
    print(); print('='*126); print('FINAL GUE STABILITY FRONTIER'); print('='*126)
    if best_raw:
        print(f"best raw: depth={best_raw['depth']} tol={best_raw['relative_tolerance']:.1e} trim={best_raw['trim_fraction']:.2f} method={best_raw['unfolding']} spacings={best_raw['spacing_count']}")
        print(f"KS GUE/GOE/Poi={best_raw['ks_gue']:.6f}/{best_raw['ks_goe']:.6f}/{best_raw['ks_poisson']:.6f} mean_r={best_raw['mean_ratio']:.6f} bootstrap={best_raw['gue_win_fraction']:.3f}")
    if best_robust:
        print('-'*126); print('BEST ROBUST ROW')
        print(f"depth={best_robust['depth']} tol={best_robust['relative_tolerance']:.1e} trim={best_robust['trim_fraction']:.2f} method={best_robust['unfolding']} spacings={best_robust['spacing_count']}")
        print(f"KS GUE/GOE/Poi={best_robust['ks_gue']:.6f}/{best_robust['ks_goe']:.6f}/{best_robust['ks_poisson']:.6f} mean_r={best_robust['mean_ratio']:.6f} bootstrap={best_robust['gue_win_fraction']:.3f}")
        print(f"zero relRMSE={best_robust['zero_relative_rmse']:.3e}")
    else:
        print('-'*126); print('NO ROBUST GUE FRONTIER ROW FOUND')
    if best_cons:
        print('-'*126); print(f"best consensus: depth={best_cons['depth']} tol={best_cons['relative_tolerance']:.1e} trim={best_cons['trim_fraction']:.2f} GUE methods={best_cons['gue_winning_methods']}/{best_cons['methods_present']} min spacings={best_cons['minimum_spacing_count']}")
    print('outputs:',str(prefix)+'_*.csv/json'); print('='*126)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
