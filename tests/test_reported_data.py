from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def test_depth_flow_improves():
 d=pd.read_csv(ROOT/'data/reported/rot_rh_primitive_resolvent_validation_stable_depth_flow.csv'); assert d.iloc[-1].penalized_relative_rmse < d.iloc[0].penalized_relative_rmse
def test_signal_control_full_coverage():
 d=pd.read_csv(ROOT/'data/reported/rot_rh_primitive_resolvent_validation_stable_control_audit.csv'); s=d[d.control=='signal'].iloc[0]; assert s.coverage==1.0
def test_all_reported_gates_pass():
 d=pd.read_csv(ROOT/'data/reported/rot_rh_primitive_resolvent_generator_gates.csv'); statuses=[]
 for c in ['status','status_h0','status_h1']:
  if c in d: statuses += d[c].dropna().tolist()
 assert statuses and all(x=='PASS' for x in statuses)
