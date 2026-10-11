"""Phase 1. Fixed formal-input full-Linear profile; no new quality search."""
import json,time
import run_v26 as r

def main():
 r.verify_sources()
 if (r.O/'PROFILE_DECISION.json').exists() and (r.O/'PROFILE_ORIGINAL.json').exists():
  original=json.loads((r.O/'PROFILE_ORIGINAL.json').read_text('utf-8'))
  decision=json.loads((r.O/'PROFILE_DECISION.json').read_text('utf-8'))
  if original['status']=='COMPLETE' and decision['status']=='PROFILE_COMPLETE':
   print(json.dumps(dict(status='PROFILE_REUSED',decision=decision['decision'],gpu_seconds=decision['gpu_wall_seconds']),indent=2));return
 budget=r.Budget();r.write('RUN_STATUS.json',dict(status='STARTING_PROFILE',gpu_wall_consumed_seconds=budget.used()))
 model,pristine=r.load_model_and_weights(budget)
 inputs=r.capture_inputs(model,budget)
 profile=r.profile_original(model,pristine,inputs,budget)
 decision={}
 for name,v in profile.items():
  rows=v['records'];factor=[x['gpu_seconds'].get('lower_cholesky',0)+x['gpu_seconds'].get('upper_cholesky',0) for x in rows]
  avg=sum(factor)/len(factor);wall=sum(x['candidate_wall_seconds'] for x in rows)/len(rows)
  spread=max(max(x['candidate_wall_seconds'] for x in rows if x['alpha']==alpha)-min(x['candidate_wall_seconds'] for x in rows if x['alpha']==alpha) for alpha in [1.0,.8])
  upper=sum(x['gpu_seconds'].get('upper_cholesky',0) for x in rows)/len(rows)
  net=avg-upper/len(rows)
  decision[name]=dict(factor_mean_seconds=avg,candidate_mean_seconds=wall,factor_share=avg/wall,warmed_repeat_range_seconds=spread,
   one_time_upper_build_estimate_seconds=upper,net_factor_saving_per_candidate_for_four_calls_seconds=net,
   meets_both=avg/wall>=.05 and net>2*spread)
 proceed=all(x['meets_both'] for x in decision.values())
 r.write('PROFILE_DECISION.json',dict(status='PROFILE_COMPLETE',decision=decision,implement_full_factor_cache=proceed,
  frozen_threshold=r.P['decision'],gpu_wall_seconds=budget.used(),reason='Both full matrices must exceed factor-share and noise gates'))
 r.write('PROFILE_ORIGINAL.json',dict(status='COMPLETE',targets=profile,gpu_wall_seconds=budget.used()))
 budget.save('PROFILE_COMPLETE',implement_full_factor_cache=proceed)
 print(json.dumps(dict(implement_cache=proceed,decision=decision,gpu_seconds=budget.used()),indent=2),flush=True)
if __name__=='__main__':main()
