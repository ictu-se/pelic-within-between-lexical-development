"""Reproduce post hoc coefficient contrasts without changing original results."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_experiment import fit_hybrid,METRICS

def main():
 data=pd.read_csv(ROOT/'results/analysis_metrics.csv');rows=[]
 original=pd.read_csv(ROOT/'results/hybrid_model_results.csv')
 for metric in METRICS:
  fit=fit_hybrid(data,metric,'jslw_contrast')['fit'];cov=fit.cov_params();b=fit.params
  for term in ['level_within','level_between']:
   old=original.query('analysis == "primary" and metric == @metric and term == @term').estimate_std.iloc[0]
   assert abs(b[term]-old)<1e-7,(metric,term,b[term],old)
  if not fit.converged:
   raise RuntimeError(f'Model did not converge: {metric}')
  diff=b['level_between']-b['level_within']
  se=np.sqrt(cov.loc['level_between','level_between']+cov.loc['level_within','level_within']-2*cov.loc['level_between','level_within'])
  rows.append(dict(metric=metric,within=b['level_within'],between=b['level_between'],difference=diff,se=se,ci_low=diff-1.96*se,ci_high=diff+1.96*se,p=2*norm.sf(abs(diff/se)),converged=bool(fit.converged)))
 result=pd.DataFrame(rows);result['q']=multipletests(result.p,method='fdr_bh')[1]
 (ROOT/'results').mkdir(exist_ok=True)
 result.to_csv(ROOT/'results/within_between_contrasts.csv',index=False)
 print(result.to_string(index=False))
if __name__=='__main__':main()
