# reconstruction
AR reconstruction analysis scripts

Input and info files:
gaptimes.csv
restlist.csv
unitlists/

Stimulus reconstruction models and analysis:
reconstruct_serve.py
stimcorr.py

Posterior predictive likelihood analysis using reconstruction models:
ar_prep.py
ar_prob.py
ar_prob_interval.py
ar_prob_loo.py

Miscellaneous plotting and analysis code fragments:
neurogram.py
prob_analysis.py
dissertation_plots.py
correlations.R

# Fetching pprox files
On a computer with local Neurobank files, to create symlinks to all pprox files,
```
cat units.txt | xargs nbank locate -L .
```
