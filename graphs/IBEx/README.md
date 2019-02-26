# Workflow
**1) Run simulation**
`../../gsl/bin/gslparser IBEx_2019_v01.gsl`

**2) First stage processing: time series and connectivities** 
From matlab run script DataProcessIBEx_2019_v01.m.
Make sure the filepath (line 36) matches your simulation output directory. 

**3) Second stage processing: SVD** 
From matlab run script SVD_SORN_Analysis_2019_v01.m.
Make sure the filepath (line 16) matches your simulation output directory, or comment line. 
