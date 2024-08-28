## Synoptic Flow

This directory contains scripts to extract and save deep layer mean (DLM) flow from the ERA5 datasets on Gadi, and then use this data to simulate TC tracks with the beta advection model (BAM).

First extract the DLM:

`qsub era5_dlm.sh`

Once this is complete, ensure that the `DATA_DIR` variable in `bam_tracks.py` points to the location of extracted DLM data and that this location also has the BoM best track and OTCR data. Then cyclones from the historical record can be simulated with the BAM. This also fits the BAM model to the data and will print out the fitted parameters.

`qsub bam_tracks.py`

Next, 10,000 years of TC tracks can be simulated by using a genesis distribution and repeating the 40 years of DLM flow 50 times:

```qsub simulate_tc_tracks.sh```

And once this is complete the results can be analyzed and plotted with:

```python analyse_tracks.py```


## Fitting BAM parameters

1. `extract_era5.sh` to extract required fields from ERA5 replication dataset on gadi (requires connection to project `rt52`)
2. `run_envflow.sh` to calculate environmental steering of TCs based on Galarneau & Davis (2013)
3. Optionally run `plot_env_flow.sh` to plot the environmental flow for each storm at time of maximum intensity
4. `fitBAMparameters.py` to fit the parameters to the intensity-conditioned BAM described by Lin et al. (2023)
5. `vorticity_analysis.py` to extract vorticity and gradients of vorticity from ERA5 at the location of storms
6. `plotBetaDrift.py` to calculate mean beta drift (residual between observed and weighted steering flow) on 2.5x2.5 degree grid