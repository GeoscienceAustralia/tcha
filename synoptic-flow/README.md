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