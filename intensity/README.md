## Running instructions

### Data extraction

First run the data extraction scripts:
```
qsub temp_extract.sh
qsub rh_extract.sh
qsub bran_extract.sh
qsub era5_extract.sh
python wind_extract.sh
```

### Running intensity

Compile the code:

`f2py3 -m hurr -c hurr.f`

Next run `intensity_analysis.py`, being careful to change the filepaths.

