Tropical Cyclone Hazard Assessment
++++++++++++++++++++++++++++++++++

Evaluating the likelihood and magnitude of tropical cyclone winds based on a
stochastic TC model (TCRM: https://github.com/geoscienceaustralia/tcrm).



Analysis of observations
------------------------

Scripts to analyse the observational records of TCs and automatic weather
station observations


TC frequency
~~~~~~~~~~~~

frequency/tc_frequency.py - calculates mean frequency and trends for a range of
TC datasets and time periods of those datasets.
frequency/jtwc_frequency.py - Uses JTWC data to evaluate frequency
frequency/frequency_distribution.py - fits a negative binomial distribution to
annual frequency, for consideration as the source model for TCRM. Negative
binomial initially selected over poisson distribution, as the distribution is
very slightly overdispersed ([mu / sigma] < 1).
frequency/tc_frequency_bayesian.py - use Bayesian MCMC methods to fit Poisson
distribution to TC frequency, and generate posterior samples that can be used
for sampling annual TC counts.


Track density
~~~~~~~~~~~~~

density/track_density.py - calculates TC frequency on a grid, counting the
number of unique events intersecting each grid point. Currently uses the BoM
best track dataset (IDCKMSTM0S.csv) as input, and a 0.5x0.5 degree grid over the
simulation domain.

Compares 1981-2020 and 1951-2020 periods.

Uses jackknife (leave-one-out) bootstrap resampling to evaluate mean track
density, by iteratively excluding seasons from the dataset for calculating track
density.

To run::

    ``python density/track_density.py``


TC landfall rates
~~~~~~~~~~~~~~~~~


Lifetime maximum intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~

lmi/extractLMI.py
lmi/extractLMI_IDCKMSTM0S.py


Potential intensity analysis
----------------------------

Using the theory of potential intensity to guide estimation of simulated TC
intensity.


Basin-wide trends
~~~~~~~~~~~~~~~~~

Monthly trends
~~~~~~~~~~~~~~


Potential intensity from climate models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Deep layer mean winds
---------------------
dlm-climatology/climatology.py -

TC-related rainfall
-------------------
precip/extract_precip.py - extracts ERA5 precipitation within a defined distance
of the cyclone centre.

Contact:
--------

Craig Arthur
craig.arthur@ga.gov.au
Last updated: 2023-07-20