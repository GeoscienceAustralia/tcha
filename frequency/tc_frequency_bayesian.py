# # Estimating mean tropical cyclone frequency
#
# ## Source data
#
# There are multiple sources of tropical cyclone track data. Each Regional
# Specialized Meteorological Center (nominated by the World Meteorological
# Organisation) maintains it's own database of TC best-track information.
# There are also several other agencies that provide TC forecast information
# and publish best track datasets (e.g. Joint Typhoon Warning Center). This
# has been collated into the International Best Track Archive for Climate
# Stewardship (IBTrACS: [link]).
#
# Even within individual organisations, there may be multiple track datasets.
# In Australia, the primary TC database is [link]. But there is also an
# objective TC reanalysis dataset (Courtney et al. 2021). And through the
# various forecast policies, observing platforms and analysis techniques,
# there are discrepancies between these databases.
#
# ## Methods
#
# We use a Bayesian model for estimating the rate parameter $\lambda$ of a
# Poisson distribution. The Poisson distribution is often used to simulate the
# rate of TCs (and other natural hazard events), and defines the likelihood of
# $k$ events occurring in a given interval, with the rate parameter $\lambda$:
#
# $P(Z=k) = \frac{\lambda^{k} e^{-k}}{k!}, \mathrm{for  } k = 0, 1, 2,...$
#
# $\lambda$ conveniently is also the expected value of the Poisson
# distribution:
#
# $E[Z | \lambda ] = \lambda$.
#
# Three different models for the rate parameter are explored: $\lambda = c$
# (i.e. the rate parameter is constant),
# a linear trend model $\lambda = \alpha + \beta t$ and
# an exponential trend model $\lambda = \alpha e^{\mu t}$.
#
#
# ## Dependencies
#
#  The following modules are required to run this notebook
#
#  * pymc
#  * pandas
#  * xarray
#  * numpy
#  * matplotlib
#  * seaborn
#  * arviz
#  * jupyter
#
# ## Notes:
# The number of cores in `pm.sample()` must be specified on a Windows machine,
# due to the way python spawns processes for multiprocessing applications.
# Typically PyMC will spawn as many processes as there are cores available and
# the number of chains will be tied to that. But unless the sampling calls
# (`pm.sample()`) are made within a "if __name__ == '__main__':" clause, the
# interpreter will try to create a new process from scratch. This isn't a
# problem if the code is run in a Jupyter notebook, or on another platform.
# See https://discourse.pymc.io/t/error-during-run-sampling-method/2522/6

from os.path import join as pjoin
import pymc as pm
import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from datetime import datetime

mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['savefig.dpi'] = 600
locator = mdates.AutoDateLocator(minticks=10, maxticks=20)
formatter = mdates.ConciseDateFormatter(locator)

# We start with loading the TC track data. There are multiple possible sources
# of TC track data - IBTrACS relies on the Bureau of Meteorology's best track
# data, so in this case we go straight to that source. There are some issues
# with the quality of this data (Courtney et al., 2021), but the length of
# record here means we can test the sensitivity of our results to the number
# of years of data available.
#
# The season is assigned based on the starting year of the season. For example,
# a TC occuring in February 2010 will be recorded as occuring in the 2009
# season. The track data has been filtered to only count those storms that
# enter the Australian area of responsibility.
#
# We have calculated the seasonal frequency separately, and only need to load
# the data. The required files can be generated using tc_frequency.py

inputPath = r"..\data\frequency"
source = "http://www.bom.gov.au/clim_data/IDCKMSTM0S.csv"
outputPath = r"..\data\frequency"

dataFile = pjoin(inputPath, r"all_tcs.csv")
df = pd.read_csv(dataFile)
tccount = df[['IDSEAS', 'ID']].set_index("IDSEAS").squeeze()
tccount = tccount.loc[1981:]  # Only analysing data from 1981 onwards
years = (tccount.index - tccount.index[0]).values

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.bar(tccount.index, tccount.values, fc='0.75', ec='k')
sns.despine()
ax.set_xlabel("Season")
ax.set_ylabel("TC count")
plt.text(0.0, -0.1, f"Source: {source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
fig.tight_layout()
plt.savefig(pjoin(outputPath, "seasonal_frequency.png"), bbox_inches='tight')

# We start with a very simple analysis - evaluating the mean frequency of TCs.
# TC occurrence is assumed to be a Poisson process, so we use a Poisson
# distribution for the MCMC sampling with the prior for the rate $\lambda$
# chosen as a normal distribution.

with pm.Model() as mmodel:
    lambda_ = pm.Normal("lambda", mu=tccount.mean(), sigma=np.std(tccount))
    observation = pm.Poisson("obs", lambda_, observed=tccount.values)
    step = pm.Metropolis()
    mtrace = pm.sample(20000, tune=10000, step=step, return_inferencedata=True, chains=4, cores=1)
    mtrace.extend(pm.sample_posterior_predictive(mtrace))

# The median rate from this process is 10.6 and the 90% credible interval is
# [9.8, 11.5]. This should be compared to the sample mean (10.6) and standard
# deviation (3.3). The credible interval is much narrower than the standard
# errors based on the standard deviation (1.96 * s.d.)

mtrace.posterior['ymodel'] = mtrace.posterior['lambda'] * \
    xr.DataArray(np.ones(len(years)))

_, ax = plt.subplots(figsize=(12, 6))
az.plot_lm(idata=mtrace, y=tccount, x=years, num_samples=100,
           axes=ax, y_model='ymodel',
           kind_pp="hdi", kind_model='hdi',
           y_model_mean_kwargs={"lw": 2, "color": 'g', 'ls': '--'},
           y_kwargs={'marker': None, 'color': '0.75', 'label': '_obs'},
           y_model_fill_kwargs={'alpha': 0.25},
           y_hat_fill_kwargs={"hdi_prob": 0.95})
ax.bar(years, tccount.values, fc='0.9', ec='k', zorder=0)
ax.legend(loc=1)
ax.set_xticks(np.arange(-1, 41, 5))
ax.set_xticklabels(np.arange(1980, 2021, 5))
ax.set_xlabel("Season")
ax.set_ylabel("TC count")
meanrate = mtrace.posterior["lambda"].mean()
stdrate = mtrace.posterior["lambda"].std()
ax.set_title(rf"Mean rate $\lambda$: {meanrate:.1f} $\pm$ {stdrate:.2f}")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
sns.despine()
plt.text(0.0, -0.1, f"Source: {source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(
    outputPath, "seasonal_frequency_posterior_predictive_mean.png"),
    bbox_inches='tight')

axes = az.plot_trace(mtrace, compact=True, var_names=(
    'lambda'), legend=True, divergences='top',)
mq = np.quantile(mtrace.posterior['lambda'], [0.05, 0.5, 0.95])
axes[0, 0].axvline(mq[1], ls='--', color='k',
                   label=rf'$\lambda = {{{mq[1]:.1f}}}$')
axes[0, 0].axvline(mq[0], ls='--', color='gray',
                   label=f"90% CI [{mq[0]:.1f}, {mq[2]:.1f}]")
axes[0, 0].axvline(mq[2], ls='--', color='gray')
axes[0, 0].legend(fontsize='x-small')

axes[0, 1].axhline(mq[0], ls='--', lw=1, color='gray',)
axes[0, 1].axhline(mq[2], ls='--', lw=1, color='gray',)
axes[0, 1].axhline(mq[1], ls='--', lw=1, color='k',)

axes[0, 0].set_title(r"$\lambda$")
axes[0, 1].set_title(r"$\lambda$")
plt.tight_layout()
plt.savefig(pjoin(outputPath, "mean_posterior_trace.png"),
            bbox_inches='tight')

# We next consider a linear trend in TC rates. There is evidence that the rate
# of TC occurrence in the Australian region is declining. For example,
# Callaghan and Power (2011) reported a decline in the rate of severe TCs
# (category 3-5) making landfall along the Queensland coastline since the
# beginnning of last century. Here we need to estimate the trend and intercept.
# To make the intercept sensible, we use the year from the start of the record
# as the first year (i.e. $t_0=1981$).
#
# $\lambda = \alpha + \beta t$
#
# The priors chosen for $\alpha$ and $\beta$ are predicated on the rate having
# no trend - we choose normally distributed priors with the mean equal to the
# observed mean rate of TC occurrence for $\alpha$ and zero mean for the
# $\beta$ coefficient.

with pm.Model() as lmodel:
    alpha = pm.Normal('alpha', mu=tccount.mean(), sigma=np.std(tccount))
    beta = pm.Normal('beta', mu=0, sigma=1.)
    lambda_ = alpha + beta * years
    observation = pm.Poisson("obs", lambda_, observed=tccount.values)
    step = pm.Metropolis()
    ltrace = pm.sample(20000, tune=10000, step=step, return_inferencedata=True, chains=4, cores=1)
    ltrace.extend(pm.sample_posterior_predictive(ltrace))

# Plot the trace of the alpha and beta parameters, including 90th percentile
# credible interval lines

axes = az.plot_trace(ltrace, compact=True, var_names=(
    'alpha', 'beta'), legend=True, divergences='top',)
aq = np.quantile(ltrace.posterior['alpha'], [0.05, 0.5, 0.95])
bq = np.quantile(ltrace.posterior['beta'], [0.05, 0.5, 0.95])
axes[0, 0].axvline(aq[1], ls='--', color='k',
                   label=rf'$\alpha = {{{aq[1]:.1f}}}$')
axes[0, 0].axvline(aq[0], ls='--', color='gray',
                   label=f"90% CI [{aq[0]:.1f}, {aq[2]:.1f}]")
axes[0, 0].axvline(aq[2], ls='--', color='gray')

axes[1, 0].axvline(bq[1], ls='--', color='k',
                   label=rf'$\beta = {{{bq[1]:.3f}}}$')
axes[1, 0].axvline(bq[0], ls='--', color='gray',
                   label=f"90% CI [{bq[0]:.3f}, {bq[2]:.3f}]")
axes[1, 0].axvline(bq[2], ls='--', color='gray')
axes[0, 0].legend(fontsize='x-small')
axes[1, 0].legend(fontsize='x-small')

axes[0, 1].axhline(aq[0], ls='--', lw=1, color='gray',)
axes[0, 1].axhline(aq[2], ls='--', lw=1, color='gray',)
axes[0, 1].axhline(aq[1], ls='--', lw=1, color='k',)

axes[1, 1].axhline(bq[0], ls='--', lw=1, color='gray',)
axes[1, 1].axhline(bq[2], ls='--', lw=1, color='gray',)
axes[1, 1].axhline(bq[1], ls='--', lw=1, color='k',)

axes[0, 0].set_title(r"$\alpha$")
axes[0, 1].set_title(r"$\alpha$")
axes[1, 0].set_title(r"$\beta$")
axes[1, 1].set_title(r"$\beta$")
plt.tight_layout()
plt.savefig(pjoin(outputPath, "linear_posterior_trace.png"),
            bbox_inches='tight')

ltrace.posterior['ymodel'] = ltrace.posterior['alpha'] + \
    ltrace.posterior['beta'] * xr.DataArray(years)

_, ax = plt.subplots(figsize=(12, 6))
ax.bar(years, tccount.values, fc='0.9', ec='k')

az.plot_lm(idata=ltrace, y=tccount, x=years, num_samples=100, axes=ax,
           y_model='ymodel', kind_pp="hdi",
           kind_model='hdi',
           y_model_mean_kwargs={"lw": 2, "color": 'g', 'ls': '--'},
           y_kwargs={'marker': None, 'color': '0.75', 'label': '_obs'},
           y_model_fill_kwargs={'alpha': 0.25},
           y_hat_fill_kwargs={'color':'orange',
                              "hdi_prob": 0.95})

ax.legend(loc=1)
ax.set_xlabel("Season")
ax.set_ylabel("TC count")
ax.set_title("Linear trend model")
ax.set_xticks(np.arange(-1, 41, 5))
ax.set_xticklabels(np.arange(1980, 2021, 5))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.text(0.0, -0.1, f"Source: {source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left',)
plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right')
plt.savefig(pjoin(
    outputPath, "seasonal_frequency_linear_posterior_predictive.png"), bbox_inches='tight')

# The issue is here that the trend is negative (in this case approximately
# -0.07 TCs/year/year). This implies that at some point in the future, the TC
# count will be negative, which is not plausible. The probability of no trend
# is quite small - it's only just within the 90% credible range. Now there are
# a range of issues with this assumption (that the trend will not change), but
# we can explore other options and evaluate which model is most suitable given
# the available data.
#
# We next try an exponential function for the rate parameter. Again, $t$ is
# taken as $t_0 = 1981$.
#
# $\lambda = \alpha \exp(\mu t)$
#
# In this case, the decay rate ($\mu$) is negative (median = -0.0069, 90%
# credible range -0.013 -- -0.0001). The (admittedly small) advantage is that
# the rate parameter $\lambda$ should not become negative, rather just
# approaching zero some time off into the future -- but only if the decay rate
# $\mu < 0$, which we cannot exclude given the 90% CI for $\mu$ only just
# excludes 0.
#
# In both these models, we cannot reject the null hypothesis of no trend in
# the Poison rate parameter $\lambda$. Both provide strong evidence that the
# trend (over the period 1981-2020) is negative, but it does not provide
# evidence to reject the zero trend. Looking at different time spans will
# lead to different conclusions, since the underlying data will be different
#  reflecting the evolution of observing practices.
#
# However, we can use these models to simulate a large number of samples of TC
# rates that are representative of the observed rate. Sampling from the
# posterior predictive distributions, we can create as many years of samples
# as we need, which still retain any trend information.

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=tccount.mean(), sigma=np.std(tccount))
    mu = pm.Normal("mu", sigma=0.1)
    lambda_ = alpha * np.exp(mu * years)
    observation = pm.Poisson("obs", lambda_, observed=tccount.values)

    step = pm.Metropolis()
    etrace = pm.sample(20000, tune=10000, step=step, return_inferencedata=True, chains=4, cores=1)
    etrace.extend(pm.sample_posterior_predictive(etrace))


# Plot the trace of the alpha and mu parameters, including 90th percentile
# credible interval lines

axes = az.plot_trace(etrace, compact=True, var_names=(
    'alpha', 'mu'), legend=True, divergences='top',)
aq = np.quantile(etrace.posterior['alpha'], [0.05, 0.5, 0.95])
mq = np.quantile(etrace.posterior['mu'], [0.05, 0.5, 0.95])
axes[0, 0].axvline(aq[1], ls='--', color='k',
                   label=rf'$\alpha = {{{aq[1]:.1f}}}$')
axes[0, 0].axvline(aq[0], ls='--', color='gray',
                   label=f"90% CI [{aq[0]:.1f}, {aq[2]:.1f}]")
axes[0, 0].axvline(aq[2], ls='--', color='gray')

axes[1, 0].axvline(mq[1], ls='--', color='k',
                   label=rf'$\mu = {{{mq[1]:.3f}}}$')
axes[1, 0].axvline(mq[0], ls='--', color='gray',
                   label=f"90% CI [{mq[0]:.3f}, {mq[2]:.3f}]")
axes[1, 0].axvline(mq[2], ls='--', color='gray')
axes[0, 0].legend(fontsize='x-small')
axes[1, 0].legend(fontsize='x-small')

axes[0, 1].axhline(aq[0], ls='--', lw=1, color='gray',)
axes[0, 1].axhline(aq[2], ls='--', lw=1, color='gray',)
axes[0, 1].axhline(aq[1], ls='--', lw=1, color='k',)

axes[1, 1].axhline(mq[0], ls='--', lw=1, color='gray',)
axes[1, 1].axhline(mq[2], ls='--', lw=1, color='gray',)
axes[1, 1].axhline(mq[1], ls='--', lw=1, color='k',)

axes[0, 0].set_title(r"$\alpha$")
axes[0, 1].set_title(r"$\alpha$")
axes[1, 0].set_title(r"$\mu$")
axes[1, 1].set_title(r"$\mu$")
plt.tight_layout()
plt.savefig(pjoin(outputPath, "exponential_posterior_trace.png"),
            bbox_inches='tight')

# Here we plot the linear(ish) model of the Poisson rate parameter, along with
# the credible interval of rates sampled from the posterior distribution. We
# can add in predictive samples to demonstrate the range of plausible annual
# TC rates that are consistent with the observed record. In the figure below,
# the green dashed line represents the rate parameter $\lambda$ estimated from
# the median result of the posterior distributions, with the corresponding
# credible interval in grey. The orange band is the credible range, based on
# posterior predictive samples. The black bars are the observed TC counts.

etrace.posterior['ymodel'] = etrace.posterior['alpha'] * \
    np.exp(etrace.posterior['mu'] * xr.DataArray(years))
_, ax = plt.subplots(figsize=(12, 6))
az.plot_lm(idata=etrace, y=tccount, x=years, num_samples=100,
           axes=ax, y_model='ymodel',
           kind_pp="hdi", kind_model='hdi',
           y_model_mean_kwargs={"lw": 2, "color": 'g', 'ls': '--'},
           y_kwargs={'marker': None, 'color': '0.75', 'label': '_obs'},
           y_model_fill_kwargs={'alpha': 0.25})
ax.bar(years, tccount.values, fc='0.9', ec='k', zorder=0)
ax.set_xlabel('Season')
ax.set_ylabel('TC count')
ax.set_xticks(np.arange(-1, 41, 5))
ax.set_xticklabels(np.arange(1980, 2021, 5))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title("Exponential trend model")
ax.legend(loc=1)

plt.text(-0.05, -0.1, f"Source:\n{source}",
         transform=ax.transAxes, fontsize='xx-small', ha='left', va='top')
plt.text(1.0, -0.1, f"Created: \n{datetime.now():%Y-%m-%d %H:%M}",
         transform=ax.transAxes, fontsize='xx-small', ha='right', va='top')
plt.savefig(pjoin(
    outputPath, "seasonal_frequency_exponential_posterior_predictive.png"),
    bbox_inches='tight')

# Save a set of samples to a csv file for later use in sampling
# TC seasonal numbers:
mtrace.posterior_predictive['obs'][0].to_dataframe().to_csv(
    pjoin(outputPath, "tccount.samples.csv")
    )

fig, ax = plt.subplots(1, 1)
ax.hist(ltrace.posterior_predictive['obs'].values.flatten(),
        bins=np.arange(0, 26, 1), density=True,
        alpha=0.75, width=0.75, label="Sampled")
ax.hist(tccount.values, bins=np.arange(0, 26, 1), density=True,
        ec='k', fc='white', alpha=0.5, width=0.5, label="Observed")
ax.set_xlabel("Seasonal TC count")
ax.legend()
ax.text(-0.1, -0.1, f"Source: {source}",
        transform=ax.transAxes, fontsize='xx-small', ha='left',)
ax.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M}",
        transform=ax.transAxes, fontsize='xx-small', ha='right')
fig.tight_layout()
plt.savefig(pjoin(
    outputPath, "seasonal_frequency_linear_posterior_sample.png"), bbox_inches='tight')

# Finally, to determine the most suitable model, we use leave-one-out (LOO)
# cross-validation. For our current model choices, the results are almost
# indistiguishable.
print(f"Mean: {az.loo(mtrace).loo:.2f}")
print(f"Linear: {az.loo(ltrace).loo:.2f}")
print(f"Exponential: {az.loo(etrace).loo:.2f}")
