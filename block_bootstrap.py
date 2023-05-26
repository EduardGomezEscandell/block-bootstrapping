#!/bin/python3
"""
BLOCK BOOTSRAPPING

This program is used to simulate future returns of risky assets.

The method used is block bootstrapping. The goal is to use historical returns
to predict future ones. Efron (1979) developped a method wherein random samples
are taken from the historical dataset and concatenated to generate a
pseudohistory. By generating a large number of pseudo-histories, we can generate
a probability distribution for the returns of the asset. This method is called
the Standard Bootstrapping.

However, Efron's method assumes asset prices are independent and identically
distributed (IID) random variables. This is known to be false: asset pricing
is autocorrelated, meaning that an asset's price depends on its history,
particularly in the short term.

In order to capture this effect, we concatenate blocks of historical data to
create the pseudo-history. We also generate a large number of pseudohistories
and study their aggregate behaviour. This methodology is named Block
Bootstrapping.

The size of each block is chosen randomly from a geometrical distribution, and
the start of the block is chosen from a uniform distribution. If a block were
to stretch beyond the end of history, it is wrapped around to the beggining;
hence treating the historical dataset like a circular array.

Replacement is allowed, meaning blocks may overlap. The mean length of a block
is a tuned parameter with no consensus on optimality.

References
1. Cogneau, P., & Zakamouline, V. (2010). "Bootstrap methods for finance:
   Review and analysis".

2. Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife". Annals
   of Statistics, 7 (1), 1-26

3. Politis, D. and Romano, J. (1994). "The Stationary Bootstrap", Journal of
   the American Statistical Association, 89 (428), 1303-1313.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from typing import Generator, Tuple
import os

# DATA ENTRY
# Number of pseudohistories to generate
sample_size = 100_000

# Duration of each simulation (in months)
duration = 12 * 5

# Mean duration of the bootstrapping blocks (in months)
block_mean_duration = 12 * 2

# Path to the historical data file
file_name = "data/MSCI World (1970-2023).csv"


# PREPROCESSING
# Read data
df = pd.read_csv(file_name, sep="\t", comment="#")
history = np.log(1 + (df["Returns (%)"].to_numpy()) / 100)

# Allocate result arrays
mean = np.zeros(shape=(duration,))
returns = np.empty(shape=(sample_size,))

# Start plots
plt.suptitle(f"Block bootstrap simulation of {os.path.splitext(os.path.basename(file_name))[0]}")
plt.subplot(211)


# COMPUTATION
def random_number_generator() -> Generator[Tuple[int, int], None, None]:
    "Generates random numbers in bulk"
    gen = np.random.default_rng()
    while True:
        geometric_cache = gen.geometric(1 / block_mean_duration, size=sample_size)
        uniform_cache = gen.integers(0, len(history), size=sample_size, dtype=np.int32)
        for i in zip(geometric_cache, uniform_cache):
            yield i


rng = random_number_generator()

# Loop foreach simulation
for k in range(sample_size):
    pseudohistory = np.empty(shape=(duration - 1,))
    t = 0
    # Loop foreach block
    while t < duration - 1:
        length, begin = next(rng)
        length = min(length, duration - 1 - t)
        if length == 0:
            continue

        # Standard case: append block to pseudohistory
        if begin + length < len(history):
            pseudohistory[t : t + length] = history[begin : begin + length]
            t += length
            continue

        # Special case: wrap-around
        L1 = len(history) - begin
        pseudohistory[t : t + L1] = history[begin:]
        t += L1

        L2 = length - L1
        pseudohistory[t : t + L2] = history[:L2]
        t += L2

    returns[k] = np.exp(np.sum(pseudohistory))

    # Plot run
    if k % int(sample_size**0.5) == 0:
        acc = np.empty(shape=(duration,))
        acc[0] = 1
        np.cumsum(pseudohistory, out=acc[1:])
        np.exp(acc[1:], out=acc[1:])
        label = "Example pseudo-histories" if k == 0 else "_"
        plt.semilogy(acc, color="#888888", alpha=0.3, label=label)

# POSTPROCESSING
plt.legend()
plt.grid()
plt.xlabel("Time (months)")
plt.ylabel("Returns")
plt.title(
    f"{sample_size} runs using overlapping blocks of geometrically distributed duration (μ={block_mean_duration/12:.2} years)",
    fontsize=8,
)

# Plot density function of annualized returns
plt.subplot(212)

annualized = returns ** (12 / duration)
bins = np.round(1000 * annualized, decimals=0).astype(int)  # Create bins of size 0.1%
to_percentage = lambda x: x / 10 - 100

x = to_percentage(np.arange(0, np.max(bins) + 1))
density = np.bincount(bins) / len(bins)

# Loss
loss = np.sum(density[:1000])
plt.bar(
    x[:1000],
    density[:1000],
    width=0.101,
    align="center",
    color="#ff0000",
    label=f"Loss probability: {loss*100:.2f}%",
)

nonzero = np.count_nonzero(density)
smoothed_density = scipy.signal.savgol_filter(density, int(5 * np.sqrt(nonzero)), 3)

# Gain
plt.bar(
    x[1000:],
    density[1000:],
    width=0.101,
    align="center",
    color="#119911",
    label=f"Gain probability: {(1-loss)*100:.2f}%",
)

plt.plot(x, smoothed_density, color="#000000")

# Cheating to print mean and std
mean = density @ x
moment2 = density @ (x - mean) ** 2
moment4 = density @ (x - mean) ** 4
median = x[np.searchsorted(np.cumsum(density), 0.5)]

std = np.sqrt(moment2)
skewness = 3 * (mean - median) / std
excess_kurtosis = moment4 / std**4 - 3

plt.plot([0], [0], label=f"Mean:  {mean:.2f}%", color="#ffffff")
plt.plot([0], [0], label=f"Standard deviation: {std:.2f} pp", color="#ffffff")
plt.plot([0], [0], label=f"Skewness:  {skewness:.2f}", color="#ffffff")
plt.plot([0], [0], label=f"Excess kurtosis: {excess_kurtosis:.2f}", color="#ffffff")

plt.legend()
plt.xlabel("Annualized returns (%)")
plt.ylabel("Probability density")

plt.xlim(to_percentage(np.min(bins)), to_percentage(np.max(bins) + 1))

plt.show()
