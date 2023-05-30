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
from typing import Generator, Tuple, List
import concurrent.futures as futures

# DATA ENTRY
def main():
    # Number of pseudohistories to generate
    sample_size = 1_000_000

    # Duration of each simulation (in years)
    horizon = 30

    # Mean duration of the bootstrapping blocks (in years)
    block_mean_duration = 3

    # Path to the historical data file
    file_name = "data/S&P500 (1900-2023).csv"

    bb = BlockBoostrapper(file_name)
    bb.compute(sample_size, horizon, block_mean_duration)
    bb.post_process()

class BlockBoostrapper:
    def __init__(self, sourceFile: str) -> None:
        df = pd.read_csv(sourceFile, sep="\t", comment="#")
        self.history = np.log(1 + (df["Returns (%)"].to_numpy()) / 100)
        self.snapshots = None

    def compute(self, npseudhistories: int, horizon: int, block_mean_length: int) -> None:
        workers = 4
        m = int(np.ceil(npseudhistories/workers))
        n = m*workers

        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            fut: List[futures.Future[np.ndarray]] = []
            for i in range(workers):
                f = executor.submit(self._compute, m, horizon, block_mean_length)
                fut.append(f)

            self.snapshots = np.empty(shape=(n, horizon+1))
            for i, f in enumerate(fut):
                self.snapshots[m*i:m*i+m] = f.result()
    
    def _compute(self, npseudhistories: int, horizon: int, block_mean_length: int) -> np.ndarray:
        """
        Compute [npseudohistories] runs with a length of [horizon] years with a mean block length of [block_mean_length] years.
        """
        snapshots = np.empty(shape=(npseudhistories, horizon+1))
        snapshots[:,0] = 1

        for k in range(npseudhistories):
            pseudohistory = self._pseudohistory(horizon, block_mean_length)
            snapshots[k, 1:] = np.exp(np.cumsum(pseudohistory))[11::12]
        return snapshots

    def _pseudohistory(self, horizon: int, block_mean_length: int):
        rng = self.__random_number_generator(block_mean_length*12, horizon)
        pseudohistory = np.empty(shape=(12*horizon,))
        t = 0
        while t < 12*horizon:
            length, begin = next(rng)
            length = min(length, horizon*12 - t)
            if length == 0:
                continue

            # Standard case: append block to pseudohistory
            if begin + length < len(self.history):
                pseudohistory[t : t + length] = self.history[begin : begin + length]
                t += length
                continue

            # Special case: wrap-around
            L1 = len(self.history) - begin
            pseudohistory[t : t + L1] = self.history[begin:]
            t += L1

            L2 = length - L1
            pseudohistory[t : t + L2] = self.history[:L2]
            t += L2
        return pseudohistory

    def post_process(self):
        if self.snapshots is None:
            raise RuntimeError("called post_process before compute")

        with futures.ProcessPoolExecutor(max_workers=4) as executor:
            fut: List[futures.Future[None]] = []
            for year, snapshot in enumerate(self.snapshots.T):
                f = executor.submit(self._pp_snapshot, snapshot, year)
                fut.append(f)
            map(lambda f: f.result(), fut)
            
    def _pp_snapshot(self, snapshot: np.ndarray, year: int) -> None:
            annualized = snapshot ** (1 / year) if year != 0 else snapshot
            bins = np.round(1000 * annualized, decimals=0).astype(int)  # Create bins of size 0.1%
            to_percentage = lambda x: x / 10 - 100

            x = to_percentage(np.arange(0, np.max(bins) + 1))
            density = np.bincount(bins) / len(bins)

            plt.figure(figsize=[8,5])
            
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
            gain = np.sum(density[1001:])
            plt.bar(
                x[1000:],
                density[1000:],
                width=0.101,
                align="center",
                color="#119911",
                label=f"Gain probability: {gain*100:.2f}%",
            )

            plt.plot(x, smoothed_density, color="#000000")

            # Cheating to print mean and std
            ann = 100*(annualized - 1)
            mean = np.mean(ann)
            std = np.std(ann)
            if abs(std) > 1e-15:
                skewness = scipy.stats.skew(ann)
            #     excess_kurtosis = scipy.stats.kurtosis(ann)
            else:
                skewness = 0
            #     excess_kurtosis = -3

            plt.plot([0], [0], label=f"Mean:  {mean:.2f}%", color="#ffffff")
            plt.plot([0], [0], label=f"Standard deviation: {std:.2f} pp", color="#ffffff")
            plt.plot([0], [0], label=f"Skewness:  {skewness:.2f}", color="#ffffff")
            # plt.plot([0], [0], label=f"Excess kurtosis: {excess_kurtosis:.2f}", color="#ffffff")

            plt.xlim(-10, 25)
            plt.ylim(0.0, 0.015)

            plt.legend(loc="upper left")
            plt.xlabel("Annualized returns (%)")
            plt.ylabel("Probability density")
            plt.title(f"Year {year} results")
            
            plt.savefig(f'results/frame_{year}.png')
            plt.close()
            print(f"Year {year} done")
    
    def __random_number_generator(self, block_duration: int, cache_size: int) -> Generator[Tuple[int, int], None, None]:
        "Generates random numbers in bulk"
        gen = np.random.default_rng()
        while True:
            geometric_cache = gen.geometric(1 / block_duration, size=cache_size)
            uniform_cache = gen.integers(0, len(self.history), size=cache_size, dtype=np.int32)
            for i in zip(geometric_cache, uniform_cache):
                yield i


if __name__ == '__main__':
    main()