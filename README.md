## Asset price simulation

This program is used to simulate future returns of risky assets. The method
used is bootstrapping. The goal is to use historical returns to predict future
ones.

### Standard bootstrapping
Efron (1979) developed a method wherein random samples are taken from the
historical dataset and concatenated to generate a pseudohistory. By generating
a large number of pseudo-histories, we can generate a probability distribution
for the returns of the asset.

However, Efron's method assumes asset prices are independent and identically
distributed (IID) random variables. This is known to be false: asset pricing
is autocorrelated, meaning that an asset's price depends on its history,
particularly in the short term.

### Block bootstrapping
To capture this effect, we concatenate blocks of historical data to create the
pseudo-history. We also generate a large number of pseudohistories and study
their aggregate behavior. The methodology implemented was developed by Politis
and Romano (1994).

The size of each block is chosen randomly from a geometrical distribution, and
the start of the block is chosen from a uniform distribution. If a block were
to stretch beyond the end of history, it is wrapped around to the beginning;
essentially treating the historical dataset like a circular array.

Replacement is allowed, meaning blocks may overlap. The mean length of a block
is a tuned parameter with no consensus on optimality.

### Usage
You need a file containing monthly historical returns from the asset to study.
The file must contain tab-separated values, sorted from old to recent. The
returns must be in a column named "Returns (%)". Then, edit the fields in the
data entry section in `block_bootstrap.py`. Once done, execute the file with
Python.

### Example run
This run samples the history of the S&P 500 index from 1900 to 2023 to simulate
1 million pseudohistories. It shows the distribution of returns year after year
for 30 years, with a piecewise 3rd-order polynomial fitting.

Despite high skewness and kurtosis (fat-tails) in month-to-month stock returns,
we see that after 20 years the distribution is highly normal.

![image](https://github.com/EduardGomezEscandell/block-bootstrapping/assets/47142856/61e98ca2-2563-4b51-b5ba-76ca006cf9ad)

### References
1. Cogneau, P., & Zakamouline, V. (2010). "Bootstrap methods for finance:
   Review and analysis".

2. Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife". Annals
   of Statistics, 7 (1), 1-26

3. Politis, D. and Romano, J. (1994). "The Stationary Bootstrap", Journal of
   the American Statistical Association, 89 (428), 1303-1313.
