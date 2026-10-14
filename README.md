# Beyond sectors: stochastic block models of US stock co-movement

Which groups of large US stocks move together once the market is removed, how those groups changed in the 2020 crash and the 2022 rate rise, and whether they help build lower-risk portfolios. Daily returns of 250 S&P 500 stocks, 2010–2026.

## Summary

Investors usually group stocks by industry labels such as GICS sectors. This project asks what groups the data themselves suggest. It measures how closely 250 large US stocks move together after removing the movement of the market as a whole, and fits a **stochastic block model** (SBM) to the resulting network with [graph-tool](https://graph-tool.skewed.de/).

An SBM is a statistical model of how the network could have arisen: each stock belongs to a group, and how closely two stocks move together depends only on their groups. Unlike most clustering methods, it does not assume that groups are tight communities. A group can also be defined by how it relates to the rest of the market. Because it is fitted in a Bayesian way, it weighs fit against complexity, so it does not report groups that the data cannot support and it chooses the number of groups itself.

Four questions, with short answers:

1. **What groups does the model find?** A few dozen groups that refine sectors to roughly the sub-industry level: utilities, regional banks, oil producers, defence primes, semiconductor equipment and so on.
2. **How are the groups related?** Almost all pairs of groups behave as communities: each group is tighter internally than with the other. The exceptions that matter are in the utilities, banks and real-estate complexes, where a looser group is tied more closely to a tighter neighbour than to itself (a *core–periphery* pair).
3. **How did the groups change in the 2020 crash and the 2022 rate rise?** COVID regrouped stocks *across* sectors, by their exposure to lockdowns and reopening; the share of a stock's group-mates from other sectors rose from 39% in 2019 to 63% in 2020. The 2022 rate rise did not: groups stayed within sectors (47%), and energy became one tight group. The old structure has only partly returned.
4. **Do the groups help build lower-risk portfolios?** A little. Shrinking the covariance matrix towards the groups gives about 4% lower out-of-sample volatility than standard constant-correlation shrinkage, and about 2% lower than shrinking towards sectors. It is a modest gain, not a reliable edge.

A longer paper-style write-up is available separately. The toolkit comes from my postdoctoral work fitting SBMs to ecological food webs.

![Correlations between all 250 stocks after removing the market, ordered by the model's groups, in 2019, 2020 and 2022](figures/posterior_heatmaps_shocks.png)

*Each row and column is one stock, in the same order on both axes; red cells are pairs that moved together beyond the market, blue cells pairs that moved oppositely. Each group is a square on the diagonal. The strips give each stock's sector: long single-colour stretches are one-sector groups. In 2019 the upper half of the strip is mostly such stretches; in 2020 sectors mix; in 2022 the stretches return.*

## Data

Daily adjusted closes from Yahoo Finance (via [`yfinance`](https://github.com/ranaroussi/yfinance)), 4 January 2010 to 30 September 2026, with sectors and sub-industries from the current Wikipedia list of S&P 500 companies. **No market data is included in this repository**; `scripts/01_download.py` downloads it to a local cache (`data/`, gitignored).

The universe is the 250 most liquid stocks, ranked on their first year only, among firms that were already in the index on 1 January 2010 and have a complete price history (`src/data.py`). This avoids picking firms that later joined the index *because* they grew, but it does not remove survivorship bias.

## Method

* **Removing the market.** Raw correlations are dominated by the market: everything moves with it. For each stock and each window, the daily return is regressed on the S&P 500 ETF (SPY) and only the residual is kept (`src/preprocess.py`). Removing the first principal component instead gives similar groups.
* **The network.** The residual correlations form a complete weighted network: every pair of stocks is linked, with the correlation as the link's weight. Nothing is thresholded (`src/network.py`).
* **The model.** A nested, weighted SBM, which groups the stocks and then groups the groups (`src/sbm.py`). It was chosen over 12 other versions, including a version that only allows communities, by minimum description length: the model that describes the data most compactly wins. This choice held on every test window and every way of handling the market.
* **Fitting.** The model has many good solutions of almost equal quality, and a single fit ends in one of them. Each calendar year is therefore fitted 30 times from independent starts, and each quarterly window 8 times (`scripts/08_basin_search.py`, `scripts/12_rolling_basins.py`). The results describe the best fit of each window, with ranges across fits where they matter.
* **How pairs of groups relate.** Following [Liu, Alexander & Altmann (2023)](https://doi.org/10.1093/pnasnexus/pgad364), each pair of groups is classified by comparing the average correlation inside each group with the average correlation between them: a **community** pair if the between-group correlation is the lowest, **core–periphery** if it lies between the two within-group values, and **disassortative** if it is the highest (`src/interactions.py`).
* **Windows.** The calendar years 2019–2024, the two calm years 2023–24 together, and rolling one-year windows refitted every quarter from 2011 to 2026.
* **Portfolio test.** Every quarter, a minimum-variance portfolio is built from the trailing year of returns, with Ledoit–Wolf shrinkage of the covariance matrix towards a target built from the model's groups (`src/covariance.py`, tested in `tests/test_covariance.py`). Its realised volatility over 2011–2026 is compared with shrinkage towards constant correlation and towards sectors, with 95% block-bootstrap intervals.

## Results

**The groups refine sectors.** The model finds a few dozen groups, much closer to GICS sub-industries than to sectors. Many are clean industry groups, such as regional banks, utilities, oil producers or defence contractors. Others join industries that share an exposure, such as rate-sensitive utilities and real-estate trusts.

**Almost all groups are communities.** For almost every pair of groups, each group is tighter internally than with the other. The few exceptions are among utilities, banks and real estate, where a looser group is tied more closely to a tighter neighbour than to itself.

**COVID mixed sectors; the 2022 rate rise did not.** In 2020 the groups lined up with sectors far less than in any other year. Stocks regrouped by their exposure to the pandemic: one group held travel and reopening stocks from five sectors, another held stay-at-home stocks. The share of a stock's group-mates from other sectors rose from 39% in 2019 to 63% in 2020. In 2022 it was 47%, close to normal; the change came within sectors instead, and energy became one tight group. The old structure has only partly returned since.

![Agreement with sectors and sub-industries, share of core–periphery pairs and number of groups, 2019–2024; each dot is one fit](figures/posterior_window_comparison.png)

**The results do not depend on the fit.** Repeated fits of the same period give very similar groupings, and every result above holds for all of them. Refitting every quarter from 2011 shows the same break when the 2020 crash enters the window.

**The groups help portfolios a little.** Building minimum-variance portfolios with a covariance estimate shrunk towards the groups gives about 4% lower out-of-sample volatility than standard constant-correlation shrinkage, and about 2% lower than shrinking towards sectors. Groups from quicker, rougher fits beat constant correlation too, but not sectors. It is a modest gain, not a reliable edge.

---

AI assistance was used in this writeup.
