import pandas as pd 
import numpy as np
from scipy import stats

def main():
    data = pd.read_csv("resultsq0.5.csv")

    mu1, mu2, mu3 = np.mean(data[['Linear Greedy Val']]), np.mean(data[['Quadratic Greedy Val']]), np.mean(data[['Projected Subgradient Val']])
    print(mu1, mu2, mu3)

    # Filter out rows with any negative values in the relevant columns
    negatives = data[
        (data['Projected Subgradient Val'] < 0) |
        (data['Quadratic Greedy Val'] < 0) |
        (data['Linear Greedy Val'] < 0)
    ].reset_index(drop=True)

    negative_vals = negatives[['Linear Greedy Val', 'Quadratic Greedy Val', 'Projected Subgradient Val']]
    n = len(negatives)

    print("Filtered negative values:")
    print(negative_vals.head())

    linear = negative_vals['Linear Greedy Val']
    quadratic = negative_vals['Quadratic Greedy Val']
    projected = negative_vals['Projected Subgradient Val']

    lmu, qmu, pmu = np.mean(linear), np.mean(quadratic), np.mean(projected)
    print(f"\nMeans:\nLinear Greedy: {lmu:.6f}, Quadratic Greedy: {qmu:.6f}, Projected: {pmu:.6f}")

    # Perform paired t-test
    t_stat, p_two_sided = stats.ttest_rel(linear, quadratic)

    # Convert to one-sided p-value (for "Linear > Quadratic")
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - p_two_sided / 2

    print(f"\nPaired t-test:\nt-statistic = {t_stat:.4f}, one-sided p-value = {p_one_sided:.4f}")

    # Interpretation
    alpha = 0.05
    if p_one_sided < alpha:
        print("Reject H0: Linear Greedy has a significantly higher mean than Quadratic Greedy.")
    else:
        print("Fail to reject H0: No significant difference in means.")

if __name__ == "__main__":
    main()
