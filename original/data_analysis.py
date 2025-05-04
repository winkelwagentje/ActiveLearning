import pandas as pd 


def main():
    data = pd.read_csv("resultsq0.5.csv")
    #  | (data['Quadratic Greedy Val'] < 0) | (data['Linear Greedy Val'] < 0)
    negatives = data[(data['Projected Subgradient Val'] < 0)].reset_index(drop=True)
    negative_vals = negatives[['Linear Greedy Val', 'Quadratic Greedy Val', 'Projected Subgradient Val']]
    n = len(negatives)
    print(negative_vals)


if __name__ == "__main__":
    main()