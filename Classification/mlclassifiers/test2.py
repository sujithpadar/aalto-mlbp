# identify the split criteria
    if splitcriteria == "entropy":
        splitclass = self.entropy()

# idenitfy the target feature type
# if less than 1% of the data is unique or there are less than 10 distinct values in the array, make the task as classification
    if np.unique(y).shape[0]/(y.shape[0]) <= 0.001 or np.unique(y).shape[0] <= 10:
        task = "classification"
    else:
        "regression"
