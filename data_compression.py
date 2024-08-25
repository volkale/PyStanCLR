import numpy as np
from patsy import dmatrices


def get_input_data_for_stan_model(formula, df):    
    # Use Patsy to create design matrices
    y, X = dmatrices(formula, df, return_type='dataframe')
    
    # Extract the response variable and predictor matrix
    y = y.values.flatten()
    X = X.values
    
    # Identify unique rows in X and their counts
    unique_X, inverse_indices = np.unique(X, axis=0, return_inverse=True)
    
    weights = np.bincount(inverse_indices)
    unique_y_sum = np.bincount(inverse_indices, weights=y)
    unique_y_sq_sum = np.bincount(inverse_indices, weights=y ** 2)
    
    # Prepare the stan_data dictionary
    stan_data = {
        'N': len(unique_X),                # Number of unique observations
        'K': unique_X.shape[1],            # Number of predictors (including intercept)
        'X': unique_X,                     # Unique predictor matrix
        'y_sum': unique_y_sum,             # Sum of outcomes for each unique X
        'y_squared_sum': unique_y_sq_sum,  # Sum of squared outcomes for each unique X
        'weights': weights                 # Weights for each unique observation
    }
    return stan_data
