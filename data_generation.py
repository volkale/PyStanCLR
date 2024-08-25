import numpy as np
import pandas as pd

# True coefficients
INTERCEPT_TRUE = 0.3
BETA_TRUE = np.array([1.0, 2.0, -1.0])
SIGMA_TRUE = 0.5


def get_sample_data(n_samples, n_feature_values):
    
    n_features = len(BETA_TRUE)

    # Generate random predictors
    X = np.random.randint(0, n_feature_values, size=(n_samples, n_features))
    
    # Generate outcome variable
    y = INTERCEPT_TRUE + X @ BETA_TRUE + np.random.randn(n_samples) * SIGMA_TRUE
    
    # Combine into a DataFrame
    feature_columns = [f'x{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_columns)
    df['y'] = y

    return df
