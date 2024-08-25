functions {
  real compressed_normal_lpdf_(real y_sum_, real y_squared_sum_, int weight, real mu, real sigma) {
    return -0.5 * log(2 * pi() * square(sigma)) * weight - 0.5 * ( (y_squared_sum_ + weight * square(mu) - 2 * y_sum_ * mu) / square(sigma) );
  }
}

data {
  int<lower=0> N;             // number of unique observations
  int<lower=0> K;             // number of predictors (including intercept)
  matrix[N, K] X;             // predictor matrix (with a column of ones for intercept)
  vector[N] y_sum;            // outcome variable but summed for constant X
  vector[N] y_squared_sum;    // squared outcome variable but summed for constant X
  int weights[N];             // weights vector indicating the number of unique feature rows
}

parameters {
  vector[K] beta;             // regression coefficients
  real<lower=0> sigma;        // error scale (standard deviation)
}

model {
    for (n in 1:N)
    target += compressed_normal_lpdf_(y_sum[n], y_squared_sum[n], weights[n], X[n] * beta, sigma);
}
