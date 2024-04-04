#Asaf Benor
import numpy as np

# The linear regression function simulates a linear regression experiment
def linear_regression(n, m, determined_beta, X, sigma):
    # Generation of the noise vector with standard deviation 'sigma'
    noise_vector = np.random.normal(loc=0, scale=np.sqrt(sigma), size=n)
    noise_vector = np.reshape(noise_vector, (n,1))
    
    # Calculation of Y with added noise
    Y = (X @ determined_beta) + noise_vector
    
    # Estimation of beta coefficients using the ordinary least squares method
    calculated_beta = np.linalg.inv((np.transpose(X) @ X)) @ (np.transpose(X) @ Y)
    
    # Displaying the noise, calculated beta, and Y for analysis
    print("The e noise is: \n", noise_vector)
    print("The calculated beta is:\n", calculated_beta)
    print("Y is: \n", Y)

def main():
    # Set the dimensions of the experiment
    n = m = 4  # Number of observations and features
    print("n = ",n," m = ",m)
    
    # Determining an initial beta for generating Y
    determined_beta = np.full((m, 1), 95)
    
    # Random generation of X matrix
    X = np.random.rand(n, m)
    print("1. The X matrix is:")
    print(X)
    print("2. The determined beta is:\n", determined_beta)
    
    # Initial experiment with a small sigma value
    sigma1 = 0.1
    noise_vector1 = np.random.normal(loc=0, scale=np.sqrt(sigma1), size=n)
    noise_vector1 = np.reshape(noise_vector1, (n,1))
    Y1 = (X @ determined_beta) + noise_vector1
    calculated_beta1 = np.linalg.inv((np.transpose(X) @ X)) @ (np.transpose(X) @ Y1)
    print("3. The e noise is: \n", noise_vector1)
    print("4. Y is: \n", Y1)
    print("5. The calculated beta is:\n", calculated_beta1)
    print("\n 6.The estimated beta coefficients in a linear regression model may differ from the initial beta values\n"
      "due to the presence of noise and the limited sample size. Noise introduces variability in the response Y,\n"
      "which does not perfectly follow the linear relationship defined by X and the true beta. This variability\n"
      "can lead to inaccuracies in the estimated coefficients because the model attempts to fit the noisy data\n"
      "as closely as possible. Even if the noise is normally distributed with a mean of zero, the random fluctuations\n"
      "it introduces can cause the estimated beta to deviate from its true value. Additionally, the way the input\n"
      "features X are generated and their specific values can affect the precision of the beta estimation. If X\n"
      "does not sufficiently capture the variability in Y or if there's multicollinearity among the features, the\n"
      "accuracy of the estimated beta can further decrease. Therefore, the changes in the estimated beta from the\n"
      "initial beta are a result of attempting to model the noisy, real-world data represented by the generated\n"
      "samples in the simulation.\n")


    # Exploring different sigma values to examine its effect on beta and Y
    sigma = [0.1, 1, 10]
    print("7. ")
    for index, sig_val in enumerate(sigma):
        print(f"\nFor sigma = {sig_val}, the results are: ")
        linear_regression(n, m, determined_beta, X, sig_val)

    print("\nVariability in Y: The variability of the response Y increases.\n"
      "This is because the noise vector, added to the linear combination of X and beta,\n"
      "has a larger range of values, causing more fluctuation in the output Y.\n\n"
      "Accuracy of Estimated Beta: The accuracy of the estimated beta coefficients decreases.\n"
      "With higher noise levels, the added randomness makes it more difficult to accurately estimate\n"
      "the true relationship between X and Y.\n"
      "This results in estimated beta values that may deviate more from the true beta values used to generate Y.\n")


if __name__ == "__main__":
    main()
