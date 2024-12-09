import numpy as np
import matplotlib.pyplot as plt

def make_usable(file: str):
    '''how we're turning the data into numpy matrices'''
    arr = np.loadtxt(file, delimiter=",")
    return arr

def fit(X: np.array, y: np.array) :
    """
    Analytical solution for linear regression using the methods we used in class.
    """
    Xt = np.transpose(X)                       # tranpose of X
    lambda_identity = 1e-5 * np.eye(X.shape[1])
    X_1 = np.linalg.pinv(np.matmul(Xt, X))      # inverse of X

    # print(Xt)
    # print(X_1)
    inner = np.matmul(X_1, Xt)
    outer = np.matmul(inner, y)
    return outer

def plot_results(y_actual, y_pred):
    """
    Visualize MLR results with Matplotlib.
    """
    # Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 5))

    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_actual, y_pred, alpha=0.6, color="blue", edgecolor="k")
    plt.plot(y_actual, y_actual, color="red", linestyle="--", label="Perfect Prediction")
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()

    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = y_actual - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, color="green", edgecolor="k")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")

    # Show plots
    plt.tight_layout()
    plt.savefig("figures/lin_reg_results.pdf", format="pdf")
    # plt.show()

def main():
    
    X_h = np.array(make_usable("Data/num_data.csv"))
    X_r = X_h[:,:-1]
    y_r = X_h[:,-1]

    # adding ones to the X_r
    x_cord = np.shape(X_r)[0]
    x_ones = np.ones((x_cord,1))

    X_r -= np.mean(X_r, axis=0)
    X_r /= np.std(X_r, axis=0)
    y_r -= np.mean(y_r, axis=0)
    y_r /= np.std(y_r, axis=0)
    X_r = np.concatenate((x_ones, X_r), axis=1)

    # should return the coefficients
    # print(fit(X_r, y_r))
    weights = fit(X_r, y_r)
    # analytic weights
    print("Analytic Solution: " + str(weights))
    y_pred = np.matmul(X_r, weights)
    
    plot_results(y_r, y_pred)
    pass

if __name__ == "__main__":
    main()