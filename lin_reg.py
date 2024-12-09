import clean_script
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
    X_1 = np.linalg.inv(np.matmul(Xt, X))      # inverse of X

    # print(Xt)
    # print(X_1)
    return np.matmul(np.matmul(X_1, Xt), y)

def main():
    
    X_h = np.array(make_usable("Data/cleaned.csv"))
    X_r = X_h[:,:-1]
    y_r = X_h[:,-1]

    # adding ones to the X_r
    x_cord = np.shape(X_r)[0]
    x_ones = np.ones((x_cord,1))

    # analytic solution before normalizing NOTE: 

    ### NOTE we weren't sure whether to include this, but we wanted to include unnormalized fit
    X_save = np.concatenate((x_ones, X_r), axis=1)
    print('Analytic Solution (unnormalized)', fit(X_save,y_r))



    X_r -= np.mean(X_r, axis=0)
    X_r /= np.std(X_r, axis=0)
    y_r -= np.mean(y_r, axis=0)
    y_r /= np.std(y_r, axis=0)
    X_r = np.concatenate((x_ones, X_r), axis=1)

    # should return the coefficients
    # print(fit(X_r, y_r))

    # analytic weights
    print("Analytic Solution: " + str(fit(X_r, y_r)))
    pass

if __name__ == "__main__":
    main()