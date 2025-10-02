import numpy as np
import pandas as pd
import matplotlib as plt

def load_data(path = "CreditCard.csv"):
    """
    Load the credit dataset and encode features of it. 
    Also assumes data is present and formatted as it is said to

    Encodings:
        - Gender: M -> 1, F/other -> 0 
        - Carowner: Y -> 1, N/other -> 0
        - PropertyOwner: Y -> 1, N/other -> 0
        - #Children: numeric
        - WorkPhone: numeric 0/1
        - Email: numeric 0/1

    Returns:
        x: (n_samples, 6) a float matrix
        y: (n_samples,) a float vector
    """
    data_frame = pd.read_csv(path)

    # Now to encode as the assigment has asked for
    x = pd.DataFrame({
        "Gender": (data_frame["Gender"] == "M").astype(int),
        "CarOwner": (data_frame["CarOwner"] == "Y").astype(int),
        "PropertyOwner": (data_frame["PropertyOwner"] == "Y").astype(int),
        "Children": data_frame["#Children"].astype(float),
        "WorkPhone": data_frame["workPhone"].astype(int),
        "Email": data_frame["Email"].astype(int),
    }).to_numpy(dtype=float)
    y = data_frame["CreditApproval"].astype(float).to_numpy()
    
    return x,y

def error(w, x, y):
    """
    Mean squared error for the linear model f(x) = x @ w.

    Parameters:
        w: (n_features,) vector (expected values in {-1, +1})
        x: (n_samples, n_features) matrix
        y: (n_samples,) target vector

    Returns:
       a scalar

    """
    f = x @ w 

    return np.mean((f - y) ** 2)

def fitness(w, x, y):
    """
    Calcaulte GA fitness from the mean squared error

    Fitness = exp(-er(w, x, y) / temperature)

    Parameters:
        w: (n_features,) array like weight vector {-1,+1}
        x: (n_samples, n_features) array like design matrix
        y: (n_samples,) array like target vector

    Returns:
        float of non negitive fitness (the higher the better)
    """

    return np.exp(-er(w, x, y))

def crossover(a, b):
    """
    Creates a child by taking the prefix from `a` and the suffix from `b`.
    Defaults to the midpoint split.

    Example (n=6, midpoint=3):
      a = [a0,a1,a2,a3,a4,a5]
      b = [b0,b1,b2,b3,b4,b5]
      child = [a0,a1,a2, b3,b4,b5]

    Parameters:
        a: a 6 legnth vector
        b: a 6 length vector

    Returns:
        A new vector that contains the prefix of a and the suffix of b
    """
    
    child = a.copy()
    child[3:] = b[3:]

    return child

def mutate(w, rng, p = 0.02):
    """
    Sign flips for data in {-1,+1}

    Each postion is fliped with a propablity of p
    keeps the vector the same length

    Parameters:
        w: an array of length n values being {-1,+1}
        rng: a numpy random number generator 
        p: float that is the probability of a sign flip for each index in w defaulted to 0.02
    """

    v = w.copy()
    flip = rng.random(v.size) < p
    v[flip] = -v[flip]

    return v


