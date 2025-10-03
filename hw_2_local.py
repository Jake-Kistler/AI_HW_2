import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        "WorkPhone": data_frame["WorkPhone"].astype(int),
        "Email": data_frame["Email_ID"].astype(int),
    }).to_numpy(dtype=float)
    y = data_frame["CreditApprove"].astype(float).to_numpy()
    
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

def neighbor_oneflip(w):
    """
    All 1-flip neighbors of w (flipping only 1 postion)
    """
    nbs = []

    for j in range(w.size):

        v = w.copy()
        v[j] = -v[j] # flip the sign at index j
        nbs.append(v)

    return nbs

def hill_climb(x, y, max_rounds=1000, seed=50):
    """
    Hill climbing over {-1,+1}^6 using 1-flip neighbors
    Starts from a random +-1 vector, moves on to a neighbor if it has a lower error
    Logging error once per round
    """
    rng = np.random.default_rng(seed)
    w = rng.choice([-1.0,1.0], size=x.shape[1]) # this is the random start
    errors = [error(w, x, y)]

    for _ in range(max_rounds):
        # evaluate neighbors and pick the best
        candidates = [(error(v, x, y), v) for v in neighbor_oneflip(w)]
        best_e, best_v = min(candidates, key=lambda t: t[0])

        if best_e < errors[-1]: # if we found a better neighbor move on
            w = best_v
            errors.append(best_e)
        else:
            break # no better neighbors, we are done
    
    return w, np.array(errors)
    
if __name__ == "__main__":
    x, y = load_data("CreditCard.csv")

    w_hc, errs_hc = hill_climb(x, y, max_rounds=1000, seed=0)
    print("HC best w:", w_hc)
    print("HC best error:", errs_hc[-1])

    plt.figure()
    plt.plot(range(len(errs_hc)), errs_hc, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Best error(w)")
    plt.title("Hill climbing convergence")
    plt.tight_layout()
    plt.savefig("fig_hill_climb.png", dpi=200)