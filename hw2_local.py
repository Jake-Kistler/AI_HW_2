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

    return np.exp(-error(w, x, y))

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

def generic_algorithm(x, y, pop_size=20, gens=60, mut_p=0.02, seed=0):

    rng = np.random.default_rng(seed)
    d = x.shape[1]
    pop = rng.choice([-1.0,1.0], size=(pop_size, d))

    best_errs = []
    best_w = pop[0].copy()
    best_e = error(best_w, x, y)

    for _ in range(gens):
        # fitness for selction
        fits = np.array([fitness(ind, x, y) for ind in pop])

        # track the best
        i_best = np.argmax(fits)
        e_cur = error(pop[i_best], x, y)

        if e_cur < best_e:
            best_e = e_cur
            best_w = pop[i_best].copy()
        best_errs.append(best_e)

        # selection propablities
        probs = fits / (fits.sum() + 1e-12)

        # create the new generation
        new_pop = []
        while len(new_pop) < pop_size:

            i, j = rng.choice(pop_size, size=2, replace=True, p=probs)
            child = crossover(pop[i], pop[j]) # now we have the midpoint/cross over

            child = mutate(child, rng, p=mut_p) # now we have the sign flips done
            new_pop.append(child)

        pop = np.array(new_pop)

    return best_w, np.array(best_errs)
    

# run it an plot it up
x,y = load_data("CreditCard.csv")
w_ga, errs_ga = generic_algorithm(x, y, pop_size=20, gens=60, mut_p=0.02, seed=0)

print("GA best w: ", w_ga)
print("GA best error: ", errs_ga[-1])

plt.figure()
plt.plot(range(len(errs_ga)), errs_ga, marker="o")
plt.xlabel("Generation")
plt.ylabel("Best error(w)")
plt.title("Generic algorithm convergence")
plt.tight_layout()
plt.savefig("fig_generic.png", dpi=200)