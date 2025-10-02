import numpy as np
import pandas as pd
import matplotlib as plt

def load_data(path = "CreditCard.csv"):
    data_frame = pd.read_csv(path)

    # Now to encode as the assigment has asked for
    x = pd.DataFrame({
        "Gender": (data_frame["Gender"] == "M").astype(int),
        "CarOwner": (data_frame["CarOwner"] == "Y").astype(int),
        "PropertyOwner": (data_frame["PropertyOwner"] == "Y").astype(int),
        "Children": data_frame["#Children"].astype(float),
        "WorkPhone": data_frame["Workhone"].astype(int),
        "Email": data_frame["Email"].astype(int),
    }).to_numpy(dtype=float)
    y = data_frame["CreditApproval"].astype(float).to_numpy()
    
    return x,y

