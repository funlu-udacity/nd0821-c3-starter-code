# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

import pickle
import os
# Add code to load in the data.

data = pd.read_csv(os.path.join(os.getcwd(), "data", "census.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.

rf_model = train_model(X_train, y_train, 100)

#preds = mdl.inference(modl, X_test)

with open(os.path.join(os.getcwd(), "model", "rf_model.pkl"), 'wb') as handle:
    pickle.dump(rf_model, handle)

with open(os.path.join(os.getcwd(), "model", "encoder.pkl"), 'wb') as handle:
    pickle.dump(encoder, handle)

with open(os.path.join(os.getcwd(), "model", "lb.pkl"), 'wb') as handle:
    pickle.dump(lb, handle)

