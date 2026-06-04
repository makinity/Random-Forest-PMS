import pickle

# load model
with open("models/model_v1.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/labels.pkl", "rb") as f:
    label_names = pickle.load(f)