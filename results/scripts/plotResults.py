import pickle
from Experiment import Experiment

results = {}
with open("selectivityVsSpeed.pkl", "rb") as f:
    results = pickle.load(f)
print(results)
