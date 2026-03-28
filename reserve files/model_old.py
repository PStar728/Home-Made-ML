import math
import json

weights_file = "MasterSave.json"
weights = {
    'fixed acidity': 0.1,
    'volatile acidity': 0.1,
    'citric acid': 0.1,
    'residual sugar': 0.1,
    'chlorides': 0.1,
    'free sulfur dioxide': 0.1,
    'total sulfur dioxide': 0.1,
    'density': 0.1,
    'pH': 0.1,
    'sulphates': 0.1,
    'alcohol': 0.1
}

def load_weights():
    global weights
    try:
        with open(weights_file) as f:
            loaded_weights = json.load(f)
            weights = loaded_weights
    except FileNotFoundError:
        weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def save_weights():
    
    # needs adjusting from change to wine instead of pass/fail classes
    
    global attendence_weight, study_weight, homework_weight
    weights = [attendence_weight, study_weight, homework_weight]
    with open(weights_file, "w") as f:
        json.dump(weights, f)
        
def sigmoid(number):
    return 1 / (1 + math.exp(-number))