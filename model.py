import math
import json

weights_file = "weights.json"

def load_weights():
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    try:
        with open(weights_file) as f:
            loaded_weights = json.load(f)
            weights = loaded_weights
    except FileNotFoundError:
        weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return weights

def save_weights(weights):
    with open(weights_file, "w") as f:
        json.dump(weights, f)

def sigmoid(number):
    return 1 / (1 + math.exp(-number))

def TotalRescale(InitialBounds, FinalBounds, number):
    return FinalBounds[0] + ((number - InitialBounds[0]) / (InitialBounds[1] - InitialBounds[0])) * (FinalBounds[1] - FinalBounds[0])

def predict(Data, Weights):
    # Data is one datapoint!! Not entire data set
    score = sum(num * weight for num, weight in zip(Data.inputs, Weights))
    prediction = TotalRescale((-11, 11), (0, 10), score)
    difference = Data.quality - prediction
    return difference

def train_weights(Data, Weights, difference, Learning_Rate = .05):
    for i, x in enumerate(Data.inputs):
        Weights[i] += Learning_Rate * x * difference
    return Weights

def train_model(Data, Weights):
    diff = predict(Data, Weights)
    return train_weights(Data, Weights, diff)
