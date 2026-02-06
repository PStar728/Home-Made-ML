import math
import json

weights_file = "weights.json"
bias_file = "bias.json"

def load_bias():
    bias = 1
    try:
        with open(bias_file, "r") as f:
            bias = json.load(f)
    except FileNotFoundError:
        bias = 0
    return bias

def load_weights():
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    try:
        with open(weights_file) as f:
            loaded_weights = json.load(f)
            weights = loaded_weights
    except FileNotFoundError:
        weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return weights

def save_bias(bias):
    with open(bias_file, "w") as f:
        json.dump(bias, f)

def save_weights(weights):
    with open(weights_file, "w") as f:
        json.dump(weights, f)

def sigmoid(number):
    return 1 / (1 + math.exp(-number))

def TotalRescale(InitialBounds, FinalBounds, number):
    return FinalBounds[0] + ((number - InitialBounds[0]) / (InitialBounds[1] - InitialBounds[0])) * (FinalBounds[1] - FinalBounds[0])

def Calc_Learning_Rate(error):
    LR = ((error * 4) ** 2) *.01
    LR = min(LR, .5)
    return LR

def predict(Data, Weights, Bias):
    # Data is one datapoint!! Not entire data set
    score = sum(num * weight for num, weight in zip(Data.inputs, Weights))
    prediction = TotalRescale((-11, 11), (0, 10), score) + Bias
    error = Data.quality - prediction
    return error

def train_weights(Data, Weights, error, Learning_Rate):
    for i, x in enumerate(Data.inputs):
        Weights[i] += Learning_Rate * x * error
    return Weights

def train_bias(Bias, error, Learning_Rate):
    Bias += Learning_Rate * error
    return Bias

def train_model(Data, Weights, Bias):
    error = predict(Data, Weights, Bias)
    Learning_Rate = Calc_Learning_Rate(error)
    Weights = train_weights(Data, Weights, error, Learning_Rate)
    Bias = train_bias(Bias, error, Learning_Rate)
    return error, Weights, Bias
