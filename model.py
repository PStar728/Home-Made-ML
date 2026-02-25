import math
import json
import numpy as np

weights_file = "weights.json"
bias_file = "bias.json"

def load_bias():
    bias = 1
    try:
        with open(bias_file, "r") as f:
            bias = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        bias = 0
    return bias


def load_weights(N_parameters, start=0.1):
    try:
        with open(weights_file) as f:

            loaded_weights = json.load(f)
            if len(loaded_weights) == N_parameters:
                return loaded_weights

    except (FileNotFoundError, json.JSONDecodeError):
        return [0] * N_parameters

    return [start] * N_parameters

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

def Calc_Learning_Rate(AvgError, BaselineError, Base_LR):
    LR = ((AvgError / BaselineError) ** 2) * Base_LR
    LR = min(LR, .5)
    return LR

def Calc_Lambda(AvgError, testError):
    if testError == 0:
        return .05

    errorDiff = testError - AvgError

    x = min(errorDiff, 0.5)  # Cap at 0.5

    if x <= 0.1:
        return 0.0
    elif x <= 0.2:
        return 0.001 * (x - 0.1) / 0.1  # 0 to 0.001
    elif x <= 0.3:
        return 0.001 + 0.009 * (x - 0.2) / 0.1  # 0.001 to 0.01
    elif x <= 0.4:
        return 0.01 + 0.04 * (x - 0.3) / 0.1  # 0.01 to 0.05
    else:  # 0.4 to 0.5
        return 0.05 + 0.05 * (x - 0.4) / 0.1  # 0.05 to 0.1

def predict(Data, Weights, Bias):
    # Data is one datapoint!! Not entire data set
    score = sum(num * weight for num, weight in zip(Data.inputs, Weights))
    prediction = TotalRescale((-11, 11), (0, 10), score) + Bias
    error = Data.quality - prediction
    return error

def train_weights(Data, Weights, error, Learning_Rate, Lambda):
    for i, x in enumerate(Data.inputs):
        errorUpdate = Learning_Rate * x * error

        regL1 = (Lambda * np.sign(Weights[i]) * Learning_Rate)
        #regL2 = ((Lambda * 0.001) * (Weights[i] ** 2) * Learning_Rate)


        Weights[i] += errorUpdate - regL1

        # this is the original function not needed RN but keep it
        # the derivitive of this is used for the term1 and term2
        #  regularize = ((1 - regSig) * abs(Weights[i])) + (regSig * Weights[i] ** 2)
    return Weights

def sigmoid(number):
    return 1 / (1 + math.exp( -number))

def train_bias(Bias, error, Learning_Rate):
    Bias += Learning_Rate * error
    return Bias

def train_model(Data, Weights, Bias, AvgError, BaselineError, testError, Base_LR = .005):
    error = predict(Data, Weights, Bias)
    Learning_Rate = Calc_Learning_Rate(AvgError, BaselineError, Base_LR)
    Lambda = Calc_Lambda(AvgError, testError)
    Weights = train_weights(Data, Weights, error, Learning_Rate, Lambda)
    Bias = train_bias(Bias, error, Learning_Rate)
    return error, Weights, Bias
