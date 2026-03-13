import math
import json
from unittest import case

import numpy as np
from openpyxl.descriptors.excel import Percentage

weights_file = "weights.json"
bias_file = "bias.json"

def load_bias():
    bias = 1.0
    try:
        with open(bias_file, "r") as f:
            #bias = json.load(f)
            bias = 5.6
    except (FileNotFoundError, json.JSONDecodeError):
        bias = 5.6
    return bias


def load_weights(N_parameters, start=0.1):
    try:
        with open(weights_file) as f:

            loaded_weights = json.load(f)
            if len(loaded_weights) == N_parameters:
                #return loaded_weights
                return [0.1] * N_parameters

    except (FileNotFoundError, json.JSONDecodeError):
        return [0.0] * N_parameters

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

def Calc_Learning_Rate(prevError: float, BaselineError: float, Base_LR) -> float:
    LR = ((prevError / BaselineError) ** 2) * Base_LR
    LR = min(LR, .5)
    return LR

def Calc_Lambda(prevError: float, testError: float) -> float:
    if testError == 0:
        return .05

    errorDiff = testError - prevError

    x = min(errorDiff, 0.5)  # Cap at 0.5

    if x <= 0.05:
        return 0
    elif x <= 0.15:
        return 0.001 * (x - 0.1) / 0.1  # 0 to 0.001
    elif x <= 0.3:
        return 0.001 + 0.009 * (x - 0.2) / 0.1  # 0.001 to 0.01
    elif x <= 0.4:
        return 0.01 + 0.04 * (x - 0.3) / 0.1  # 0.01 to 0.05
    else:  # 0.4 to 0.5
        return 0.05 + 0.05 * (x - 0.4) / 0.1  # 0.05 to 0.1

def predict(matData: np.ndarray, matQuality, matWeights: np.ndarray, matBias: np.ndarray) -> np.ndarray:

    score = matData @ matWeights

    '''
    print("Max abs feature:", np.max(np.abs(matData)))
    print("Mean abs feature:", np.mean(np.abs(matData)))
    print("Max abs score:", np.max(np.abs(score)))
    print("Mean abs score:", np.mean(np.abs(score)))
    '''


    #prediction = TotalRescale((-55, 55), (0, 10), score) + matBias
    prediction = score + matBias
    matError = matQuality.reshape(-1, 1) - prediction
    '''
    print(f"matBias: {matBias.shape}")
    print(f"score: {score.shape}")
    print(f"prediction: {prediction.shape}")
    print(f"matQuality: {matQuality.shape}")
    print(f"matError: {matError.shape}")
    '''

    return matError

def Get_Current_Boosted(epochNum: int) -> tuple:
    boundaries = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, float('inf')]
    n_groups = len(boundaries) - 1
    current_group = epochNum % n_groups
    return (boundaries[current_group], boundaries[current_group + 1])

def Get_Boost_Mat(Weirdness: np.ndarray, epochNum: int) -> np.ndarray:

    (boundLow, boundHigh) = Get_Current_Boosted(epochNum)

    count = sum(1 for w in Weirdness if boundLow <= w < boundHigh)
    percentage = count / len(Weirdness)

    boostMult: float = Get_Boost_Multiplier(percentage)

    matBoost: np.ndarray = np.ones((len(Weirdness), 1))

    if count == 0:
        return matBoost

    for i, w in enumerate(Weirdness):
        if boundLow <= w < boundHigh:
            matBoost[i] = boostMult/count
        else :
            matBoost[i] = (1-boostMult)/(1200-count) #number of points is hard coded here "1200"

    matBoost = matBoost / matBoost.mean()

    return matBoost

def Get_Boost_Multiplier(percentage: float) -> float:

    k = 11
    center = 0.25
    y_min = 0.75
    y_range = 0.20

    share = y_min + (y_range / (1 + math.exp(k * (percentage - center))))

    return max(share, 0.75)



def train_weights(matData: np.ndarray, matWeights: np.ndarray, matBoost: np.ndarray, matError: np.ndarray, Learning_Rate: float, Lambda: float) -> np.ndarray:
    """
    matData: (n_samples, n_features)
    matError: (n_samples,1)
    matWeights: (n_features,1)
    """

    matError = matError.reshape(-1, 1)
    matWeights = matWeights.reshape(-1, 1)

    errorUpdate: np.ndarray = Learning_Rate * (matData.T @ (matError * matBoost)) / matData.shape[0]

    regL1: np.ndarray = (Lambda * Learning_Rate) * np.sign(matWeights)

    matWeights += errorUpdate - regL1

    '''
    for i, x in enumerate(matData.inputs):
        errorUpdate = Learning_Rate * x * matError

        regL1 = (Lambda * np.sign(matWeights[i]) * Learning_Rate)
        #regL2 = ((Lambda * 0.001) * (matWeights[i] ** 2) * Learning_Rate)


        matWeights[i] += errorUpdate - regL1

        # this is the original function not needed RN but keep it
        # the derivitive of this is used for the term1 and term2
        #  regularize = ((1 - regSig) * abs(matWeights[i])) + (regSig * matWeights[i] ** 2)
        '''
    return matWeights

def sigmoid(number):
    return 1 / (1 + math.exp( -number))

def train_bias(matBias: np.ndarray, matError: np.ndarray, Learning_Rate: float, matWeights: np.ndarray, matBoost: np.ndarray) -> np.ndarray:

    if(matBoost is not None):
        avgError: float = np.mean(matError * matBoost)
    else:
        avgError: float = np.mean(matError)
    avgWeight: float = np.mean(abs(matWeights))
    matBias += Learning_Rate * avgError * ((avgWeight + 1) ** 2)
    '''
    if(avgWeight > 1):
        matBias += Learning_Rate * avgError * (avgWeight ** 2)
    else:
        matBias += Learning_Rate * avgError * (avgWeight + 1)
    '''
    #print("Bias: ", matBias)
    return matBias

def train_model(matData: np.ndarray, Weirdness: np.ndarray, matQuality: np.ndarray, matWeights: np.ndarray, matBias: np.ndarray, prevError: float, BaselineError: float, testError: float, epochNum: int, Base_LR = .01):
    # predict done ... i think
    #Base LR used to be .005
    matError = predict(matData, matQuality, matWeights, matBias)


    '''
    (low, high) = Get_Current_Boosted(epochNum)

    # 2. Find which wines belong to this bin
    # (We flatten Weirdness to make it a simple 1D list for the comparison)
    mask = (Weirdness.flatten() >= low) & (Weirdness.flatten() < high)

    # 3. Pull those specific errors
    bin_errors = matError[mask]

    # 4. Calculate the Raw Mean (to see the "Shift")
    bin_mean = np.mean(bin_errors) if len(bin_errors) > 0 else 0

    print(f"Epoch {epochNum} | Bin {epochNum % 13} | Wines in Bin: {len(bin_errors)} / {len(matError)}")

    from openpyxl import Workbook, load_workbook
    import log
    file_temp = "Temp_Log.xlsx"
    wb_temp = log.try_temp_workbook(file_temp)
    sheet_temp = wb_temp["TEMP_DATA"]
    sheet_temp.append(["Epoch #"] + [epochNum] + ["Bin"] + [epochNum%13] + ["Error"] + [bin_mean])
    wb_temp.save(file_temp)
    '''




    # Learning rate done
    Learning_Rate = Calc_Learning_Rate(prevError, BaselineError, Base_LR)
    # lambda done
    Lambda = Calc_Lambda(prevError, testError)

    matBoost = Get_Boost_Mat(Weirdness, epochNum)

    #train weights done
    matWeights = train_weights(matData, matWeights, matBoost, matError, Learning_Rate, Lambda)
    #bias done
    matBias = train_bias(matBias, matError, Learning_Rate, matWeights, None)
    return matError, matWeights, matBias
