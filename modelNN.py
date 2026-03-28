import math
import json
from unittest import case

import numpy as np
from openpyxl.descriptors.excel import Percentage

master_file = "MasterSave.json"
boundaries = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, float('inf')]


def load_brain(numParameters, numNeurons, numOutputs, numBins):
    W0 = load_saved_data(numParameters, numNeurons, "W0", "random")
    W1 = load_saved_data(numNeurons, numOutputs, "W1", 0.1)
    B0 = load_saved_data(1, numNeurons, "B0", 0.0)
    B1 = load_saved_data(1, numBins, "B1", 5.0)
    janMat0 = load_saved_data(numParameters, numNeurons, "janMat0", 1)
    janMat1 = load_saved_data(numNeurons, numOutputs, "janMat1", 1)
    return W0, W1, B0, B1, janMat0, janMat1

def load_saved_data(N_Inputs: int, N_outputs: int, name: str, start):
    Data = np.full((N_Inputs, N_outputs), start)
    if start == "random":
        Data = np.random.uniform(-.2, .2, (N_Inputs, N_outputs))


    # line under makes each neron have a different starting point
    # dont add until code that ensures they settle to different points works
    #Data = Data + (np.arange(N_outputs) * 0.0001)

    try:
        with open(master_file) as f:

            master = json.load(f)
            savedData = np.array(master[name])
            if savedData.shape == (N_Inputs, N_outputs):
                #Data = master[name]
                return Data

    except (FileNotFoundError, json.JSONDecodeError):
        return Data

    return Data

def save_brain(W0: np.ndarray, W1: np.ndarray, B0: np.ndarray, B1: np.ndarray, janMat0: np.ndarray, janMat1: np.ndarray):
    master = {
        "W0": W0.tolist(),
        "B0": B0.tolist(),
        "W1": W1.tolist(),
        "B1": B1.tolist(),
        "janMat0": janMat0.tolist(),
        "janMat1": janMat1.tolist()
    }

    with open(master_file, "w") as f:
        json.dump(master, f)

def sigmoid(mat : np.ndarray) -> np.ndarray:
    mat = np.clip(mat, -500, 500)
    return 1 / (1 + np.exp(-mat))


def TotalRescale(InitialBounds, FinalBounds, number):
    return FinalBounds[0] + ((number - InitialBounds[0]) / (InitialBounds[1] - InitialBounds[0])) * (
                FinalBounds[1] - FinalBounds[0])


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


def predictL1(matSig1: np.ndarray, matQuality, mW1: np.ndarray, mB1: np.ndarray,
            matBin: np.ndarray) -> np.ndarray:
    score = matSig1 @ mW1

    '''
    print("Max abs feature:", np.max(np.abs(matData)))
    print("Mean abs feature:", np.mean(np.abs(matData)))
    print("Max abs score:", np.max(np.abs(score)))
    print("Mean abs score:", np.mean(np.abs(score)))
    '''

    #print(f"mB1 shape: {mB1.shape}")
    #print(f"matBin: {matBin}")

    # prediction = TotalRescale((-55, 55), (0, 10), score) + matBias
    prediction = score + mB1.T[matBin.flatten()]
    matError = matQuality.reshape(-1, 1) - prediction
    '''
    print(f"matBias: {matBias.shape}")
    print(f"score: {score.shape}")
    print(f"prediction: {prediction.shape}")
    print(f"matQuality: {matQuality.shape}")
    print(f"matError: {matError.shape}")
    '''

    return matError

def predictL0(matData: np.ndarray, matWeights: np.ndarray, matBias: np.ndarray) -> np.ndarray:

    score = matData @ matWeights
    '''
    print("Max abs feature:", np.max(np.abs(matData)))
    print("Mean abs feature:", np.mean(np.abs(matData)))
    print("Max abs score:", np.max(np.abs(score)))
    print("Mean abs score:", np.mean(np.abs(score)))
    '''

    # prediction = TotalRescale((-55, 55), (0, 10), score) + matBias

    matSig1 = sigmoid(score + matBias)
    '''
    print(f"matBias: {matBias.shape}")
    print(f"score: {score.shape}")
    print(f"prediction: {prediction.shape}")
    print(f"matQuality: {matQuality.shape}")
    print(f"matError: {matError.shape}")
    '''
    #print(f"matSig1 (1200, 16)?: {matSig1.shape}")
    return matSig1


def Get_Current_Boosted(epochNum: int) -> tuple:
    # boundaries = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, float('inf')]
    n_groups = len(boundaries) - 1
    return epochNum % n_groups


def Get_Boost_Mat(matBin: np.ndarray, epochNum: int) -> np.ndarray:
    currentBoost = Get_Current_Boosted(epochNum)

    count = sum(1 for b in matBin if b == currentBoost)
    percentage = count / len(matBin)

    boostMult: float = Get_Boost_Multiplier(percentage)

    matBoost: np.ndarray = np.ones((len(matBin), 1))

    if count == 0:
        return matBoost

    for i, b in enumerate(matBin):
        if b == currentBoost:
            matBoost[i] = boostMult / count
        else:
            matBoost[i] = (1 - boostMult) / (len(matBin) - count)  # number of points is hard coded here "1200"

    matBoost = matBoost / matBoost.mean()

    return matBoost, currentBoost


def Get_Boost_Multiplier(percentage: float) -> float:
    k = 11
    center = 0.25
    y_min = 0.75
    y_range = 0.20

    share = y_min + (y_range / (1 + math.exp(k * (percentage - center))))

    return max(share, 0.75)


def train_weightL1(matSigL1: np.ndarray, mW1: np.ndarray, matError: np.ndarray,
                  Learning_Rate: float, Lambda: float, janMat2: np.ndarray) -> np.ndarray:
    """
    i dont think ill need boost, maybe not janMat either
    matData: (n_samples, n_features)
    matError: (n_samples,1)
    matWeights: (n_features,1)
    """

    matError = matError.reshape(-1, 1)
    mW1 = mW1.reshape(-1, 1)

    errorUpdate: np.ndarray = Learning_Rate * (matSigL1.T @ matError) / matSigL1.shape[0]

    #regL1 = regularization L1 (lasso)
    regL1: np.ndarray = (Lambda * Learning_Rate) * np.sign(mW1)

    mW1 += errorUpdate - regL1

    #matWeights *= janMat2

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
    return mW1

def train_weightL0(matData: np.ndarray, mW0: np.ndarray, matBoost: np.ndarray, matBlame: np.ndarray,
                  Learning_Rate: float, Lambda: float, janMat1: np.ndarray) -> np.ndarray:
    """
    i dont think ill need boost, maybe not janMat either
    matData: (n_samples, n_features)
    matError: (n_samples,1)
    matWeights: (n_features,1)
    """

    #matError = matBlame.reshape(-1, 1)
    #print(f"mW0 shape: {mW0.shape}")
    #mW0 = mW0.reshape(-1, 1)
    #print(f"mW0 shape: {mW0.shape}")

    errorUpdate: np.ndarray = Learning_Rate * (matData.T @ (matBlame * matBoost)) / matData.shape[0]

    #regL1 = regularization L1 (lasso)
    regL1: np.ndarray = (Lambda * Learning_Rate) * np.sign(mW0)


    mW0 += errorUpdate - regL1

    mW0 *= janMat1

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
    return mW0

def train_biasL1(mB1: np.ndarray, matError: np.ndarray, Learning_Rate: float, mW1: np.ndarray, boostedBin: int,
               matBin: np.ndarray) -> np.ndarray:
    mask = (matBin.flatten() == boostedBin)

    avgError: float = np.mean(matError[mask])
    avgWeight: float = np.mean(abs(mW1))
    mB1.T[boostedBin] += Learning_Rate * avgError #* ((avgWeight + 1) ** 2)
    '''
    if(avgWeight > 1):
        matBias += Learning_Rate * avgError * (avgWeight ** 2)
    else:
        matBias += Learning_Rate * avgError * (avgWeight + 1)
    '''
    # print("Bias: ", matBias)
    return mB1

def get_blame(mW1: np.ndarray, matSig1: np.ndarray, matError: np.ndarray) -> np.ndarray:
    i = (np.arange(matSig1.shape[1]) / 2) - 3.5

    exponents = -np.sign(i) * (2 ** (np.abs(i) - 0.5))

    safe_error = np.abs(matError) + 1e-7

    is_even_neuron = (np.arange(16) % 2 == 0).reshape(1, -1)
    mask = (is_even_neuron == (matError > 0))

    blame = (mW1.T * matSig1) * (matError * (safe_error ** exponents)) * mask
    #print(f"matSig1 shape: {matSig1.shape}")
    #print(f"matError shape: {matError.shape}")
    #print(f"mW1 shape: {mW1.shape}")
    #print(f"blame shape: {blame.shape}")
    return sigmoid(blame) - 0.5


def predict_all(matData: np.ndarray, matQuality: np.ndarray, mW0: np.ndarray, mW1: np.ndarray, mB0: np.ndarray, mB1: np.ndarray, matBin: np.ndarray) -> np.ndarray:
    matSig1 = predictL0(matData, mW0, mB0)
    matError = predictL1(matSig1, matQuality, mW1, mB1, matBin)
    return matError, matSig1

def back_prop(mW0, mB0, mW1, mB1, matSig1, matData, matError, janMat1, janMat2, matBin, matBoost, boostedBin, Learning_Rate, Lambda):
    mW1 = train_weightL1(matSig1, mW1, matError, Learning_Rate, Lambda, janMat2)
    mB1 = train_biasL1(mB1, matError, Learning_Rate, mW1, boostedBin, matBin)
    Blame = get_blame(mW1, matSig1, matError)
    mW0 = train_weightL0(matData, mW0, matBoost, Blame, Learning_Rate, Lambda, janMat1)

    # idk yet how to ues the first biases. or if i need them at all
    # at the end its easy to say the baises should be at 5.6ish, avg quality
    # there is not a way to do that for signals and patterns.
    mB0 = np.zeros((1, 16))#train_ biasL0()
    return mW0, mW1, mB0, mB1

def train_model(matData: np.ndarray, matQuality: np.ndarray, mW0: np.ndarray, mW1: np.ndarray, mB0: np.ndarray, mB1: np.ndarray,
                   prevError: float, BaselineError: float, testError: float, epochNum: int,
                   matBin: np.ndarray, janMat0, janMat1, Base_LR):  # B_LR = 0.01
    # predict done ... i think
    # Base LR used to be .005

    # Variable Definitions
    Learning_Rate = Calc_Learning_Rate(prevError, BaselineError, Base_LR)
    Lambda = Calc_Lambda(prevError, testError)
    matBoost, boostedBin = Get_Boost_Mat(matBin, epochNum)

    matError, matSig1 = predict_all(matData, matQuality, mW0, mW1, mB0, mB1, matBin)

    mW0, mW1, mB0, mB1 = back_prop(mW0, mB0, mW1, mB1, matSig1, matData, matError, janMat0, janMat1, matBin, matBoost, boostedBin, Learning_Rate, Lambda)


    return matError, mW0, mW1, mB0, mB1
