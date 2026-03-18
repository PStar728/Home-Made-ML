import numpy as np
from model import predict
import Test

def Clean(janMat: np.ndarray, matWeights: np.ndarray, matBias: np.ndarray, Inputs: np.ndarray, Quality: np.ndarray, matBin) -> np.ndarray:
    trainError = predict(Inputs, Quality, matWeights, matBias, matBin)
    trainGrad = (Inputs.T @ trainError) / Inputs.shape[0]
    Test.Test(matWeights, matBias)
    quizGrad = Test.quizGrad

    traitor = ((trainGrad * quizGrad) < 0)

    threshold = .05
    unstable = (np.abs(trainGrad - quizGrad) > (threshold * np.abs(matWeights)))
    invalids = traitor | unstable
    janMat[invalids] = 0

    return janMat