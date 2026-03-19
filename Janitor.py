import numpy as np
from model import predict
import Test

def Clean(janMat: np.ndarray, matWeights: np.ndarray, matBias: np.ndarray, Inputs: np.ndarray, Quality: np.ndarray, matBin, epoch: int) -> np.ndarray:

    counts = np.bincount(matBin.flatten(), minlength=13)
    member_weights = np.where(counts > 0, 1.0 / counts, 0)
    equalizer = member_weights[matBin.flatten()].reshape(-1, 1)

    trainError = predict(Inputs, Quality, matWeights, matBias, matBin)
    trainGrad = (Inputs.T @ (trainError * equalizer))
    quizGrad = Test.TestClean(matWeights, matBias)

    traitor = ((trainGrad * quizGrad) < 0)
    janMat[traitor] = 0

    threshold = .05
    unstable = (np.abs(trainGrad - quizGrad) > (threshold * np.abs(matWeights)))
    invalids = unstable

    score = abs(trainGrad)

    if epoch > 2000:
        score = abs(trainGrad - quizGrad)

    percentActive: int = int(round(np.sum(janMat) * .2))

    candidates = (janMat == 1) & (invalids == True)
    candidatesI = np.where(candidates.flatten())[0]

    if len(candidatesI) > 0:
        candidateScores = score.flatten()[candidatesI]
        rankedIndex = np.argsort(candidateScores)[::-1]
        to_fire_indices = candidatesI[rankedIndex[:percentActive]]

    janMat.ravel()[to_fire_indices] = 0

    print(np.sum(janMat == 0))

    return janMat

def UpdateB_LR(Base_LR: float) -> float:
    return max((Base_LR * .675), 0.01)