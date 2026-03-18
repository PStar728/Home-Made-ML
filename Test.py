from pickle import GLOBAL

from data import DataSet
from model import predict
from log import Calculate_Epoch_Data, Log_Tests
import numpy as np

QuizData = DataSet()
TestData = DataSet()

bins = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

TestData.load_from_csv("winequality-red.csv", 1400, 1599, False)
QuizData.load_from_csv("winequality-red.csv", 1200, 1399, False)

quizGrad = None

def Test(matWeight, matBias):

    quizData: np.ndarray = np.array([dp.inputs for dp in QuizData.samples])
    quizData = quizData - quizData.mean(axis=0)
    quizQuality: np.ndarray = np.array([dp.quality for dp in QuizData.samples])

    quizWeirdness: np.ndarray = np.array(QuizData.weirdness).reshape(-1, 1)
    matTBin: np.ndarray = np.digitize(quizWeirdness.flatten(), bins)
    matTBin = matTBin.reshape(-1, 1)

    quizErrors = predict(quizData, quizQuality, matWeight, matBias, matTBin)

    global quizGrad
    quizGrad = ((quizData.T @ quizErrors) / quizData.shape[0])


    #for point in QuizData.samples:
     #   testErrors.append(predict(point, matWeight, matBias))

    testsLine = Calculate_Epoch_Data("Test", quizErrors)
    Log_Tests(testsLine)
    return testsLine