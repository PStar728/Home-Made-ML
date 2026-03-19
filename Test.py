from pickle import GLOBAL

from data import DataSet
from model import predict
from log import Calculate_Epoch_Data, Log_Tests
import numpy as np

QuizData = DataSet()
TestData = DataSet()

bins = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

TestData.load_from_csv("wine_shuffled.csv", 1400, 1599, False)
QuizData.load_from_csv("wine_shuffled.csv", 1200, 1399, False)

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

quizGrad = None

def TestClean(matWeight, matBias):

    quizData: np.ndarray = np.array([dp.inputs for dp in QuizData.samples])
    quizData = quizData - quizData.mean(axis=0)
    quizQuality: np.ndarray = np.array([dp.quality for dp in QuizData.samples])

    quizWeirdness: np.ndarray = np.array(QuizData.weirdness).reshape(-1, 1)
    matTBin: np.ndarray = np.digitize(quizWeirdness.flatten(), bins)
    matTBin = matTBin.reshape(-1, 1)

    quizErrors = predict(quizData, quizQuality, matWeight, matBias, matTBin)

    counts = np.bincount(matTBin.flatten(), minlength=13)
    member_weights = np.where(counts > 0, 1.0 / counts, 0)
    equalizer = member_weights[matTBin.flatten()].reshape(-1, 1)

    return  (quizData.T @ (quizErrors * equalizer))
    global quizGrad
    quizGrad = ((quizData.T @ quizErrors) / quizData.shape[0])


    #for point in QuizData.samples:
     #   testErrors.append(predict(point, matWeight, matBias))

    testsLine = Calculate_Epoch_Data("Test", quizErrors)
    Log_Tests(testsLine)
    return testsLine

def FinalTest(matWeight, matBias):

    testData: np.ndarray = np.array([dp.inputs for dp in TestData.samples])
    testData = testData - testData.mean(axis=0)
    testQuality: np.ndarray = np.array([dp.quality for dp in TestData.samples])

    testWeirdness: np.ndarray = np.array(TestData.weirdness).reshape(-1, 1)
    matTBin: np.ndarray = np.digitize(testWeirdness.flatten(), bins)
    matTBin = matTBin.reshape(-1, 1)

    testErrors = predict(testData, testQuality, matWeight, matBias, matTBin)

    testsLine = Calculate_Epoch_Data("Final Test", testErrors)
    Log_Tests(testsLine)
    #return testsLine