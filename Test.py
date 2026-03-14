from data import DataSet
from model import predict
from log import Calculate_Epoch_Data, Log_Tests
import numpy as np

TESTDATA = DataSet()
bins = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

def Define_Testing_Ranges(dataStart, dataEnd):

    #TESTDATA.load_from_csv("winequality-red.csv", 0, dataStart, False)
    TESTDATA.load_from_csv("winequality-red.csv", dataEnd, None ,False)

def Test(dataStart: int, dataEnd: int, matWeight, matBias):

    if len(TESTDATA.samples) == 0:
        Define_Testing_Ranges(dataStart, dataEnd)

    testData: np.ndarray = np.array([dp.inputs for dp in TESTDATA.samples])
    testData = testData - testData.mean(axis=0)
    testQuality: np.ndarray = np.array([dp.quality for dp in TESTDATA.samples])

    TestWeirdness: np.ndarray = np.array(TESTDATA.weirdness).reshape(-1, 1)
    matTBin: np.ndarray = np.digitize(TestWeirdness.flatten(), bins)
    matTBin = matTBin.reshape(-1, 1)

    testErrors = predict(testData, testQuality, matWeight, matBias, matTBin)

    #for point in TESTDATA.samples:
     #   testErrors.append(predict(point, matWeight, matBias))

    testsLine = Calculate_Epoch_Data("Test", testErrors)
    Log_Tests(testsLine)
    return testsLine