from data import DataSet
from model import predict
from log import Calculate_Epoch_Data, Log_Tests
import numpy as np

TESTDATA = DataSet()

def Define_Testing_Ranges(dataStart, dataEnd):

    TESTDATA.load_from_csv("winequality-red.csv", 0, dataStart, False)
    TESTDATA.load_from_csv("winequality-red.csv", dataEnd, None ,False)

def Test(dataStart: int, dataEnd: int, matWeight, matBias):

    if len(TESTDATA.samples) == 0:
        Define_Testing_Ranges(dataStart, dataEnd)

    testData: np.ndarray = np.array([dp.inputs for dp in TESTDATA.samples])
    testData = testData - testData.mean(axis=0)
    testQuality: np.ndarray = np.array([dp.quality for dp in TESTDATA.samples])

    testErrors = predict(testData, testQuality, matWeight, matBias)

    #for point in TESTDATA.samples:
     #   testErrors.append(predict(point, matWeight, matBias))

    testsLine = Calculate_Epoch_Data("Test", testErrors)
    Log_Tests(testsLine)
    return testsLine