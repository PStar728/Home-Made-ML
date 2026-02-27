from data import DataSet
from model import predict
from log import Calculate_Epoch_Data, Log_Tests

TESTDATA = DataSet()

def Define_Testing_Ranges(dataStart, dataEnd):
    TESTDATA.load_from_csv("winequality-red.csv", 0, dataStart)
    TESTDATA.load_from_csv("winequality-red.csv", dataEnd)

def Test(dataStart, dataEnd, Weights, Bias):

    if len(TESTDATA.samples) == 0:
        Define_Testing_Ranges(dataStart, dataEnd)

    testErrors = []
    for point in TESTDATA.samples:
        testErrors.append(predict(point, Weights, Bias))

    testsLine = Calculate_Epoch_Data("Test", testErrors)
    Log_Tests(testsLine)
    return testsLine