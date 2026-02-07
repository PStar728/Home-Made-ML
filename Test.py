from data import DataSet
from model import predict
from log import Calculate_Epoch_Data, Log_Test_Data

def Test(dataStart, dataEnd, Weights, Bias):
    TESTDATA = DataSet()
    TESTDATA.load_from_csv("winequality-red.csv", 0, dataStart)
    TESTDATA.load_from_csv("winequality-red.csv", dataEnd)

    testErrors = []
    for point in TESTDATA.samples:
        testErrors.append(predict(point, Weights, Bias))

    Calculate_Epoch_Data("Test", testErrors)
    Log_Test_Data()



