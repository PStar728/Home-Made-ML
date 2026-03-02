import copy

from data import DataSet
from Test import Test
import model
import log
import numpy as np

#saves and closes ML_log.xlsx
log.Save_Close("ML_log.xlsx")

#defines training data points
print("=== ML Training Session ===")
dataStart: int = 0
dataEnd: int = 1200
DATA:DataSet = DataSet()
DATA.load_from_csv("winequality-red.csv", dataStart, dataEnd)

print(f"Loaded {len(DATA.samples)} samples\n")

#defines weights and biases
Weights: list = model.load_weights(len(DATA.samples[0].inputs))
Bias: list = model.load_bias()

print(f"Initial weights: {Weights}")
print("Loading data...")

#epochs = int(input("How many times should I train with this dataset? "))

# redefines as a numpy matrix
Inputs: np.ndarray = np.array([dp.inputs for dp in DATA.samples])
Quality: np.ndarray = np.array([dp.quality for dp in DATA.samples])
matWeights: np.ndarray = np.array(Weights)
matBias: np.ndarray = np.array(Bias)
matError: np.ndarray = np.array(None)

#initializes variables for the main
ErrorList: list = []
AvgErrorList: list = []
prevError: float = 0.5
BaselineError: float = .4
bestWeights: list = None
bestBias: list = None
strikes: int = 0
patience: int = 2
bestError: float = float('inf')
currentTestError: float = 0

#starts the main loop
#loop define to be inf. until the comment is deleted
epoch: int = 1
while epoch < 5000:
    #print(f"--- Epoch {epoch} ---")

    # Display checks if epoch == (1, 2, 5, 10, 20, 50, 100...)
    # Display is used for anything that should only be done periodically not every epoch
    Display = log.Display_Check(epoch)

    matError, matWeights, matBias = model.train_model(Inputs, Quality, matWeights, matBias, prevError, BaselineError, currentTestError)
    prevError = np.average(matError)
    log.Write_To_TempData((dataEnd - dataStart), epoch, matError)


    '''
    # loops through every data point
    for i, point in enumerate(DATA.samples, 1):
        matError, matWeights, matBias = model.train_model(point, matWeights, matBias, prevError, BaselineError, currentTestError)
        # print(f"  Sample {i}: Quality={point.quality}, Error={matError:.4f}")

        # Calcs the recent avg matError for dynamic LR
        AvgErrorList.append(abs(matError))
        if len(AvgErrorList) > 20:
            AvgErrorList.pop(0)
        prevError = sum(AvgErrorList) / len(AvgErrorList)

        # write to tempholder.txt
        if Display:
            # define an array type of all the errors for epoch
            # this will be for "Epoch matData" sheet
            ErrorList.append(matError)
            # pass array into write to xlsx to compute correct datas
            log.Write_To_txt(matWeights, matBias, i, point.quality, matError)
    '''





    # print()
    # write to ML tmep Log
    if Display:
        #log.Write_To_txt(matWeights, matBias, Quality, matError)
        log.Write_To_TempEpoch(epoch, matError)
        # clear .txt file
        # list train errors when display ---- avgtrain matError - .1
        BaselineError = max((sum(matError) / len(matError)) - 0.1, 0.25)
        ErrorList = []
        with open("Temp_Holder.txt", "w") as f:
            f.write("")

        testData = Test(dataStart, dataEnd, matWeights, matBias)
        currentTestError = testData[8]

        Weights = matWeights.flatten().tolist()

        print(f"Epoch: {epoch}")
        if currentTestError < bestError:
            # NEW BEST! Save and reset
            bestError = currentTestError
            bestWeights = copy.deepcopy(Weights)
            bestBias = copy.deepcopy(Bias)
            strikes = 0

            print(f"✓ New best: {currentTestError:.4f}")

        else:
            # Didn't beat best
            if epoch > 100:
                strikes += 1
            print(f"✗ No improvement: {currentTestError:.4f} (strike {strikes}/{patience})")

            if strikes >= patience:
                Weights = copy.deepcopy(bestWeights)
                Bias = copy.deepcopy(bestBias)
                print(f"Early stopping! Restored best: {bestError:.4f}")
                break
    epoch += 1
            
model.save_weights(Weights)
model.save_bias(Bias)

print("\n=== Training Complete ===")
print(f"Final weights: {Weights}")

Test(dataStart, dataEnd, Weights, Bias)

log.Write_To_xlsx()
log.Open_xlsm("ML_log.xlsx")