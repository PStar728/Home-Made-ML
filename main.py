import copy

from data import DataSet
from model import boundaries
from Test import Test
from Test import FinalTest
import model
import log
import numpy as np
import time
import Janitor
start = time.time()

#saves and closes ML_log.xlsx
log.Save_Close("ML_log.xlsx")
log.Clear_Temp("Temp_Log.xlsx")

#defines training data points
print("=== ML Training Session ===")
dataStart: int = 0
dataEnd: int = 1199
DATA:DataSet = DataSet()
DATA.load_from_csv("wine_shuffled.csv", dataStart, dataEnd, True)

print(f"Loaded {len(DATA.samples)} samples\n")

#defines weights and biases
Weights: list = model.load_weights(len(DATA.samples[0].inputs))
Bias: list = model.load_bias()

print(f"Initial weights: {Weights}")
print("Loading data...")

#epochs = int(input("How many times should I train with this dataset? "))

bins = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

# redefines as a numpy matrix
Inputs: np.ndarray = np.array([dp.inputs for dp in DATA.samples])
Inputs = Inputs - Inputs.mean(axis=0)
Quality: np.ndarray = np.array([dp.quality for dp in DATA.samples]).reshape(-1, 1)
Weirdness: np.ndarray = np.array(DATA.weirdness).reshape(-1, 1)
matWeights: np.ndarray = np.array(Weights).reshape(-1, 1)
matBias: np.ndarray = np.array([Bias]).reshape(-1,1)
matError: np.ndarray = np.array(None)
matBin: np.ndarray = np.digitize(Weirdness.flatten(), bins)
matBin = matBin.reshape(-1, 1)

#initializes variables for the main
ErrorList: list = []
AvgErrorList: list = []
prevError: float = 0.5
BaselineError: float = .4
matBestWeights: np.ndarray = None
matBestBias: np.ndarray = None
strikes: int = 0
patience: int = 3
bestError: float = float('inf')
currentTestError: float = 0
janMat: np.ndarray = np.ones_like(matWeights)
Base_LR: float = .07

#starts the main loop
#loop define to be inf. until the comment is deleted
epoch: int = 1
while epoch <= 30000:
    #print(f"--- Epoch {epoch} ---")

    # Display checks if epoch == (1, 2, 5, 10, 20, 50, 100...)
    # Display is used for anything that should only be done periodically not every epoch
    Display = log.Display_Check(epoch)
    #t0 = time.time()
    matError, matWeights, matBias = model.train_model(Inputs, Weirdness, Quality, matWeights, matBias, prevError, BaselineError, currentTestError, epoch, matBin, janMat, Base_LR)
    #print(time.time() - t0)
    prevError = np.average(np.abs(matError))


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
        print("Bias: ", matBias)
        log.Write_To_TempData((dataEnd - dataStart), epoch, matError)
        #log.Write_To_txt(matWeights, matBias, Quality, matError)
        #t1 = time.time()
        log.Write_To_TempEpoch(epoch, matError)
        #print(time.time() - t1)
        # clear .txt file
        # list train errors when display ---- avgtrain matError - .1
        BaselineError = max((sum(matError) / len(matError)) - 0.1, 0.25)
        ErrorList = []
        with open("Temp_Holder.txt", "w") as f:
            f.write("")

        testData = Test(matWeights, matBias)
        FinalTest(matWeights, matBias)
        currentTestError = testData[8]

        #Weights = matWeights.flatten().tolist()
        #Bias = matBias.flatten().tolist()

        print(f"Epoch: {epoch}")
        if currentTestError < bestError:
            # NEW BEST! Save and reset
            bestError = currentTestError
            matBestWeights = matWeights.copy()
            matBestBias = matBias.copy()
            strikes = 0

            print(f"✓ New best: {currentTestError:.4f}")

        else:
            # Didn't beat best
            if strikes == 0:
                janMat = Janitor.Clean(janMat, matBestWeights, matBestBias, Inputs, Quality, matBin, epoch)
                Base_LR = .025
                matWeights = matBestWeights.copy()
                matBias = matBestBias.copy()
            strikes += 1
                #break

            print(f"✗ No improvement: {currentTestError:.4f} (strike {strikes}/{patience})")



            if strikes >= patience:
                matWeights = matBestWeights.copy()
                matBias = matBestBias.copy()
                print(f"Early stopping! Restored best: {bestError:.4f}")
                break
    epoch += 1
            
model.save_weights(matWeights.flatten().tolist())
model.save_bias(matBias.flatten().tolist())

print("\n=== Training Complete ===")
print(f"Final weights: {matWeights}")
print(f"Final bias: {matBias}")

Test(matBestWeights, matBestBias)
FinalTest(matBestWeights, matBestBias)

log.Write_To_xlsx()
log.Open_xlsm("ML_log.xlsx")
log.Open_xlsm("Temp_Log.xlsx")

print("Time:", time.time() - start)