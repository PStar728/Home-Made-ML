import copy

from data import DataSet
from model import boundaries
from Test import Test, FinalTest
from Janitor import Clean
import modelNN
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
neuronsL1: int = 16
DATA:DataSet = DataSet()
DATA.load_from_csv("wine_shuffled.csv", dataStart, dataEnd, True)

print(f"Loaded {len(DATA.samples)} samples\n")

#defines weights and biases
#Weights: list = model.load_weights(len(DATA.samples[0].inputs), neuronsL1, "W1")
#bins and outputs are hardcoded
# definition changes for NN
mW0, mW1, mB0, mB1, janMat0, janMat1 = modelNN.load_brain(len(DATA.samples[0].inputs), neuronsL1, 1, 13)
#Bias: list = model.load_bias()

print(f"Initial weights: {mW1}")
print("Loading data...")

#epochs = int(input("How many times should I train with this dataset? "))

bins = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

# redefines as a numpy matrix
Inputs: np.ndarray = np.array([dp.inputs for dp in DATA.samples])
Inputs = Inputs - Inputs.mean(axis=0)
Quality: np.ndarray = np.array([dp.quality for dp in DATA.samples]).reshape(-1, 1)
Weirdness: np.ndarray = np.array(DATA.weirdness).reshape(-1, 1)
#matWeights: np.ndarray = np.array(Weights).reshape(-1, 1)
#matBias: np.ndarray = np.array([Bias]).reshape(-1,1)
matError: np.ndarray = np.array(None)
matBin: np.ndarray = np.digitize(Weirdness.flatten(), bins)
matBin = matBin.reshape(-1, 1)

#initializes variables for the main
ErrorList: list = []
AvgErrorList: list = []
prevError: float = 0.5
BaselineError: float = .4
mBestW0: np.ndarray = None
mBestW1: np.ndarray = None
mBestB0: np.ndarray = None
mBestB1: np.ndarray = None
strikes: int = 0
patience: int = 10
bestError: float = float('inf')
currentTestError: float = 0
Base_LR: float = .07

#starts the main loop
#loop define to be inf. until the comment is deleted
epoch: int = 1
while epoch <= 300000:
    #print(f"--- Epoch {epoch} ---")

    # Display checks if epoch == (1, 2, 5, 10, 20, 50, 100...)
    # Display is used for anything that should only be done periodically not every epoch
    Display = log.Display_Check(epoch)
    #t0 = time.time()
    matError, mW0, mW1, mB0, mB1 = modelNN.train_model(Inputs, Quality, mW0, mW1, mB0, mB1, prevError, BaselineError, currentTestError, epoch, matBin, janMat0, janMat1, Base_LR)
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
        #print("Bias: ", matBias)
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

        testData = Test(mW0, mW1, mB0, mB1)
        FinalTest(mW0, mW1, mB0, mB1)
        currentTestError = testData[8]

        #Weights = matWeights.flatten().tolist()
        #Bias = matBias.flatten().tolist()

        print(f"Epoch: {epoch}")
        if currentTestError < bestError:
            # NEW BEST! Save and reset
            bestError = currentTestError
            mBestW0 = mW0.copy()
            mBestW1 = mW1.copy()
            mBestB0 = mB0.copy()
            mBestB1 = mB1.copy()
            strikes = 0

            print(f"✓ New best: {currentTestError:.4f}")

        else:
            # Didn't beat best
            print(f"janMat shape: {janMat0.shape}")
            if ((strikes >= 1) & (epoch >= 500)):
                mW0 = mBestW0.copy()
                mW1 = mBestW1.copy()
                mB0 = mBestB0.copy()
                mB1 = mBestB1.copy()

                print("running janitor!")
                janMat0, Base_LR, mW0, mW1 = Clean(janMat0, janMat1, mW0, mW1, mB0, mB1, Inputs, Quality, matBin, Base_LR, epoch)
                #Base_LR = Janitor.UpdateB_LR(Base_LR)

                mBestW0 = mW0.copy()
                mBestW1 = mW1.copy()
                mBestB0 = mB0.copy()
                mBestB1 = mB1.copy()
            strikes += 1
                #break

            print(f"✗ No improvement: {currentTestError:.4f} (strike {strikes}/{patience})")

            if strikes >= patience:
                mW0 = mBestW0.copy()
                mW1 = mBestW1.copy()
                mB0 = mBestB0.copy()
                mB1 = mBestB1.copy()
                print(f"Early stopping! Restored best: {bestError:.4f}")
                break
    epoch += 1

modelNN.save_brain(mW1, mW1, mB0, mB1, janMat0, janMat1)

print("\n=== Training Complete ===")
#print(f"Final weights: {matWeights}")
#print(f"Final bias: {matBias}")

Test(mBestW0, mBestW1, mBestB0, mBestB1)
FinalTest(mBestW0, mBestW1, mBestB0, mBestB1)

log.Write_To_xlsx()
log.Open_xlsm("ML_log.xlsx")
log.Open_xlsm("Temp_Log.xlsx")

print("Time:", time.time() - start)