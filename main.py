import copy

from data import DataSet
from Test import Test
import model
import log

#saves and closes ML_log.xlsx
log.Save_Close("ML_log.xlsx")

#defines training data points
print("=== ML Training Session ===")
dataStart = 0
dataEnd = 1200
DATA = DataSet()
DATA.load_from_csv("winequality-red.csv", dataStart, dataEnd)
print(f"Loaded {len(DATA.samples)} samples\n")

#defines weights and biases
Weights = model.load_weights(len(DATA.samples[0].inputs))
Bias = model.load_bias()

print(f"Initial weights: {Weights}")
print("Loading data...")

#epochs = int(input("How many times should I train with this dataset? "))

#initializes variables for the main
ErrorList = []
AvgErrorList = []
AvgError = 0.5
BaselineError = .4
prevError = 100
bestWeights = None
bestBias = None
strikes = 0
patience = 2
bestError = float('inf')
currentError = 0

#starts the main loop
#loop define to be inf. until the comment is deleted
epoch = 0
while epoch + 1:# < 100000:
    #print(f"--- Epoch {epoch} ---")

    # Display checks if epoch == (1, 2, 5, 10, 20, 50, 100...)
    # Display is used for thing that should only be done periodically not every epoch
    Display = log.Display_Check(epoch)

    # loops through every data point
    for i, point in enumerate(DATA.samples, 1):
        error, Weights, Bias = model.train_model(point, Weights, Bias, AvgError, BaselineError, currentError)
        # print(f"  Sample {i}: Quality={point.quality}, Error={error:.4f}")

        # Calcs the recent avg error for dynamic LR
        AvgErrorList.append(abs(error))
        if len(AvgErrorList) > 20:
            AvgErrorList.pop(0)
        AvgError = sum(AvgErrorList) / len(AvgErrorList)

        # write to tempholder.txt
        if Display:
            # define an array type of all the errors for epoch
            # this will be for "Epoch Data" sheet
            ErrorList.append(error)
            # pass array into write to xlsx to compute correct datas
            log.Write_To_txt(Weights, Bias, i, point.quality, error)
    # print()
    # write to ML tmep Log
    if Display:
        log.Write_To_Temp(epoch, ErrorList)
        # clear .txt file
        BaselineError = max((sum(ErrorList) / len(ErrorList)) - 0.1, 0.25)
        ErrorList = []
        with open("Temp_Holder.txt", "w") as f:
            f.write("")

        testData = Test(dataStart, dataEnd, Weights, Bias)
        currentError = testData[8]

        print(f"Epoch: {epoch}")
        if currentError < bestError:
            # NEW BEST! Save and reset
            bestError = currentError
            bestWeights = copy.deepcopy(Weights)
            bestBias = copy.deepcopy(Bias)
            strikes = 0

            print(f"✓ New best: {currentError:.4f}")

        else:
            # Didn't beat best
            if epoch > 100:
                strikes += 1
            print(f"✗ No improvement: {currentError:.4f} (strike {strikes}/{patience})")

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