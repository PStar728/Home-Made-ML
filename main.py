import copy

from data import DataSet
from Test import Test
import model
import log

print("=== ML Training Session ===")
dataStart = 100
dataEnd = 200

DATA = DataSet()
DATA.load_from_csv("winequality-red.csv", dataStart, dataEnd)
print(f"Loaded {len(DATA.samples)} samples\n")

Weights = model.load_weights(len(DATA.samples[0].inputs))
Bias = model.load_bias()

print(f"Initial weights: {Weights}")
print("Loading data...")

epochs = int(input("How many times should I train with this dataset? "))
EpochNum = 0
ErrorList = []
AvgErrorList = []
AvgError = 0.5
BaselineError = .4
prevError = 100
prevWeights = None
prevBias = None

for epoch in range(epochs):
    print(f"--- Epoch {epoch + 1} ---")
    Display = log.Display_Check(epoch + 1)
    for i, point in enumerate(DATA.samples, 1):
        error, Weights, Bias = model.train_model(point, Weights, Bias, AvgError, BaselineError)
        print(f"  Sample {i}: Quality={point.quality}, Error={error:.4f}")

        # Calcs the recent avg error for LR
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
    print()
    # write to ML Log.xlsx
    if Display:
        log.Write_To_xlsx(epoch, ErrorList)
        # clear .txt file
        BaselineError = max((sum(ErrorList) / len(ErrorList)) - 0.1, 0.25)
        ErrorList = []
        with open("Temp_Holder.txt", "w") as f:
            f.write("")

        if EpochNum >= 50:
            testData = Test(dataStart, dataEnd, Weights, Bias)
            if testData[8] > prevError:
                Weights = prevWeights
                Bias = prevBias
                testData = Test(dataStart, dataEnd, Weights, Bias)
                break
            prevError = testData[8]
            prevWeights = copy.deepcopy(Weights)
            prevBias = copy.deepcopy(Bias)

            
model.save_weights(Weights)
model.save_bias(Bias)

print("\n=== Training Complete ===")
print(f"Final weights: {Weights}")

Test(dataStart, dataEnd, Weights, Bias)


