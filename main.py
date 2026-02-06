from data import DataSet
import model
import log

print("=== ML Training Session ===")
Weights = model.load_weights()
Bias = model.load_bias()

print(f"Initial weights: {Weights}")
print("Loading data...")
DATA = DataSet()
DATA.load_from_csv("winequality-red.csv", 100, 200)
print(f"Loaded {len(DATA.samples)} samples\n")

epochs = int(input("How many times should I train with this dataset? "))
EpochNum = 0
ErrorList = []
AvgErrorList = []

for epoch in range(epochs):
    print(f"--- Epoch {epoch + 1} ---")
    Display = log.Display_Check(epoch + 1)
    for i, point in enumerate(DATA.samples, 1):
        error, Weights, Bias = model.train_model(point, Weights, Bias)
        print(f"  Sample {i}: Quality={point.quality}, Error={error:.4f}")

        # Calcs the recent avg error for LR
        AvgErrorList.append(abs(error))
        if AvgErrorList.len > 20:
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
        ErrorList = []
        with open("Temp_Holder.txt", "w") as f:
            f.write("")
            
    
            
model.save_weights(Weights)
model.save_bias(Bias)

print("\n=== Training Complete ===")
print(f"Final weights: {Weights}")


