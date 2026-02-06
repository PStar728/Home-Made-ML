from data import DataSet
import model
import log
import json

print("=== ML Training Session ===")
Weights = model.load_weights()
print(f"Initial weights: {Weights}")
print("Loading data...")
DATA = DataSet()
DATA.load_from_csv("winequality-red.csv", 100, 200)
print(f"Loaded {len(DATA.samples)} samples\n")

epochs = int(input("How many times should I train with this dataset? "))
EpochNum = 0
ErrorList = []
for epoch in range(epochs):
    print(f"--- Epoch {epoch + 1} ---")
    Display = log.Display_Check(epoch + 1)
    for i, point in enumerate(DATA.samples, 1):
        old_weights = Weights.copy()
        Weights = model.train_model(point, Weights)
        diff = model.predict(point, old_weights)
        print(f"  Sample {i}: Quality={point.quality}, Error={diff:.4f}")
        # write to tempholder.txt
        if (Display == True):
            # define an array type of all the errors for epoch
            # this will be for "Epoch Data" sheet
            ErrorList.append(diff)
            # pass array into write to xlsx to compute correct datas
            log.Write_To_txt(Weights, i, point.quality, diff)
    print()
    # write to ML Log.xlsx
    if (Display == True):
        log.Write_To_xlsx(epoch, ErrorList)
        # clear .txt file
        ErrorList = []
        with open("Temp_Holder.txt", "w") as f:
            f.write("")
            
    
            
model.save_weights(Weights)

print("\n=== Training Complete ===")
print(f"Final weights: {Weights}")


