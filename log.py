from openpyxl import Workbook, load_workbook
from datetime import datetime
import os
import subprocess

# checks if epoch number == (1, 2, 5, 10, 20, 50...)
def Display_Check(Epoch):
    while Epoch % 10 == 0:
        Epoch //= 10
        if (Epoch == 0):
            break
        
    if (Epoch in (1, 2, 5)):
        return True
    
    return False

def Write_To_xlsx(EpochNum, ErrorList):
    file_path_xlsx = "ML_log.xlsx"
    file_path_txt = "Temp_Holder.txt"
    
    wb = load_workbook(file_path_xlsx)
    sheet = wb["Temp Dump"]
    
    # formatting
    sheet.append([f"Epoch {EpochNum + 1}", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "Bias", "Quality", "Prediction", "Error"])
    
    # adds the data
    with open(file_path_txt, "r") as txt:
        for line in txt:
            line = line.strip()
            columns = [col.strip() for col in line.split(",")]
            
            # convert numbers to float if necessary
            for i in range(1, len(columns)):
                columns[i] = float(columns[i])
            sheet.append(columns)

    Epoch_Data_Line = Calculate_Epoch_Data((EpochNum + 1), ErrorList)

    sheet = wb["Epoch Data"]

    # formatting
    if EpochNum == 0:
        sheet.append([f"Epoch #", "Min Error", "Quartile 1", "Median Error", "Quartile 3", "Max Error", "IQR", "Error Range", "Mean Error", "STDV"])

    sheet.append(Epoch_Data_Line)

    wb.save(file_path_xlsx)

    
def Write_To_txt(Weights, Bias, DataPoint, Quality, Error):
    file_path = "Temp_Holder.txt"
    
    with open(file_path, "a") as f:
        weights_str = ", ".join(str(w) for w in Weights)
        prediction = Quality - Error
        
        f.write(
            f"{DataPoint}, {weights_str}, {Bias}, {Quality}, {prediction}, {Error}\n"
        ) 

def Calculate_Epoch_Data(EpochNum, ErrorList):
    n = len(ErrorList)

    ABSerror = [abs(e) for e in ErrorList]
    errors = sorted(ABSerror)  # do NOT mutate original list

    total = sum(errors)
    mean = total / n

    # Median
    mid = n // 2
    if n % 2 == 1:
        median = errors[mid]
    else:
        median = (errors[mid - 1] + errors[mid]) / 2

    min_err = errors[0]
    max_err = errors[-1]
    err_range = max_err - min_err

    # Population standard deviation
    variance = sum((x - mean) ** 2 for x in errors) / n
    stdv = variance ** 0.5

    # Quartiles (simple)
    q1 = errors[n // 4]
    q3 = errors[(n * 3) // 4]
    iqr = q3 - q1

    return [
        EpochNum,
        min_err,
        q1,
        median,
        q3,
        max_err,
        iqr,
        err_range,
        mean,
        stdv
    ]
def Log_Tests(testErrors):
    file_path = "ML_log.xlsx"

    wb = load_workbook(file_path)
    sheet = wb["Epoch Data"]

    sheet.append(testErrors)
    wb.save(file_path)

def Open_xlsm(file_path):
    subprocess.call(["open", file_path])

def Save_Close_Open(file_path):
    script = f'''
    tell application "Microsoft Excel"
        if it is running then
            repeat with wb in workbooks
                if (full name of wb) is "{file_path}" then
                    save wb
                    close wb
                end if
            end repeat
        end if
    end tell
    '''

    subprocess.run(["osascript", "-e", script])