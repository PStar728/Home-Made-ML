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

def Write_To_Temp(EpochNum, ErrorList):
    file_path_temp = "Temp_Log.xlsx"
    file_path_txt = "Temp_Holder.txt"
    
    wb = try_temp_workbook(file_path_temp)
    sheet = wb["TEMP_DATA"]
    
    # formatting
    sheet.append([f"Epoch {EpochNum}", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "Bias", "Quality", "Prediction", "Error"])
    
    # adds the data
    with open(file_path_txt, "r") as txt:
        for line in txt:
            line = line.strip()
            columns = [col.strip() for col in line.split(",")]
            
            # convert numbers to float if necessary
            for i in range(1, len(columns)):
                columns[i] = float(columns[i])
            sheet.append(columns)

    Epoch_Data_Line = Calculate_Epoch_Data((EpochNum), ErrorList)

    sheet = wb["EPOCH_HOLDER"]

    # formatting
    if EpochNum == 1:
        sheet.append([f"Epoch #", "Min Error", "Quartile 1", "Median Error", "Quartile 3", "Max Error", "IQR", "Error Range", "Mean Error", "STDV"])

    sheet.append(Epoch_Data_Line)

    wb.save(file_path_temp)

def Write_To_xlsx():
    file_main = "ML_log.xlsx"
    file_temp = "Temp_Log.xlsx"
    file_stage = file_main + ".tmp"  # short-lived safety file

    # Load temp workbook (scratch file — allowed to break)
    wb_temp = load_workbook(file_temp)
    sheet_temp = wb_temp["EPOCH_HOLDER"]

    # Load main workbook (must be protected)
    wb_main = load_workbook(file_main)
    sheet_main = wb_main["Epoch Data"]

    # Append rows (skip header row from temp)
    for i, row in enumerate(sheet_temp.iter_rows(values_only=True)):
        if i == 0:
            continue
        sheet_main.append(row)

    # Save safely to staging file first
    wb_main.save(file_stage)

    wb_temp.close()
    wb_main.close()

    # Atomic replace protects main file
    os.replace(file_stage, file_main)


    
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
    file_path = "Temp_Log.xlsx"

    wb = load_workbook(file_path)
    sheet = wb["EPOCH_HOLDER"]

    sheet.append(testErrors)
    wb.save(file_path)

def Open_xlsm(file_path):
    subprocess.call(["open", file_path])


def Save_Close(file_path):
    """Save and close an Excel workbook on macOS using AppleScript."""
    file_name = os.path.basename(file_path)

    script = f'''
    tell application "Microsoft Excel"
        if it is running then
            if (name of workbooks) contains "{file_name}" then
                save workbook "{file_name}"
                close workbook "{file_name}"
            end if
        end if
    end tell
    '''

    subprocess.run(["osascript", "-e", script], capture_output=True, text=True)

def try_temp_workbook(file_path):
    try:
        wb = load_workbook(file_path)
    except Exception:
        # File missing or corrupted → recreate
        if os.path.exists(file_path):
            os.remove(file_path)

        wb = Workbook()
        wb.remove(wb.active)  # remove default sheet

        wb.create_sheet("TEMP_DATA")
        wb.create_sheet("EPOCH_HOLDER")

        wb.save(file_path)

    # Ensure required sheets exist
    required_sheets = ["TEMP_DATA", "EPOCH_HOLDER"]
    for name in required_sheets:
        if name not in wb.sheetnames:
            wb.create_sheet(name)

    return wb