import pandas as pd
from datetime import datetime

def find_corrections(a_csv, b_csv):
    try:
    # Read CSV files into pandas DataFrames
        dataframe_of_reference = pd.read_csv(a_csv)
        dataframe_of_input = pd.read_csv(b_csv)

        # Find differences between the two DataFrames
        differences = []
        for row in range(len(dataframe_of_reference)):
            for coloumn in range(len(dataframe_of_reference.columns)):
                if dataframe_of_reference.iloc[row, coloumn] != dataframe_of_input.iloc[row, coloumn]:
                    differences.append(((row+1, coloumn+1), dataframe_of_reference.iloc[row, coloumn], dataframe_of_input.iloc[row, coloumn]))
        # printing to console in case no errors are found
        if len(differences) == 0:
            return print("No errors found")
        else:
        # Write differences to corrections.csv
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"assignment_week_4/{current_datetime}_corrections.csv", 'w') as f:
                f.write("Location (row_number, col_number), Actual Value, Incorrect Value\n")
                print(differences)
                for diff in differences:
                    location = diff[0]
                    actual_value = diff[1]
                    incorrect_value = diff[2]
                    f.write(f"{location}, {actual_value}, {incorrect_value}\n")
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
find_corrections("assignment_week_4/a.csv", "assignment_week_4/b.csv")