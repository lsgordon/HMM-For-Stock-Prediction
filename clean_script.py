import numpy as np
from copy import deepcopy
import csv
import math

def splitting(line: str):
    split_string = deepcopy(line)
    
    return split_string

def clean_csv(input_file, output_file):
    """
    Removes rows with blank entries from a CSV file.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the cleaned CSV file.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header row
        rows = [row for row in reader if all(cell.strip() for cell in row)]  # Keep rows with no blank entries

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header row
        writer.writerows(rows)  # Write the filtered rows
    
    num_data_path = "Data/num_data.csv"
    with open(num_data_path, 'w', newline='', encoding='utf-8') as numfile:
        writer = csv.writer(numfile)
        reduced_rows = [row[5:] for row in rows if len(row) > 5 and not any(math.isnan(float(entry)) for entry in row[5:])]
        # reduced_rows = [row[5:] for row in rows if len(row) > 5]  # Exclude first 5 columns; unfortunate presence of NaN value in data, hence above line
        writer.writerows(reduced_rows)  # Write the reduced rows

def main():
    # Usage
    input_csv = "Data/stock_data_for_lin_reg.csv"  # Replace with your input CSV file path
    output_csv = "Data/cleaned.csv"  # Replace with your desired output CSV file path
    clean_csv(input_csv, output_csv)
    '''
    data = []
    print("" in "what")
    old_data = open("Data/stock_data_for_lin_reg.csv", 'r')
    for line in old_data:
        tokens = line.strip().split(",")
        if "" not in tokens:
            data.append(tokens)
    # print(data)
    with open("Data/cleaned.csv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(data)
        pass
    '''

if __name__ == "__main__":
    main()