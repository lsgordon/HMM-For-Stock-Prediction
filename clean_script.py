import numpy as np
from copy import deepcopy
import csv

def splitting(line: str):
    split_string = deepcopy(line)
    
    return split_string

def main():
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

if __name__ == "__main__":
    main()