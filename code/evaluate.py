import csv
import pandas as pd


# Function to calculate sensitivity and specificity

def calculate_metrics(csv_file):
    # Initialize counts
    TP = TN = FP = FN = 0

    # Read CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header if there is one
        for row in reader:
            predicted, ground_truth = int(row[1]), int(row[2])
            if predicted == 1 and ground_truth == 1:
                TP += 1
            elif predicted == 0 and ground_truth == 0:
                TN += 1
            elif predicted == 1 and ground_truth == 0:
                FP += 1
            elif predicted == 0 and ground_truth == 1:
                FN += 1

    # Calculate sensitivity and specificity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    acc = (TP + TN)/(TP + TN + FP + FN)

    return sensitivity, specificity, acc


# Example usage
# Update this path to your CSV file
csv_file = '/home/yz2337/project/multi_fmri/results/fold1.v0/fold1.v0.csv'
sensitivity, specificity, acc = calculate_metrics(csv_file)
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Accuracy: {acc}")
