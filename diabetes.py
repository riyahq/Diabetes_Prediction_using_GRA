import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import entropy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer , MinMaxScaler  # For Quantile Normalization

# Load Dataset
def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    features = data.drop(columns=["target"])
    target = data["target"]
    return features, target


# Step 1: Data Normalization (Min-Max Scaling)
def normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

# Calculate Gray Relational Coefficients (GRC)
def grey_relational_coefficient(norm_data, ref_sequence, rho=0.5):
    diff = np.abs(ref_sequence - norm_data)
    min_diff = np.min(diff, axis=0)
    max_diff = np.max(diff, axis=0)
    grc = (min_diff + rho * max_diff) / (diff + rho * max_diff)
    return grc


# Calculate Weights Using Correlation
def calculate_weights_correlation(features, target):
    """Calculate feature weights using correlation with the target."""
    correlations = features.corrwith(target)
    weights = np.abs(correlations) / np.abs(correlations).sum()  # Normalize
    print("\nCalculated Weights (from Correlation):")
    print(weights.values)
    return weights.values

def grey_relational_grade(grc,weights):
    """Calculate the Gray Relational Grade (GRG) for each sample using feature weights."""
    grg = np.dot(grc,weights)  # Weighted average
    return grg

# Handle Missing Values
def handle_missing_values(features):
    """Impute missing values in the dataset using mean imputation."""
    imputer = SimpleImputer(strategy='mean')  # Mean imputation
    features_imputed = imputer.fit_transform(features)
    features_imputed_df = pd.DataFrame(features_imputed, columns=features.columns)
    return features_imputed_df


def classify_using_nearest_grg(grg, target):
    """Classify each sample based on the nearest GRG value from class 0 or class 1."""
    grg_class_0 = grg[target == 0]  # GRG values of class 0
    grg_class_1 = grg[target == 1]  # GRG values of class 1

    predictions = []
    for grg_val in grg:
        # Find the closest GRG value from each class
        closest_0 = min(grg_class_0, key=lambda x: abs(x - grg_val))
        closest_1 = min(grg_class_1, key=lambda x: abs(x - grg_val))

        # Assign to the class with the closest GRG value
        predicted_class = 0 if abs(closest_0 - grg_val) < abs(closest_1 - grg_val) else 1
        predictions.append(predicted_class)

    return np.array(predictions)



#Main Function
def main(file_path):
    features, target = load_data(file_path)
    features_imputed = handle_missing_values(features)
    
    norm_data = normalize(features_imputed)
    
    # Reference sequence (max values of normalized data)
    ref_sequence = norm_data.max().values

    # Calculate GRC and GRG
    grc = grey_relational_coefficient(norm_data, ref_sequence)
    weights = calculate_weights_correlation(features,target)
    grg = grey_relational_grade(grc,weights)

    print("\nGray Relational Grades (GRG):")
    print(grg)

    # Save GRG values to CSV
    grg_df = pd.DataFrame({'GRG': grg})
    grg_df.to_csv("grg_values.csv", index=False)
    print("GRG values saved to grg_values.csv")


    # **Use Rule-Based Classification**
    predictions =classify_using_nearest_grg(grg,target)
    print(predictions)

    # Print Accuracy and Confusion Matrix
    acc = accuracy_score(target, predictions)
    conf_matrix = confusion_matrix(target, predictions)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)

# Run the script
if __name__ == "__main__":
    file_path = r"c:\Users\RIYA PC\Downloads\diabetes.csv"
    main(file_path)


    
