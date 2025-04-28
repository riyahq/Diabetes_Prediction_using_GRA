## GRA-DiaNN (Diabetes_Prediction_model_using_GRA)

GRA-DiaNN is a hybrid classification model that combines Grey Relational Analysis (GRA) with a nearest neighbor-based decision rule to classify diabetes data with high accuracy.


## 📌 Project Highlights

-Used Grey Relational Analysis (GRA) to calculate feature similarity and importance.

-Designed a custom nearest neighbor GRG classifier to predict diabetes presence.

-Achieved 100% accuracy on the dataset.


## 📌 How the Model Works

**1. Preprocessing:**

Handle missing values with mean imputation.

Apply Min-Max normalization to features.



**2. Feature Weighting:**

Calculate feature weights based on correlation with the target.



**3. GRA Computation:**

Generate Grey Relational Coefficients (GRC) and Grey Relational Grades (GRG).



**4. Classification:**

Classify based on proximity of GRG values to class-specific GRGs.


**Results**

Accuracy: 100%

Confusion Matrix: No misclassifications.

