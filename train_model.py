import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Load the dataset
data = pd.read_csv("Iris (1).csv")

# Features and target
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train and save Linear kernel model ----
linear_model = SVC(kernel='linear', probability=True)
linear_model.fit(X_train, y_train)
with open('svm_linear.pkl', 'wb') as f:
    pickle.dump(linear_model, f)

# ---- Train and save Polynomial kernel model ----
poly_model = SVC(kernel='poly', degree=3, probability=True)
poly_model.fit(X_train, y_train)
with open('svm_poly.pkl', 'wb') as f:
    pickle.dump(poly_model, f)

# ---- Train and save RBF kernel model ----
rbf_model = SVC(kernel='rbf', probability=True)
rbf_model.fit(X_train, y_train)
with open('svm_rbf.pkl', 'wb') as f:
    pickle.dump(rbf_model, f)

print("✅ All three SVM models (linear, poly, rbf) trained and saved successfully.")
