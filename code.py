import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df=pd.read_csv("data.csv")

df.head()

df.info()

df.isna().sum()

df.shape

df.fillna(0, inplace=True)

df.shape

df.describe()

df['diagnosis'].value_counts()

sns.countplot(x='diagnosis', data=df, label="count")

# Show the plot
plt.show()

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.head()

sns.pairplot(df.iloc[:,1:5],hue="diagnosis")

plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:15].corr(),annot=True,fmt=".0%")

X=df.iloc[:,2:32].values 
Y=df.iloc[:,1].values 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (1 / m) * np.sum((-y * np.log(h)) - ((1 - y) * np.log(1 - h)))
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    costs = []

    for _ in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1 / m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        costs.append(cost)

    return theta, costs
# Add a column of ones to X_train for the bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Initialize the parameters
theta = np.zeros(X_train.shape[1])

# Set the learning rate and number of iterations
alpha = 0.01
num_iterations = 1000

# Perform gradient descent
theta, costs = gradient_descent(X_train, Y_train, theta, alpha, num_iterations)

# Predict labels for training data
X_train_prediction = sigmoid(X_train.dot(theta))
X_train_prediction = np.round(X_train_prediction).astype(int)


# Calculate the training accuracy
training_data_accuracy = np.mean(X_train_prediction == Y_train)

# Print the training accuracy
print('accuracy on training data =', training_data_accuracy)

X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
# Predict labels for test data
X_test_prediction = sigmoid(X_test.dot(theta))
X_test_prediction = np.round(X_test_prediction).astype(int)

# Calculate the test accuracy
testing_data_accuracy = np.mean(X_test_prediction == Y_test)

# Print the test accuracy
print('accuracy on testing data =', testing_data_accuracy)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(input_data, theta):
    input_data = np.hstack((np.ones((input_data.shape[0], 1)), input_data))
    predictions = sigmoid(input_data.dot(theta))
    predictions = np.round(predictions).astype(int)
    return predictions

# Define the input data
input_data = np.array([17.14,16.4,116,912.7,0.1186,0.2276,0.2229,0.1401,0.304,0.07413,1.046,0.976,7.276,111.4,0.008029,0.03799,0.03732,0.02397,0.02308,0.007444,22.25,21.4,152.4,1461,0.1545,0.3949,0.3853,0.255,0.4066,0.1059])

# Reshape the input data
input_data_reshaped = input_data.reshape(1, -1)

# Perform prediction using the trained model
predictions = predict(input_data_reshaped, theta)

if predictions[0] == 1:
    print('The breast tumor is Malignant')
else:
    print('The breast tumor is Benign')

import tkinter as tk
from tkinter import messagebox
import numpy as np

# Create the main application window
root = tk.Tk()
root.title("Breast Cancer Classification")
root.geometry("1500x400")
root.configure(bg="#FFD6E7")
heading_label = tk.Label(root, text="Breast Cancer Prediction", font=("Helvetica", 20, "bold"), bg="#FFD6E7", fg="#333333")
heading_label.pack(pady=20)
# Create a frame for input fields
input_frame = tk.Frame(root, bg="#FFD6E7")
input_frame.pack(pady=20)

# Create labels and entry fields for input data
input_labels = ['Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean', 'Compactness_mean', 
                'Concavity_mean', 'Concave_points_mean', 'Symmetry_mean','Fractal_dimension_mean', 'Radius_se', 'Texture_se',
                'Perimeter_se', 'Area_se', 'Smoothness_se', 'Compactness_se', 'Concavity_se', 'Concave_points_se',          
                'Symmetry_se', 'Fractal_dimension_se', 'Radius_worst', 'Texture_worst', 'Perimeter_worst', 'Area_worst', 
                'Smoothness_worst', 'Compactness_worst', 'Concavity_worst', 'Concave_points_worst', 'Symmetry_worst', 
                'Fractal_dimension_worst']

input_entries = []
for i, label in enumerate(input_labels):
    row = i // 5
    col = i % 5
    
    label = tk.Label(input_frame, text=label + ":", bg="#FFD6E7", fg="#333333", font=("Helvetica", 12))
    label.grid(row=row, column=2*col, sticky="e", padx=5, pady=5)

    entry = tk.Entry(input_frame, width=10, font=("Helvetica", 12))
    entry.grid(row=row, column=2*col+1, padx=5, pady=5)
    input_entries.append(entry)

# Create a prediction function
def predict_cancer():
    input_data = [float(entry.get()) for entry in input_entries]
    input_data = np.array(input_data)

    # Reshape the input data
    input_data_reshaped = input_data.reshape(1, -1)

    # Perform prediction using the trained model
    predictions = predict(input_data_reshaped, theta)

    if predictions[0] == 1:
        messagebox.showinfo("Prediction Result", "The breast tumor is Malignant", icon="warning", bg="#FFD6E7", fg="#333333")
    else:
        messagebox.showinfo("Prediction Result", "The breast tumor is Benign", icon="info", bg="#FFD6E7", fg="#333333")

# Create a predict button

style = ttk.Style()
style.configure("TButton",
                foreground="#FFFFFF",
                background="#FF69B4",
                font=("Helvetica", 12),
                padding=(10, 5))

predict_button = ttk.Button(root, text="Classify", command=predict_cancer, style="TButton")
predict_button.pack(pady=10)
# Start the main event loop
root.mainloop()
