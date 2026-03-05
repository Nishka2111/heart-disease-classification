import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("heart.csv")  # update path as needed

# Drop rows with missing values
df = df.dropna()

# Convert target to binary: 0 = no disease, 1 = has disease
df['label'] = (df['num'] > 0).astype(int)

# Select two features for plotting and modeling
X_real = df[['thalach', 'oldpeak']].values
y_real = df['label'].values

print("X_real shape:", X_real.shape)
print("y_real class distribution:", np.bincount(y_real))

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(X_real[y_real==0, 0], X_real[y_real==0, 1], label='No Disease', alpha=0.8)
plt.scatter(X_real[y_real==1, 0], X_real[y_real==1, 1], label='Disease', alpha=0.8)

plt.xlabel("Thalach (max heart rate)")
plt.ylabel("Oldpeak (ST depression)")
plt.title("Heart Disease Dataset (2 Features)")
plt.legend()
plt.grid(True)
plt.show()

# Random initialize weights and bias
w_real = np.random.randn(X_real.shape[1])  # shape (2,)
b_real = np.random.randn()


# Training settings
lr = 0.1
epochs = 50
errors_real = []

# Step function
def step(z): return 1 if z >= 0 else 0

# Training loop
for epoch in range(epochs):
    total_errors = 0
    for xi, yi in zip(X_real, y_real):
        z = np.dot(w_real, xi) + b_real
        y_hat = step(z)

        error = yi - y_hat
        if error != 0:
            w_real += lr * error * xi
            b_real += lr * error
            total_errors += 1
    errors_real.append(total_errors)

print("Final weights:", w_real)
print("Final bias:", b_real)

plt.plot(range(1, epochs+1), errors_real, marker='o')
plt.title('Real Data: Number of Misclassifications Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.grid(True)
plt.show()

import numpy as np

# Step function vectorized
def step_function(z):
    return (z >= 0).astype(int)

# Compute predictions
z = X_real @ w_real + b_real
y_pred = step_function(z)

# Compute accuracy
correct_predictions = np.sum(y_pred == y_real)
total_predictions = len(y_real)

expected_accuracy = correct_predictions / total_predictions

print(f" Expected Accuracy: {expected_accuracy:.2f}")

# Initialize weights (same as Section 2)
np.random.seed(42)
W1_real = np.random.randn(2, 3)    # (2 → 3)
b1_real = np.random.randn(1, 3)    # (1, 3)
W2_real = np.random.randn(3, 1)    # (3 → 1)
b2_real = np.random.randn(1, 1)    # (1, 1)

# ReLU function
def relu(z):
    return np.maximum(0, z)

# Forward pass (no training!)
z1 = np.dot(X_real, W1_real) + b1_real
h1 = relu(z1)
y_logits = np.dot(h1, W2_real) + b2_real

# Create a grid over standardized [0, 1] feature space
xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Forward pass
z1_grid = np.dot(grid_points, W1_real) + b1_real
h1_grid = relu(z1_grid)
y_grid_logits = np.dot(h1_grid, W2_real) + b2_real
y_grid_logits = y_grid_logits.reshape(xx.shape)

# Plot
plt.figure(figsize=(7, 5))

# Contour plot of raw logits
contour = plt.contourf(xx, yy, y_grid_logits, cmap='coolwarm', alpha=0.8, levels=30)

# Decision surface with data points
plt.scatter(X_real[y_real == 0][:, 0], X_real[y_real == 0][:, 1], label='No Disease', alpha=0.7)
plt.scatter(X_real[y_real == 1][:, 0], X_real[y_real == 1][:, 1], label='Disease', alpha=0.7)

# Axes and labels
plt.xlabel('Standardized Thalach')
plt.ylabel('Standardized Oldpeak')
plt.title('Real Data + Raw Output of Random FNN')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

# Explicit colorbar linked to contour plot
plt.colorbar(contour, label="Raw Logit Output")
plt.show()

# Forward pass functions
def relu(z):
    return np.maximum(0, z)

# Step 1: Hidden layer linear transformation
z1 = X_real @ W1_real + b1_real

# Step 2: Apply ReLU activation
h1 = relu(z1)

# Step 3: Compute output logits
logits = h1 @ W2_real + b2_real

# Step 4: Convert logits to class predictions
y_pred = (logits >= 0).astype(int)

# Step 5: Compute accuracy
model_accuracy = np.mean(y_pred.flatten() == y_real)

print(f"Model Accuracy: {model_accuracy:.4f}")

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Apply sigmoid to logits
y_grid_probs = sigmoid(y_grid_logits)

# Plot
plt.figure(figsize=(7, 5))
contour = plt.contourf(xx, yy, y_grid_probs, cmap='coolwarm', alpha=0.4, levels=30)
plt.contour(xx, yy, y_grid_probs, levels=[0.5], colors='black', linestyles='--')  # Decision boundary

# Scatter original data
plt.scatter(X_real[y_real == 0][:, 0], X_real[y_real == 0][:, 1], label='No Disease', alpha=0.7)
plt.scatter(X_real[y_real == 1][:, 0], X_real[y_real == 1][:, 1], label='Disease', alpha=0.7)

# Labels and formatting
plt.xlabel('Standardized Thalach')
plt.ylabel('Standardized Oldpeak')
plt.title('Real Data + Sigmoid Output of Random FNN')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.colorbar(contour, label="Probability of Disease")  # <- attach explicitly
plt.show()

import pandas as pd

# Load data
df = pd.read_csv("heart.csv")
print(df.shape)
print(df.head())

df.info()
df.isnull().sum()

df = df.dropna()
print("Remaining rows:", df.shape[0])

df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df['target'].value_counts()

X = df.drop(columns=['num', 'target'])
y = df['target']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
import torch

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(13, 16),   # Input layer → Hidden layer (13 → 16)
    nn.ReLU(),           # Activation
    nn.Linear(16, 1),    # Hidden → Output layer (16 → 1)
    nn.Sigmoid()         # Output activation for binary classification
)

# Show architecture
print(model)

sum(p.numel() for p in model.parameters())

import torch.nn as nn

# Alternative architecture (not trained in this experiment)
custom_model = nn.Sequential(
    nn.Linear(13, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 16),
    nn.Sigmoid(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

# Print model architecture
print(custom_model)

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

import torch.optim as optim

# Loss function: Binary Cross-Entropy
loss_fn = nn.BCELoss()

# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Loss function:", loss_fn)
print("Optimizer:", optimizer)

# Number of epochs
num_epochs = 200
train_losses = []

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save loss
    train_losses.append(loss.item())

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

num_epochs = 200
train_losses = []
test_losses = []  # Track test loss

for epoch in range(num_epochs):
    model.train()

    # Forward pass (training)
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save training loss
    train_losses.append(loss.item())

    # ===== Evaluate on test set =====
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = loss_fn(test_outputs, y_test)

    # Store test loss
    test_losses.append(test_loss.item())

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss.item():.4f}, "
              f"Test Loss: {test_loss.item():.4f}")
        
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, linestyle='--', marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")
plt.grid(True)
plt.show()

# predicting based on train set
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train)
    y_train_pred = (y_train_pred > 0.5).float()

# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

confusion_matrix(y_train, y_train_pred)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot confusion matrix
cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on train Set")
plt.grid(False)
plt.show()

from sklearn.metrics import accuracy_score

# Put the model in evaluation mode
model.eval()

# Turn off gradient tracking
with torch.no_grad():

    # Forward pass on test data
    outputs = model(X_test)

    # Compute test loss
    test_loss = loss_fn(outputs, y_test)

    # Convert probabilities to binary predictions
    preds = (outputs >= 0.5).float()

    # Compute test accuracy
    test_acc = accuracy_score(y_test.numpy(), preds.numpy())

# Final print
print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# predicting based on train set
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred = (y_test_pred > 0.5).float()

# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on Test Set")
plt.grid(False)
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))