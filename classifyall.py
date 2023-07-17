import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.4)
        self.dense1 = nn.Linear(2048, 256)  # Adjusted input size
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, 128)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        self.dense3 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = nn.ReLU()(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = nn.ReLU()(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.batchnorm3(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.batchnorm4(x)
        x = nn.ReLU()(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        return x

# Preprocess the data
def preprocess_data(X, y):
    # Normalize the data
    X = (X - np.mean(X)) / np.std(X)

    # Convert to torch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return X, y

# Read the data
df_features = pd.read_csv('testdata.txt', delimiter=',', header=None)
df_labels = pd.read_csv('testlabels.txt', header=None)
df_test = pd.read_csv('testdata.txt', delimiter=',', header=None)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_features.values,
    df_labels.values.ravel(),
    test_size=0.3,
    random_state=42
)

# Define some constants
INPUT_SHAPE = (71, 1)  # Number of input features and channels
NUM_CLASSES = 10  # Number of output classes (0-9)
LEARNING_RATE = 0.025  # Adjust as necessary
BATCH_SIZE = 64
NUM_EPOCHS = 30

# Preprocess the data
X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)

# Create the CNN model
model = CNNClassifier()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    X_train, y_train = shuffle(X_train, y_train)  # Shuffle the training data
    num_batches = len(X_train) // BATCH_SIZE

    for batch in range(num_batches):
        start = batch * BATCH_SIZE
        end = start + BATCH_SIZE

        # Forward pass
        outputs = model(X_train[start:end].unsqueeze(1))  # Add unsqueeze(1) to match input shape
        loss = criterion(outputs, y_train[start:end])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss every few epochs
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test.unsqueeze(1))  # Add unsqueeze(1) to match input shape
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(predicted)

print(f'Test Accuracy: {accuracy:.4f}')

# Save predicted values as a PDF file
pdf_filename = 'predicted_values.pdf'
with PdfPages(pdf_filename) as pdf:
    classes, counts = np.unique(predicted, return_counts=True)
    plt.bar(classes, counts)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Count')
    plt.title('Predicted Label Distribution')
    pdf.savefig()
    plt.close()

print(f'Predicted values saved as {pdf_filename}')

with open('testlables.txt','w') as f:
    for predict in predicted:
        f.write(str(predict) + '\n')
          