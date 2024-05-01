import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Data collection
data = pd.read_csv("fraud_email_.csv")

# One-hot encoding the text
encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(data[['Text']])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Model architecture
model = LogisticRegression(max_iter=1000)

# Training the model
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Accuracy metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

