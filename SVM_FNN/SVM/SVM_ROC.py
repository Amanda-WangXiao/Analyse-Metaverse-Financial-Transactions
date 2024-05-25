import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

# Load the dataset
file_path = '../metaverse_transactions_dataset.csv'
dataset = pd.read_csv(file_path)

# Convert timestamp to datetime and extract year and month
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
dataset['year'] = dataset['timestamp'].dt.year
dataset['month'] = dataset['timestamp'].dt.month

# Encode categorical features
categorical_features = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group', 'anomaly']
label_encoders = {feature: LabelEncoder() for feature in categorical_features}

for feature in categorical_features:
    dataset[feature] = label_encoders[feature].fit_transform(dataset[feature])

# Select features and target
features = ['hour_of_day', 'amount', 'transaction_type', 'location_region', 'ip_prefix',
            'login_frequency', 'session_duration', 'purchase_pattern', 'age_group',
            'risk_score', 'year', 'month']
target = 'anomaly'

X = dataset[features]
y = dataset[target]

# Binarize the output
y_bin = label_binarize(y, classes=list(range(len(label_encoders['anomaly'].classes_))))
n_classes = y_bin.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42, probability=True)
y_score = np.zeros((X_test.shape[0], n_classes))
y_pred = np.zeros((X_test.shape[0], n_classes))

for i in range(n_classes):
    svm_model.fit(X_train, y_train[:, i])
    y_score[:, i] = svm_model.predict_proba(X_test)[:, 1]
    y_pred[:, i] = svm_model.predict(X_test)

# Output classification report
y_test_inversed = label_encoders['anomaly'].inverse_transform(y_test.argmax(axis=1))
y_pred_inversed = label_encoders['anomaly'].inverse_transform(y_pred.argmax(axis=1))

report = classification_report(y_test_inversed, y_pred_inversed, target_names=label_encoders['anomaly'].classes_)
print("Classification Report:")
print(report)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for each class
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for multi-class')
plt.legend(loc='lower right')
plt.show()
