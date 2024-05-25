import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
file_path = '../metaverse_transactions_dataset.csv'
data = pd.read_csv(file_path)

# 数据预处理
# 删除时间戳列
data = data.drop(columns=['timestamp'])

# 处理类别特征
label_encoders = {}
for column in ['sending_address', 'receiving_address', 'transaction_type', 'location_region', 'purchase_pattern', 'age_group']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 将 anomaly 列转换为多分类
anomaly_mapping = {'low_risk': 0, 'moderate_risk': 1, 'high_risk': 2}
data['anomaly'] = data['anomaly'].map(anomaly_mapping)

# 标准化数值特征
scaler = StandardScaler()
numerical_columns = ['amount', 'login_frequency', 'session_duration', 'risk_score', 'ip_prefix']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# 确保所有特征都是数值类型
data = data.apply(pd.to_numeric)

# 划分数据集
X = data.drop(columns=['anomaly'])
y = data['anomaly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 构建FNN模型
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = FNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 保存每个epoch的训练损失和测试准确率
train_losses = []
test_accuracies = []

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    test_accuracies.append(accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Test Accuracy: {accuracy:.4f}')


# 生成分类报告和混淆矩阵
def generate_report_and_confusion_matrix(loader, dataset_type='Test'):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(predicted.numpy())

    report = classification_report(y_true, y_pred, target_names=['low_risk', 'moderate_risk', 'high_risk'])
    print(f'{dataset_type} Classification Report:')
    print(report)

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['low_risk', 'moderate_risk', 'high_risk'],
                yticklabels=['low_risk', 'moderate_risk', 'high_risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{dataset_type} Confusion Matrix')
    plt.show()


print("X_train.shape[1]:")
print(X_train.shape[1])

# 生成测试集的报告和混淆矩阵
generate_report_and_confusion_matrix(test_loader, 'Test')

# 生成训练集的报告和混淆矩阵
generate_report_and_confusion_matrix(train_loader, 'Train')
