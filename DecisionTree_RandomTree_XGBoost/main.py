import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, auc

from sklearn.preprocessing import label_binarize



# 加载数据
data = pd.read_csv('metaverse_transactions_dataset.csv')

# 数据预处理
data.ffill(inplace=True)

# 检查 'risk_score' 列是否存在，并移除
if 'risk_score' in data.columns:
    data = data.drop('risk_score', axis=1)
else:
    print("Colulamn 'risk_score' not found in the dataset.")

# 处理时间戳
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['Year'] = data['timestamp'].dt.year
    data['Month'] = data['timestamp'].dt.month
    data['Day'] = data['timestamp'].dt.day
    data['Hour'] = data['timestamp'].dt.hour
    data = data.drop('timestamp', axis=1)

# 编码处理
for column in data.columns:
    if data[column].dtype == type(object):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

# 划分特征和目标变量
# 目标变量二值化处理！
X = data.drop('anomaly', axis=1)
y = label_binarize(data['anomaly'], classes=[0, 1, 2])


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 决策树模型

dt_model = DecisionTreeClassifier(
    max_depth=5,                  # 限制最大深度
    min_samples_split=20,         # 增加分割所需的最小样本数
    min_samples_leaf=10,          # 增加叶节点的最小样本数
    random_state=42
)

#dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=50, min_samples_leaf=25, random_state=42)
dt_model.fit(X_train, y_train)
#测试集
dt_test_pred = dt_model.predict(X_test)
dt_test_accuracy = accuracy_score(y_test, dt_test_pred) #这里改train还是test
#dt_report = classification_report(y_test, dt_pred)
dt_test_report = classification_report(y_test, dt_test_pred, target_names=['high_risk', 'low_risk', 'moderate_risk'])
#训练集
dt_train_pred = dt_model.predict(X_train)
dt_train_accuracy = accuracy_score(y_train, dt_train_pred)
dt_train_report = classification_report(y_train, dt_train_pred, target_names=['high_risk', 'low_risk', 'moderate_risk'])

# 随机森林模型
#rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
#                                  min_samples_leaf=2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=4, bootstrap=True, random_state=42)
rf_model.fit(X_train, y_train)
rf_test_pred = rf_model.predict(X_test)
rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
#rf_report = classification_report(y_test, rf_pred)
rf_test_report = classification_report(y_test, rf_test_pred, target_names=['high_risk', 'low_risk', 'moderate_risk'])
#训练集
rf_train_pred = rf_model.predict(X_train)
rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_train_report = classification_report(y_train, rf_train_pred, target_names=['high_risk', 'low_risk', 'moderate_risk'])

# 设置简化的网格搜索参数
param_grid = {
    'n_estimators': [100, 200],  # 减少树的数量选项
    'max_depth': [10, 20],  # 减少最大深度选项
    'min_samples_split': [10, 20],  # 减少分割内部节点所需的最小样本数选项
    'min_samples_leaf': [1, 2]  # 减少叶节点所需的最小样本数选项
}
# 创建随机森林模型
rf = RandomForestClassifier(random_state=42)
'''
# 实例化网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# 训练网格搜索模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# 使用最佳参数创建新的随机森林模型
best_rf = grid_search.best_estimator_

# 对测试集进行预测
best_rf_pred = best_rf.predict(X_test)

# 评估模型
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
best_rf_report = classification_report(y_test, best_rf_pred, target_names=['high_risk', 'low_risk', 'moderate_risk'])
'''
# XGBoost模型
'''
xgb_model = XGBClassifier(
    n_estimators=10,              # 更少的树的数量
    max_depth=2,                  # 更小的树深度
    subsample=0.6,                # 更低的行采样比例
    colsample_bytree=0.6,         # 更低的特征采样比例
    learning_rate=0.1,            # 保持适当的学习速率
    use_label_encoder=False,      # 禁用旧的标签编码方式
    eval_metric='mlogloss',       # 使用多类对数损失作为评估指标
    random_state=42               # 设定随机种子以确保结果的一致性
)
'''
xgb_model = XGBClassifier(n_estimators=30, max_depth=2, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_test_pred = xgb_model.predict(X_test)
xgb_test_accuracy = accuracy_score(y_test, xgb_test_pred)
#xgb_report = classification_report(y_test, xgb_pred)
xgb_test_report = classification_report(y_test, xgb_test_pred, target_names=['high_risk', 'low_risk', 'moderate_risk'])
#训练集
xgb_train_pred = xgb_model.predict(X_train)
xgb_train_accuracy = accuracy_score(y_train, xgb_train_pred)
xgb_train_report = classification_report(y_train, xgb_train_pred, target_names=['high_risk', 'low_risk', 'moderate_risk'])
# 打印结果（可选，根据实际需要输出）
print("Decision Tree Accuracy of Train Data:", dt_train_accuracy)
print("Decision Tree Report of Train Data:\n", dt_train_report)
print("Decision Tree Accuracy of Test Data:",dt_test_accuracy)
print("Decision Tree Report of Test Data:\n", dt_test_report)
print("Random Forest Accuracy of Train Data:", rf_train_accuracy)
print("Random Forest Report of Train Data:\n", rf_train_report)
print("Random Forest Accuracy of Test Data:", rf_test_accuracy)
print("Random Forest Report of Test Data:\n", rf_test_report)
#print("Random Forest Accuracy:", best_rf_accuracy)
#print("Random Forest Report:", best_rf_report)
print("XGBoost Accuracy of Train Data:", xgb_train_accuracy)
print("XGBoost Report of Train Data:\n", xgb_train_report)
print("XGBoost Accuracy of Test Data:", rf_test_accuracy)
print("XGBoost Report of Test Data:\n", rf_test_report)

