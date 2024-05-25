from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np


def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    joblib.dump(log_reg, 'model/Logistic_Regression.pkl')
    return log_reg


def train_knn(X_train, y_train):
    # 添加噪声
    X_train_noisy = add_noise(X_train)

    # 标准化数据并使用管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # 使用网格搜索找到最佳的k值和距离度量
    param_grid = {
        'knn__n_neighbors': range(3, 11),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train_noisy, y_train)

    best_knn = grid_search.best_estimator_
    joblib.dump(best_knn, 'model/KNN.pkl')
    return best_knn


def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise
