from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    joblib.dump(log_reg, 'model/Logistic_Regression.pkl')
    return log_reg

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, 'model/KNN.pkl')
    return knn
