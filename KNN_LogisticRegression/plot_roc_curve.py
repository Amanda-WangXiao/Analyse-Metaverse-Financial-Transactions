import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib
import numpy as np

def plot_macro_averaged_roc_curve(model, X_test, y_test, output_path):
    # 将标签二值化
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    # 获取预测概率
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        raise ValueError("The model does not support predict_proba method.")

    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)

    # 确保曲线从(0,0)开始到(1,1)结束
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    # 绘制ROC曲线
    plt.figure()
    plt.plot(all_fpr, mean_tpr, color='orange', label=f'ROC curve (AUC = {macro_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # 保存图像
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    # 加载模型

    model = joblib.load('model/Logistic_Regression.pkl')
    #model = joblib.load('model/KNN.pkl')
    # 加载测试数据
    from src.data_processing import load_and_process_data

    filepath = 'dataset/metaverse_transactions_dataset.csv'
    _, X_test, _, y_test = load_and_process_data(filepath, 'anomaly')

    # 定义输出路径
    output_path = 'roc_curve_Logistic_Regression.png'
    #output_path = 'roc_curve_KNN.png'

    # 绘制宏平均ROC曲线并保存图像
    plot_macro_averaged_roc_curve(model, X_test, y_test, output_path)