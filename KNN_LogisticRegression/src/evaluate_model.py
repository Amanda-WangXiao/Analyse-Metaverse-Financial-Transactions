from sklearn.metrics import classification_report, roc_auc_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report['accuracy']
    return report, accuracy

def print_results(report, accuracy):
    print(f"{'Metric':<15}{'Precision':<15}{'Recall':<15}{'F1 Score':<15}{'Support':<15}")
    print(f"{'low_risk':<15}{report['0']['precision']:<15.2f}{report['0']['recall']:<15.2f}{report['0']['f1-score']:<15.2f}{report['0']['support']:<15.0f}")
    print(f"{'moderate_risk':<15}{report['1']['precision']:<15.2f}{report['1']['recall']:<15.2f}{report['1']['f1-score']:<15.2f}{report['1']['support']:<15.0f}")
    print(f"{'high_risk':<15}{report['2']['precision']:<15.2f}{report['2']['recall']:<15.2f}{report['2']['f1-score']:<15.2f}{report['2']['support']:<15.0f}")
    print(f"\n{'Accuracy:':<45}{accuracy:<15.2f}{int(sum([report[str(i)]['support'] for i in range(3)]))}")
    print(f"{'macro avg':<15}{report['macro avg']['precision']:<15.2f}{report['macro avg']['recall']:<15.2f}{report['macro avg']['f1-score']:<15.2f}{report['macro avg']['support']:<15.0f}")
    print(f"{'weighted avg':<15}{report['weighted avg']['precision']:<15.2f}{report['weighted avg']['recall']:<15.2f}{report['weighted avg']['f1-score']:<15.2f}{report['weighted avg']['support']:<15.0f}")
    print(f"Accuracy: {accuracy:.10f}")