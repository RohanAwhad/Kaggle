from sklearn.metrics import accuracy_score, classification_report, \
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_fscore_support

import numpy as np

def get_classification_score(y_true, y_pred_probabs, target_names=None):
    y_pred = np.argmax(y_pred_probabs, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    
    cr = classification_report(y_true, y_pred, target_names=target_names)

    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    cm_plot = ConfusionMatrixDisplay(cm)
    
    ras = roc_auc_score(y_true, y_pred_probabs, multi_class="ovr")

    precision, recall, f1, _ = \
        precision_recall_fscore_support(y_true, y_pred, average="macro", 
                                        zero_division=0)

    ret = {"accuracy" : acc, "classification_report" : cr,
            "confusion_matrix_plot" : cm_plot, 
            "f1_score" : f1, "precision" : precision,
            "recall" : recall,
            "roc_auc_score" : ras}

    return ret