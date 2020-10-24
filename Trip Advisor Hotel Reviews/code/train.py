from dataset_loader import load_dataset
from engine import get_engine
from metrics import get_classification_score
import argparse
import json
import matplotlib.pyplot as plt
import os
import pickle

TAB_1 = "    - "
def save_results(model, metrics_dict, save_dir):
    model_filepath = os.path.join(save_dir, "model.sav")
    pickle.dump(model, open(model_filepath, "wb"))
    print(f"{TAB_1}Model saved at {model_filepath}")
    

    classification_report_filepath = os.path.join(save_dir, "classification_report.txt")
    with open(classification_report_filepath, "w") as f:
        f.write(metrics_dict.pop("classification_report"))
    print(f"{TAB_1}Classification Report saved at {classification_report_filepath}")
    
    
    cm_filepath = os.path.join(save_dir, "confusion_matrix.png")
    cm_plot = metrics_dict.pop("confusion_matrix_plot")
    fig, ax = plt.subplots(1, 1)
    cm_plot.plot(ax=ax)
    plt.savefig(cm_filepath)
    print(f"{TAB_1}Confusion Matrix plot saved at {cm_filepath}")


    scores_filepath = os.path.join(save_dir, "scores.json")
    with open(scores_filepath, "w") as f:
        json.dump(metrics_dict, f)
    print(f"{TAB_1}Scores saved at {scores_filepath}")


def main(clf_name):
    clf = get_engine(clf_name)
    if clf is not None:
        models_save_dir = os.path.join("..", "models", clf_name)

        print(f"Training {clf_name.upper()} classifier...")

        for i, (x_train, x_test, y_train, y_test) in enumerate(load_dataset()):
            clf.fit(x_train, y_train)
            y_pred_probabs = clf.predict_proba(x_test)
            # pickle.dump(y_pred_probabs, open("temp.pkl", "wb"))
            metrics_dict = get_classification_score(y_test, y_pred_probabs)
            
            save_dir = os.path.join(models_save_dir, str(i+1))
            os.makedirs(save_dir, exist_ok=True)
            save_results(clf, metrics_dict, save_dir)

            print(f"{TAB_1}Set {i+1} done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clf_name", default="rfc")

    args = parser.parse_args()
    main(args.clf_name)