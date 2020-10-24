from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

rfc = OneVsRestClassifier(RandomForestClassifier(n_estimators=10, verbose=1))
svc = OneVsRestClassifier(SVC(probability=False, verbose=1))
xgb = OneVsRestClassifier(XGBClassifier(verbosity=1))

def get_engine(clf_name):
    if clf_name.lower() == "svc":
        return svc
    elif clf_name.lower() == "rfc":
        return rfc
    elif clf_name.lower() == "xgb":
        return xgb
    else:
        return None