from datetime import datetime
import shap
import matplotlib.pyplot as plt

import mlflow

def get_last_run_date(run_id):
    try:
        run = mlflow.get_run(run_id)
        start_time = run.info.start_time / 1000
        return datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "Nieznana"
    
def features_explainer(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    fig = plt.figure(figsize=(10,6), dpi=300)
    shap.summary_plot(shap_values, X_train, show=False)
    return fig