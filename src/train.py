import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import mlflow.lightgbm
from sklearn.metrics import root_mean_squared_error, r2_score

def train_model(X_train, y_train, X_test, y_test):
    # Ustawienie bazy danych w konkretnym miejscu
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Tire Forecast")

    with mlflow.start_run():

        params = {
            "learning_rate": 0.11,
            "n_estimators": 96,
            "num_leaves": 12,
            "random_state": 42
            }
        
        mlflow.log_params(params)

        model = lgb.LGBMRegressor(**params, verbosity = -1)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.lightgbm.log_model(model, "lightgbm")

        print(f"Model zapisany, RMSE: {rmse}")
        return model