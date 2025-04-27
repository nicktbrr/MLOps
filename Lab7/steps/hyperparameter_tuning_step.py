from metaflow import step
import mlflow
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def hyperparameter_tuning_step(self):
    """Perform hyperparameter tuning"""
    params = hp.choice('regressor_type', [
        {
            'type': 'dt',  # Decision Tree
            'max_depth': hp.choice('dt_max_depth', [3, 6, 10, None]),
            'min_samples_split': hp.choice('dt_min_samples_split', [2, 5]),
            'criterion': hp.choice('dt_criterion', ['squared_error', 'friedman_mse'])
        },
        {
            'type': 'rf',
            'n_estimators': hp.choice('rf_n_estimators', [50, 100]),
            'max_features': hp.choice('rf_max_features', ['sqrt', 'log2']),
            'max_depth': hp.choice('rf_max_depth', [None, 10])
        },
        {
            'type': 'ridge',
            'alpha': hp.choice('ridge_alpha', [0.1, 1.0]),
            'fit_intercept': hp.choice('ridge_fit_intercept', [True, False])
        }
    ])

    best_val_rmse = float('inf')

    def objective(params):
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_experiment('boston')
        with mlflow.start_run()as run:
            regressor_type = params['type']
            del params['type']

            if regressor_type == 'dt':  # Decision tree
                reg = DecisionTreeRegressor(**params)
            elif regressor_type == 'rf':
                reg = RandomForestRegressor(**params)
            elif regressor_type == 'ridge':
                reg = Ridge(**params)
            else:
                return 0

            # Fit the model using PCA-transformed training data
            reg.fit(self.X_train_pca, self.y_train)

            # Evaluate on validation set for hyperparameter tuning
            y_val_pred = reg.predict(self.X_val_pca)
            val_mse = np.mean((self.y_val - y_val_pred) ** 2)
            val_rmse = np.sqrt(val_mse)


            # Log the model parameters and performance metrics
            mlflow.set_tag("Model", regressor_type)
            mlflow.log_params(params)
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.sklearn.log_model(reg, artifact_path='regression_models')

            # Reference the data preparation run
            mlflow.set_tag("data_preparation_run", mlflow.active_run().info.run_id)

            if val_rmse < best_val_rmse:
                self.best_model = reg
                self.best_run_id = run.info.run_id

            # Return validation RMSE as the loss to minimize
            return {'loss': val_rmse, 'status': STATUS_OK}
    algo = tpe.suggest
    trials = Trials()
    
    best_result = fmin(
        fn=objective,
        space=params,
        algo=algo,
        max_evals=18,
        trials=trials
    )

    self.best_result = best_result
    # self.best_model = create_model(best_result)