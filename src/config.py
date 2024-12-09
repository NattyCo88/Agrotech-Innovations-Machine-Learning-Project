CONFIG = {
    "db_path": "Data/agri.db",
    "cleaned_data_path": "Data/agri_data_cleaned.csv",
    "tasks": {
        "regression": {
            "target": "Temperature Sensor (°C)",  # Target column
            "models": {
                "linear_regression": {},  # No hyperparameters for linear regression
                "random_forest": {
                    "param_grid": {
                        "n_estimators": [200],
                        "max_depth": [None],
                        "min_samples_split": [2],
                        "min_samples_leaf": [1]
                    }
                },
                "gradient_boosting": {
                    "param_grid": {
                        "n_estimators": [150],
                        "learning_rate": [0.1],
                        "max_depth": [15],
                        "min_samples_split": [2],
                        "min_samples_leaf": [1],
                        "subsample": [0.8]
                    }
                },
                "xgboost": {
                    "param_grid": {
                        "n_estimators": [150],
                        "learning_rate": [0.1],
                        "max_depth": [15],
                        "subsample": [0.8],
                        "colsample_bytree": [1.0]
                    }
                }
            }
        },
        "classification": {
            "target": "Plant Stage Encoded",  # Target column
            "models": {
                "logistic_regression": {
                    "param_grid": {
                        "C": [100],
                        "max_iter": [200]
                    }
                },
                "random_forest": {
                    "param_grid": {
                        "n_estimators": [150],
                        "max_depth": [20],
                        "min_samples_split": [2],
                        "min_samples_leaf": [1]
                    }
                },
                "xgboost": {
                    "param_grid": {
                        "n_estimators": [150],
                        "learning_rate": [0.1],
                        "max_depth": [15],
                        "subsample": [0.8],
                        "colsample_bytree": [1.0]
                    }
                },
                "knn": {
                    "param_grid": {
                        "n_neighbors": [10],
                        "weights": ["distance"],
                        "metric": ["manhattan"]
                    }
                }
            }
        }
    },
    "evaluation_metrics": {
        "regression": "neg_mean_squared_error",  # Scoring for regression
        "classification": "f1_weighted"          # Scoring for classification
    }
}
