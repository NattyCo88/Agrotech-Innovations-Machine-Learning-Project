import pandas as pd  # Data handling
import numpy as np  # Numerical operations
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet  # Models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier  # Models
from xgboost import XGBRegressor, XGBClassifier  # XGBoost models
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier  # KNN models
from sklearn.model_selection import train_test_split, GridSearchCV  # Data splitting and parameter tuning
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score  # Evaluation metrics
from sklearn.inspection import permutation_importance  # Feature importance for non-linear models
from config import CONFIG  # Configuration file for model setup and paths
from visualizations import plot_residuals, plot_predicted_vs_actual, plot_confusion_matrix, plot_roc_curve, plot_feature_importance  # Custom visualization utilities
import time  # Timing model runs


def train_and_evaluate(task_type, data_path):
    # Load configuration
    config = CONFIG["tasks"][task_type]
    target = config["target"]

    # Load data
    data = pd.read_csv(data_path)
    print(f"Columns in dataset: {data.columns}")
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    
    X = data.drop(columns=[target])
    y = data[target]

    # Handle categorical data
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Store results for summary
    model_results = []

    # Iterate through models
    for model_name, model_config in config["models"].items():
        print(f"\nTraining {model_name} for {task_type} task...")

        # Initialize model
        if model_name == "linear_regression":
            model = LinearRegression()
        elif model_name == "logistic_regression":
            model = LogisticRegression()
        elif model_name == "gradient_boosting":
            model = (
                GradientBoostingRegressor(random_state=42)
                if task_type == "regression"
                else GradientBoostingClassifier(random_state=42)
            )
        elif model_name == "xgboost":
            model = (
                XGBRegressor(random_state=42)
                if task_type == "regression"
                else XGBClassifier(random_state=42)
            )
        elif model_name == "knn":
            model = (
                KNeighborsRegressor()
                if task_type == "regression"
                else KNeighborsClassifier()
            )
        elif model_name == "random_forest":
            model = (
                RandomForestRegressor(random_state=42)
                if task_type == "regression"
                else RandomForestClassifier(random_state=42)
            )
        elif model_name == "elastic_net":
            if task_type == "regression":
                model = ElasticNet(random_state=42)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Perform parameter tuning if param_grid is available
        param_grid = model_config.get("param_grid", None)
        best_params = None
        try:
            start_time = time.time()  # Initialize timing before training starts
            if param_grid:
                print(f"Tuning {model_name} with parameters: {param_grid}")
                grid_search = GridSearchCV(
                    model,
                    param_grid=param_grid,
                    scoring=CONFIG["evaluation_metrics"].get(task_type, "accuracy"),
                    cv=3,
                    verbose=2,
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
            elapsed_time = time.time() - start_time  # Calculate the elapsed time
            print(f"{model_name} training completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            print(f"Error during parameter tuning or training for {model_name}: {e}")
            continue


        # Evaluate the model
        try:
            y_pred = model.predict(X_test)
            if task_type == "regression":
                # Regression evaluation
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
                elapsed_time = time.time() - start_time  # Calculate time taken for this model
                print(f"{model_name} RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, Time Taken: {elapsed_time:.2f} seconds")
                model_results.append({
                    "model": model_name, 
                    "metric": rmse,  # Ranking based on RMSE
                    "mape": mape,
                    "time_taken": elapsed_time,
                    "best_params": best_params,
                    "model_instance": model
                })
                plot_residuals(y_test, y_pred, model_name)
                plot_predicted_vs_actual(y_test, y_pred, model_name)
            elif task_type == "classification":
                # Classification evaluation
                f1 = f1_score(y_test, y_pred, average="weighted")
                accuracy = accuracy_score(y_test, y_pred)
                elapsed_time = time.time() - start_time  # Calculate time taken for this model
                print(f"{model_name} F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}, Time Taken: {elapsed_time:.2f} seconds")
                model_results.append({
                    "model": model_name, 
                    "metric": f1,  # Ranking based on F1-score
                    "accuracy": accuracy,
                    "time_taken": elapsed_time,
                    "best_params": best_params,
                    "model_instance": model
                })
                
                # Dynamically determine class names from y_test
                unique_classes = np.unique(y_test)
                class_names = [str(c) for c in unique_classes]

                # Plot confusion matrix
                try:
                    plot_confusion_matrix(y_test, y_pred, model_name, class_names)
                except Exception as e:
                    print(f"Error plotting confusion matrix for {model_name}: {e}")
                
                # Plot ROC curve
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)
                        plot_roc_curve(y_test, y_proba, model_name, len(class_names))
                    except Exception as e:
                        print(f"Error plotting ROC curve for {model_name}: {e}")
        except Exception as e:
            print(f"Error during evaluation for {model_name}: {e}")
            continue

        # Summary of results
        print(f"\n--- Summary for {task_type.capitalize()} Task ---")
        # Sort models by metric
        sorted_results = sorted(
            model_results,
            key=lambda x: x["metric"],
            reverse=(task_type == "classification")  # Higher F1 is better
        )

        for rank, result in enumerate(sorted_results, start=1):
            print(
                f"Rank {rank}: {result['model']} | Metric: {result['metric']:.4f} | "
                f"Best Params: {result['best_params']} | Time Taken: {result['time_taken']:.2f} seconds"
            )


    # Feature importance for the best model
    best_model = sorted_results[0]  # Best model
    model_instance = best_model["model_instance"]
    print(f"\n--- Feature Importance for Best {task_type.capitalize()} Model ({best_model['model']}) ---")

    # Determine feature importance
    if hasattr(model_instance, "feature_importances_"):
        feature_importance = model_instance.feature_importances_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
        print(importance_df.sort_values(by="Importance", ascending=False))
        plot_feature_importance(importance_df, f"{task_type}_{best_model['model']}")
    elif hasattr(model_instance, "coef_"):
        feature_importance = model_instance.coef_
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
        print(importance_df.sort_values(by="Importance", ascending=False))
        plot_feature_importance(importance_df, f"{task_type}_{best_model['model']}")
    else:
        result = permutation_importance(
            model_instance, X_test, y_test, scoring=CONFIG["evaluation_metrics"].get(task_type), random_state=42
        )
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": result.importances_mean})
        print(importance_df.sort_values(by="Importance", ascending=False))
        plot_feature_importance(importance_df, f"{task_type}_{best_model['model']}")


if __name__ == "__main__":
    try:
        train_and_evaluate("regression", CONFIG["cleaned_data_path"])
        train_and_evaluate("classification", CONFIG["cleaned_data_path"])
    except Exception as e:
        print(f"Pipeline error: {e}")
