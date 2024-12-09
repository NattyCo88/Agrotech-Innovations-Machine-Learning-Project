import os  # For creating directories
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For enhanced visualizations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc  # Metrics for evaluation
import numpy as np  # For numerical operations

# Ensure "Plots" directory exists
os.makedirs("Plots", exist_ok=True)

def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Residual Plot for {model_name}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    # Save the plot
    save_file = f"Plots/{model_name}_residual_plot.png"
    plt.savefig(save_file)
    print(f"Residual Plot saved to {save_file}")
    plt.close()  # Close the plot to free memory

def plot_predicted_vs_actual(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, label="Predictions")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="red", linestyle="--", label="Perfect Fit")
    plt.title(f"Predicted vs Actual for {model_name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    # Save the plot
    save_file = f"Plots/{model_name}_pred_vs_actual.png"
    plt.savefig(save_file)
    print(f"Predicted Versus Actual saved to {save_file}")
    plt.close()  # Close the plot to free memory

def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    # Dynamically determine class names if not provided
    if class_names is None:
        class_names = [str(c) for c in np.unique(y_true)]
    cm = confusion_matrix(y_true, y_pred)
    
    if len(class_names) != cm.shape[0]:
        raise ValueError(f"Number of class names ({len(class_names)}) does not match the number of unique classes ({cm.shape[0]}).")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix for {model_name}")
    # Save the plot
    save_file = f"Plots/{model_name}_confusion_matrix.png"
    plt.savefig(save_file)
    print(f"Confusion matrix saved to {save_file}")
    plt.close()  # Close the plot to free memory

def plot_roc_curve(y_true, y_proba, model_name, n_classes):
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(y_true)
    for i in range(n_classes):
        if i >= len(unique_classes):
            print(f"Skipping class {i} as it is not in y_true.")
            continue
        fpr, tpr, _ = roc_curve(y_true == unique_classes[i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {unique_classes[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title(f"ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    # Save the plot
    save_file = f"Plots/{model_name}_roc_curve.png"
    plt.savefig(save_file)
    print(f"ROC Curve saved to {save_file}")
    plt.close()  # Close the plot to free memory


def plot_feature_importance(importance_df, model_name, top_n=10):
    """
    Plot a bar chart of feature importances.

    Parameters:
    - importance_df: DataFrame with 'Feature' and 'Importance' columns.
    - model_name: Name of the model for the title.
    - top_n: Number of top features to display (default is 10).
    """
    # Sort and select top features
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        palette="viridis"
    )
    plt.title(f"Top {top_n} Feature Importances for {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    # Save the plot
    save_file = f"Plots/{model_name}_feature_importance.png"
    plt.savefig(save_file)
    print(f"Feature importance plot saved to {save_file}")
    plt.close()
