import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from config import *
from pathlib import Path
import tensorflow as tf

def evaluate_model():
    # Load model
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    # Load validation data
    labels = pd.read_csv(PROCESSED_DATA_DIR / "trainLabels.csv")
    _, val_idx = next(GroupKFold(n_splits=5).split(labels, groups=labels["id"]))
    val_df = labels.iloc[val_idx]
    
    # Create dataset
    val_ds = tf.data.Dataset.from_tensor_slices((
        [str(TRAIN_DIR / f"{row.id}_{row.laterality}.jpeg") for _, row in val_df.iterrows()],
        val_df["diagnosis"].values
    ))
    
    # Preprocess function
    def preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label
    
    val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Get predictions
    y_true = np.concatenate([y for _, y in val_ds])
    y_pred = model.predict(val_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create evaluation directory
    eval_dir = BASE_DIR / "outputs" / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # 1. Classification Report
    report = classification_report(
        y_true,
        y_pred_classes,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    )
    with open(eval_dir / "metrics.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred_classes)))
    
    # 2. Confusion Matrix
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
                yticklabels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(eval_dir / "confusion_matrix.png")
    plt.close()
    
    # 3. ROC Curves
    plt.figure(figsize=(10,8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # One-vs-Rest ROC curves
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.savefig(eval_dir / "roc_curves.png")
    plt.close()
    
    # 4. Additional Metrics
    macro_f1 = f1_score(y_true, y_pred_classes, average='macro')
    weighted_f1 = f1_score(y_true, y_pred_classes, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred_classes)
    
    with open(eval_dir / "metrics.txt", "a") as f:
        f.write(f"\n\nAdditional Metrics:\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"ROC AUC (Macro): {roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro'):.4f}")

    print("Evaluation complete! Results saved in outputs/evaluation/")

if __name__ == "__main__":
    evaluate_model()
