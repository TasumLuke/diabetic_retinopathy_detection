import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
output_path = Path(BASE_DIR) / "outputs" / "evaluation"
output_path.mkdir(parents=True, exist_ok=True)


def create_dataset(df, augment=False):
    def data_generator():
        paths = df["path"].values
        labels_values = df["level"].values
        for path, label in zip(paths, labels_values):
            try:
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                img = tf.keras.applications.efficientnet.preprocess_input(img)
                yield img, tf.one_hot(label, NUM_CLASSES)
            except tf.errors.NotFoundError:
                tf.print(
                    f"Warning (Eval - Generator): File not found - Skipping: {path}")
            except tf.errors.InvalidArgumentError:
                tf.print(
                    f"Warning (Eval - Generator): Invalid JPEG data - Skipping: {path}")

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
        )
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def evaluate_model():
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    labels = pd.read_csv(PROCESSED_DATA_DIR / "trainLabels.csv")
    labels["id"] = labels["image"].apply(lambda x: x.split("_")[0])
    labels["path"] = labels.apply(
        lambda x: str(
            TRAIN_DIR / f"{x['image'].split('_')[0]}_{x['image'].split('_')[1]}.jpeg"),
        axis=1
    )
    _, val_idx = next(GroupKFold(n_splits=5).split(
        labels, groups=labels["id"]))
    val_df = labels.iloc[val_idx].reset_index(drop=True)

    valid_predictions = []
    valid_true_labels_final = []

    def data_generator():
        paths = val_df["path"].values
        true_labels = val_df["level"].values
        for path, label in zip(paths, true_labels):
            try:
                img = tf.io.read_file(path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                img = tf.keras.applications.efficientnet.preprocess_input(img)
                prediction = model.predict(tf.expand_dims(img, axis=0))[0]
                valid_predictions.append(prediction)
                valid_true_labels_final.append(label)
                yield img  
            except (tf.errors.NotFoundError, tf.errors.InvalidArgumentError) as e:
                tf.print(
                    f"Warning (Eval - Generator): Skipping {path} due to error: {e}")

    val_ds = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tf.TensorSpec(
            shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    for _ in val_ds:
        pass

    predicted_classes = np.argmax(np.array(valid_predictions), axis=1)
    true_classes = np.array(valid_true_labels_final)

    print(
        f"y_true shape: {len(true_classes)}, y_pred shape: {len(predicted_classes)}")
    if len(true_classes) > 0 and len(predicted_classes) > 0 and len(true_classes) == len(predicted_classes):
        kappa = cohen_kappa_score(
            true_classes, predicted_classes, weights="quadratic")
        print(f"Quadratic Kappa Score: {kappa:.4f}")

        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes,
              target_names=["No DR", "Mild", "Moderate",
                            "Severe", "Proliferative"], zero_division=1))

        # 2. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_classes, predicted_classes)
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=["No DR", "Mild", "Moderate",
                                 "Severe", "Proliferative"],
                    yticklabels=["No DR", "Mild", "Moderate", "Severe", "Proliferative"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(output_path / "confusion_matrix.png") 
        plt.close()

        # 3. ROC Curves
        plt.figure(figsize=(10, 8))
        lb = LabelBinarizer()
        lb.fit(true_classes)
        y_true_bin = lb.transform(true_classes)
        y_pred_bin = np.array(valid_predictions)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        for i, color in zip(range(NUM_CLASSES), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig(output_path / "roc_curves.png")
        plt.close()

    else:
        print("Warning: No valid predictions could be made due to missing files.")


if __name__ == "__main__":
    evaluate_model()
