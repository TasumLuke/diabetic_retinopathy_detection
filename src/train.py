import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GroupKFold
from config import *
from model import build_model
from pathlib import Path
from tensorflow.keras.optimizers.legacy import Adam  # type: ignore
import os


def create_dataset(df, augment=True):
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
                    f"Warning: File not found - Skipping in generator: {path}")
            except tf.errors.InvalidArgumentError:
                tf.print(
                    f"Warning: Invalid JPEG data - Skipping in generator: {path}")

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
        )
    )

    def augmentation(img, label):
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, label

    dataset = dataset.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    labels = pd.read_csv(PROCESSED_DATA_DIR / "trainLabels.csv")
    print(labels.columns)
    print(labels.head())
    labels["path"] = labels.apply(
        lambda x: str(
            TRAIN_DIR / f"{x['image'].split('_')[0]}_{x['image'].split('_')[1]}.jpeg"),
        axis=1
    )
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(labels, groups=labels["image"]))
    print(f"Train indices: {train_idx}")
    print(f"Validation indices: {val_idx}")
    train_ds = create_dataset(labels.iloc[train_idx], augment=True)
    val_ds = create_dataset(labels.iloc[val_idx], augment=False)

    model = build_model()
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_SAVE_PATH),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(BASE_DIR / "outputs" / "logs"))
    ]

    EPOCHS_FOR_TESTING = 2  
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FOR_TESTING,
        callbacks=callbacks
    )

    print("Training completed (for testing).")
