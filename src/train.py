import tensorflow as tf
import pandas as pd
from sklearn.model_selection import GroupKFold
from config import *
from model import build_model
from pathlib import Path

def create_dataset(df, augment=True):
    def preprocess_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img
    
    def augmentation(img):
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
        return img
    
    paths = df["path"].values
    labels = tf.one_hot(df["diagnosis"], NUM_CLASSES)
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda path, label: (preprocess_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if augment:
        dataset = dataset.map(
            lambda img, label: (augmentation(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    # Load and prepare data
    labels = pd.read_csv(PROCESSED_DATA_DIR / "trainLabels.csv")
    labels["path"] = labels.apply(
        lambda x: str(TRAIN_DIR / f"{x.id}_{x.laterality}.jpeg"), 
        axis=1
    )

    # Grouped K-Fold split
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(labels, groups=labels["id"]))

    # Create datasets
    train_ds = create_dataset(labels.iloc[train_idx], augment=True)
    val_ds = create_dataset(labels.iloc[val_idx], augment=False)

    # Build and compile model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
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

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
