import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from config import *

def predict():
    # Load model
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    # Get test images
    test_files = list(Path(TEST_DIR).glob("*.jpeg"))
    
    predictions = []
    for img_path in test_files:
        # Preprocess
        img = tf.io.read_file(str(img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        
        # Predict
        pred = model.predict(tf.expand_dims(img, axis=0))
        predictions.append({
            "id": img_path.stem.split("_")[0],
            "diagnosis": np.argmax(pred)
        })
    
    # Save predictions
    pd.DataFrame(predictions).to_csv(
        BASE_DIR / "outputs" / "submissions" / "submission.csv",
        index=False
    )

if __name__ == "__main__":
    predict()
