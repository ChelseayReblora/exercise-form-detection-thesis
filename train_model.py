import numpy as np
import joblib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# PATH CONFIGURATION

BASE_DIR = Path(r"C:\Users\Neil\Downloads\backex-feature-roi-point-isolation\backex-feature-roi-point-isolation - Copy\back_exercise_form_detector")
FEATURE_DIR = BASE_DIR / "features"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# LOAD FEATURES

X = np.load(FEATURE_DIR / "X.npy")
exercise_labels = np.load(FEATURE_DIR / "exercise_labels.npy", allow_pickle=True)
form_labels = np.load(FEATURE_DIR / "form_labels.npy", allow_pickle=True)

print("X shape:", X.shape)
print("Exercise labels:", np.unique(exercise_labels))
print("Form labels:", np.unique(form_labels))


# ENCODE LABELS

exercise_encoder = LabelEncoder()
form_encoder = LabelEncoder()

y_exercise = exercise_encoder.fit_transform(exercise_labels)
y_form = form_encoder.fit_transform(form_labels)

print("Exercise classes:", exercise_encoder.classes_)
print("Form classes:", form_encoder.classes_)


# TRAIN / VALIDATION / TEST SPLIT

X_train, X_temp, y_ex_train, y_ex_temp, y_form_train, y_form_temp = train_test_split(
    X,
    y_exercise,
    y_form,
    test_size=0.30,
    random_state=42,
    stratify=y_exercise
)

X_val, X_test, y_ex_val, y_ex_test, y_form_val, y_form_test = train_test_split(
    X_temp,
    y_ex_temp,
    y_form_temp,
    test_size=0.50,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)


# NORMALIZE FEATURES

num_features = X_train.shape[-1]

scaler = StandardScaler()

X_train_2d = X_train.reshape(-1, num_features)
scaler.fit(X_train_2d)

def scale_sequences(data):
    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])
    scaled_2d = scaler.transform(data_2d)
    return scaled_2d.reshape(original_shape).astype(np.float32)

X_train_scaled = scale_sequences(X_train)
X_val_scaled = scale_sequences(X_val)
X_test_scaled = scale_sequences(X_test)


# SAVE SCALER AND ENCODERS

joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(exercise_encoder, MODEL_DIR / "exercise_encoder.pkl")
joblib.dump(form_encoder, MODEL_DIR / "form_encoder.pkl")


# BUILD GRU MODEL

SEQ_LEN = X_train.shape[1]
FEATURE_DIM = X_train.shape[2]

num_exercise_classes = len(exercise_encoder.classes_)
num_form_classes = len(form_encoder.classes_)

input_layer = Input(shape=(SEQ_LEN, FEATURE_DIM), name="motion_sequence")

x = GRU(64, return_sequences=False, name="gru_layer")(input_layer)
x = Dropout(0.30, name="dropout_layer")(x)
x = Dense(32, activation="relu", name="dense_layer")(x)

exercise_output = Dense(
    num_exercise_classes,
    activation="softmax",
    name="exercise_output"
)(x)

form_output = Dense(
    num_form_classes,
    activation="softmax",
    name="form_output"
)(x)

model = Model(
    inputs=input_layer,
    outputs=[exercise_output, form_output]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={
        "exercise_output": "sparse_categorical_crossentropy",
        "form_output": "sparse_categorical_crossentropy"
    },
    metrics={
        "exercise_output": ["accuracy"],
        "form_output": ["accuracy"]
    }
)

model.summary()


# TRAIN MODEL

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-06
    )
]

history = model.fit(
    X_train_scaled,
    {
        "exercise_output": y_ex_train,
        "form_output": y_form_train
    },
    validation_data=(
        X_val_scaled,
        {
            "exercise_output": y_ex_val,
            "form_output": y_form_val
        }
    ),
    epochs=80,
    batch_size=16,
    callbacks=callbacks
)


# EVALUATE MODEL

predictions = model.predict(X_test_scaled)

exercise_probs = predictions[0]
form_probs = predictions[1]

exercise_pred = np.argmax(exercise_probs, axis=1)
form_pred = np.argmax(form_probs, axis=1)

exercise_accuracy = accuracy_score(y_ex_test, exercise_pred)
form_accuracy = accuracy_score(y_form_test, form_pred)

combined_correct = np.logical_and(
    exercise_pred == y_ex_test,
    form_pred == y_form_test
)

combined_accuracy = np.mean(combined_correct)

incorrect_predictions = np.sum(
    np.logical_or(
        exercise_pred != y_ex_test,
        form_pred != y_form_test
    )
)

total_samples = len(y_ex_test)
percentage_error = (incorrect_predictions / total_samples) * 100

print("\nExercise Accuracy:", exercise_accuracy)
print("Form Accuracy:", form_accuracy)
print("Combined Accuracy:", combined_accuracy)
print(f"Percentage Error: {percentage_error:.2f}%")

print("\nExercise Confusion Matrix:")
print(confusion_matrix(y_ex_test, exercise_pred))

print("\nForm Confusion Matrix:")
print(confusion_matrix(y_form_test, form_pred))

print("\nExercise Classification Report:")
print(classification_report(
    y_ex_test,
    exercise_pred,
    target_names=exercise_encoder.classes_
))

print("\nForm Classification Report:")
print(classification_report(
    y_form_test,
    form_pred,
    target_names=form_encoder.classes_
))

# SAVE MODEL

model.save(MODEL_DIR / "gru_model.keras")

print("\nSaved model files to:", MODEL_DIR)
