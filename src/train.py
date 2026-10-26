"""
Step 2: SoH model.

- Linear regression baseline + small MLP (4 -> 16 -> 8 -> 1)
- Leave-one-battery-out (LOBO): train on 3 batteries, test on the 4th
- Full-integer INT8 quantization with TensorFlow Lite (LiteRT)

Run:  python src/train.py      (needs results/features.csv from src/data.py)
Out:  results/soh_results.csv, results/soh_pred_vs_true.png,
      results/models/ (INT8 model + scaler for the firmware fold)
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import RESULTS, BATTERIES, FEATURES

MODELS = os.path.join(RESULTS, "models")
FIRMWARE_TEST_BATTERY = "B0005"  # the held-out battery we later run on the ESP32
SEED = 42
EPOCHS = 400
LEARNING_RATE = 0.001
L2 = 1e-3  # weight penalty; without it the MLP overfits and fails badly on B0018


def build_mlp():
    """4 -> 16 -> 8 -> 1. Output is SoH as a fraction (0..1)."""
    reg = tf.keras.regularizers.l2(L2)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model


def to_tflite(model, x_train_scaled, int8):
    """Convert a Keras model to a .tflite flatbuffer (bytes).

    int8=True -> full-integer quantization: weights, activations, input and
    output are all int8. The representative dataset lets the converter see
    real input ranges so it can pick good scales.
    """
    export_dir = os.path.join(MODELS, "_saved_model")
    model.export(export_dir, verbose=False)
    conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    if int8:
        def representative_data():
            for row in x_train_scaled:
                yield [row.reshape(1, 4).astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_data
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        # One scale per weight tensor (not per output channel): the TFLite Micro
        # version in the ESP32 library only supports per-tensor Dense weights.
        conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv.convert()


def run_tflite(tflite_bytes, x_scaled):
    """Run a .tflite model on each row; handles int8 input/output scaling."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for row in x_scaled:
        x = row.reshape(1, 4).astype(np.float32)
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"]).astype(np.float32)
        if out["dtype"] == np.int8:
            s, z = out["quantization"]
            y = (y - z) * s
        preds.append(y[0, 0])
    return np.array(preds)


def mae_rmse(y_true, y_pred):
    err = y_pred - y_true
    return np.mean(np.abs(err)), np.sqrt(np.mean(err ** 2))


def main():
    os.makedirs(MODELS, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS, "features.csv"))
    rows = []

    for test_bat in BATTERIES:
        tf.keras.utils.set_random_seed(SEED)
        train = df[df["battery"] != test_bat]
        test = df[df["battery"] == test_bat]

        # Scaler is fit on the 3 training batteries only (no test leakage)
        scaler = StandardScaler().fit(train[FEATURES].values)
        x_tr = scaler.transform(train[FEATURES].values)
        x_te = scaler.transform(test[FEATURES].values)
        y_tr = train["soh"].values / 100.0   # train on fraction 0..1
        y_te = test["soh"].values            # report in SoH %

        # Baseline: linear regression
        lr = LinearRegression().fit(x_tr, y_tr)
        pred_lr = lr.predict(x_te) * 100

        # MLP (FP32)
        mlp = build_mlp()
        mlp.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=32, verbose=0)
        pred_fp32 = mlp.predict(x_te, verbose=0).ravel() * 100

        # TFLite FP32 and INT8
        tfl_fp32 = to_tflite(mlp, x_tr, int8=False)
        tfl_int8 = to_tflite(mlp, x_tr, int8=True)
        pred_int8 = run_tflite(tfl_int8, x_te) * 100

        for name, pred in [("linear", pred_lr), ("mlp_fp32", pred_fp32), ("mlp_int8", pred_int8)]:
            mae, rmse = mae_rmse(y_te, pred)
            rows.append({"test_battery": test_bat, "model": name,
                         "mae_soh_pct": mae, "rmse_soh_pct": rmse})
        print(f"held-out {test_bat}: " + ", ".join(
            f"{r['model']} MAE {r['mae_soh_pct']:.2f}%" for r in rows[-3:]))

        # Keep the fold whose test battery runs on the ESP32
        if test_bat == FIRMWARE_TEST_BATTERY:
            with open(os.path.join(MODELS, "soh_fp32.tflite"), "wb") as f:
                f.write(tfl_fp32)
            with open(os.path.join(MODELS, "soh_int8.tflite"), "wb") as f:
                f.write(tfl_int8)
            with open(os.path.join(MODELS, "scaler.json"), "w") as f:
                json.dump({"features": FEATURES, "mean": scaler.mean_.tolist(),
                           "scale": scaler.scale_.tolist(),
                           "train_batteries": [b for b in BATTERIES if b != test_bat]}, f, indent=2)
            sizes = {"fp32_tflite_bytes": len(tfl_fp32), "int8_tflite_bytes": len(tfl_int8)}
            with open(os.path.join(MODELS, "sizes.json"), "w") as f:
                json.dump(sizes, f, indent=2)
            print(f"Model size: FP32 {len(tfl_fp32)} B, INT8 {len(tfl_int8)} B")

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(test["cycle"], y_te, "k-", label="true SoH")
            ax.plot(test["cycle"], pred_lr, ".", alpha=0.6, label="linear")
            ax.plot(test["cycle"], pred_fp32, ".", alpha=0.6, label="MLP FP32")
            ax.plot(test["cycle"], pred_int8, "x", alpha=0.6, label="MLP INT8")
            ax.set_xlabel("discharge cycle")
            ax.set_ylabel("SoH (%)")
            ax.set_title(f"Held-out battery {test_bat} (trained on the other 3)")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(RESULTS, "soh_pred_vs_true.png"), dpi=120)

    res = pd.DataFrame(rows)
    avg = res.groupby("model", as_index=False)[["mae_soh_pct", "rmse_soh_pct"]].mean()
    avg.insert(0, "test_battery", "average")
    res = pd.concat([res, avg], ignore_index=True)
    res.to_csv(os.path.join(RESULTS, "soh_results.csv"), index=False, float_format="%.3f")
    print(res.pivot(index="test_battery", columns="model", values="mae_soh_pct").round(2))


if __name__ == "__main__":
    main()
