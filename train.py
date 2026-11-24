"""
Step 2: train the gesture classifier (a CNN) with TensorFlow / Keras.

Two models can be trained, so you can compare them in the presentation:

  cnn        Our OWN convolutional neural network, built layer by layer and
             trained from scratch on the hand-gesture dataset.
  mobilenet  TRANSFER LEARNING: MobileNetV2 (a CNN Google pre-trained on
             1.4 million ImageNet photos) with a new classifier head for our
             gestures, then fine-tuned.

How the photos are split (the important part for an honest score):
  Every photo's file name starts with the recording session, for example
  s20261107_1400_....jpg. Photos from one session look very alike (same room,
  light and clothes), so if the test photos come from the same session as the
  training photos the score is fake-high. So:
    * the TEST set is one whole session that the model never trains on
      (the newest one, or the one given with --test-session);
    * the VALIDATION set is the last 15% (in time) of every other session;
    * everything else is TRAINING.
  With only one session it falls back to: first 70% of each gesture's photos
  (in time order) train, next 15% validation, last 15% test.

Outputs go to models/<model>/ (Keras model, TFLite model, labels, plots,
report). With --deploy the chosen model is copied into models/, which is
where detect.py / classifier.py load it from.

Examples:
    python train.py --model cnn
    python train.py --model mobilenet
    python train.py --model both --deploy best     # train both, use the better
"""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow chatter

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# File names look like <session>_<milliseconds>_<counter>.jpg
NAME_PATTERN = re.compile(r"^(?P<session>.+)_(?P<ms>\d{10,})_(?P<n>\d+)$")

# Input size per model. MobileNetV2 accepts 96/128/160/192/224.
IMG_SIZES = {"cnn": config.CNN_IMG_SIZE, "mobilenet": config.IMG_SIZE}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def describe_file(path):
    """Return (session, time_ms) for a dataset photo."""
    match = NAME_PATTERN.match(path.stem)
    if match:
        return match["session"], int(match["ms"])
    return "unknown", int(path.stat().st_mtime * 1000)


def find_images(data_dir):
    """Return (class_names, {class: [(path, session, time_ms), ...]})."""
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    files = {}
    for name in class_names:
        items = []
        for p in (data_dir / name).iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                session, ms = describe_file(p)
                items.append((str(p), session, ms))
        if items:
            files[name] = sorted(items, key=lambda item: item[2])
    class_names = [name for name in class_names if name in files]
    return class_names, files


def split_files(class_names, files, test_session=None):
    """Split by recording session (see the top of this file)."""
    sessions = sorted({s for items in files.values() for _, s, _ in items})
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}

    def add(split, items, label):
        splits[split][0].extend(path for path, _, _ in items)
        splits[split][1].extend([label] * len(items))

    if len(sessions) >= 2:
        test_session = test_session or sessions[-1]
        if test_session not in sessions:
            raise SystemExit(f"No session called {test_session}. "
                             f"Sessions found: {sessions}")
        mode = f"session split: test = session {test_session}"
        for label, name in enumerate(class_names):
            for session in sessions:
                items = [it for it in files[name] if it[1] == session]
                if session == test_session:
                    add("test", items, label)
                    continue
                n_val = int(round(len(items) * 0.15))
                add("train", items[:len(items) - n_val], label)
                add("val", items[len(items) - n_val:], label)
    else:
        mode = "one session only: time-ordered 70/15/15 split"
        for label, name in enumerate(class_names):
            items = files[name]
            n_train = int(len(items) * 0.70)
            n_val = int(len(items) * 0.15)
            add("train", items[:n_train], label)
            add("val", items[n_train:n_train + n_val], label)
            add("test", items[n_train + n_val:], label)
    return splits, sessions, mode


def make_dataset(paths, labels, img_size, training):
    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size))
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=config.SEED)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE)
    if training:
        # Data augmentation: every epoch the model sees slightly changed
        # copies (moved, turned, zoomed, darker/brighter), so it learns the
        # hand shape instead of memorising the exact photos.
        augment = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),   # left and right hands
            keras.layers.RandomRotation(0.10),
            keras.layers.RandomZoom(0.20),
            keras.layers.RandomTranslation(0.15, 0.15),
            keras.layers.RandomContrast(0.3),
            keras.layers.RandomBrightness(0.3, value_range=(0, 255)),
        ], name="augmentation")
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def conv_block(x, filters):
    """Two 3x3 convolutions + batch norm + ReLU, then 2x2 max pooling.

    Convolution layers slide small 3x3 filters over the picture to find
    patterns (edges first, then finger shapes); pooling halves the size so the
    next block looks at bigger parts of the hand.
    """
    for _ in range(2):
        x = keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization(momentum=0.9)(x)
        x = keras.layers.ReLU()(x)
    return keras.layers.MaxPooling2D()(x)


def build_cnn(num_classes, img_size):
    """Our own CNN, trained from scratch."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.layers.Rescaling(1.0 / 255)(inputs)       # pixels 0-255 -> 0-1
    for filters in (32, 64, 128, 256):                   # 128 -> 64 -> 32 -> 16 -> 8
        x = conv_block(x, filters)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="gesture_cnn")


def build_mobilenet(num_classes, img_size, weights):
    """Transfer learning: frozen MobileNetV2 + a new classifier head."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    # MobileNetV2 expects pixels in [-1, 1]; doing it inside the model means
    # detect.py can feed plain 0-255 RGB pixels.
    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights)
    base.trainable = False
    x = base(x, training=False)  # keep BatchNorm layers in inference mode
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="gesture_mobilenetv2"), base


def compile_model(model, learning_rate):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def class_weights(labels, num_classes):
    """Give rare classes more weight so an unbalanced dataset still works."""
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    return {i: total / (num_classes * c) for i, c in enumerate(counts) if c}


def train_cnn(model, train_ds, val_ds, weights, epochs):
    compile_model(model, config.CNN_LEARNING_RATE)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                      start_from_epoch=10,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
    ]
    print(f"\n=== Training our own CNN from scratch ({epochs} epochs max) ===")
    return [model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                      class_weight=weights, callbacks=callbacks)]


def train_mobilenet(model, base, train_ds, val_ds, weights,
                    epochs_frozen, epochs_fine_tune):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True)
    print("\n=== MobileNetV2 phase 1: training the new classifier head ===")
    compile_model(model, config.LEARNING_RATE)
    histories = [model.fit(train_ds, validation_data=val_ds,
                           epochs=epochs_frozen, class_weight=weights,
                           callbacks=[early_stop])]
    if epochs_fine_tune > 0:
        print(f"\n=== MobileNetV2 phase 2: fine-tuning the last "
              f"{config.FINE_TUNE_LAYERS} layers ===")
        base.trainable = True
        for layer in base.layers[:-config.FINE_TUNE_LAYERS]:
            layer.trainable = False
        compile_model(model, config.FINE_TUNE_LEARNING_RATE)
        histories.append(model.fit(train_ds, validation_data=val_ds,
                                   epochs=epochs_fine_tune,
                                   class_weight=weights,
                                   callbacks=[early_stop]))
    return histories


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def plot_history(histories, title, path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, train, val, name in ((axes[0], acc, val_acc, "Accuracy"),
                                 (axes[1], loss, val_loss, "Loss")):
        ax.plot(epochs, train, label="training")
        ax.plot(epochs, val, label="validation")
        if len(histories) > 1:
            ax.axvline(len(histories[0].history["accuracy"]) + 0.5,
                       color="grey", linestyle="--", label="fine-tuning starts")
        ax.set_title(f"{title}: {name}")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_matrix(true, pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        matrix[t, p] += 1
    return matrix


def plot_confusion_matrix(matrix, class_names, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    ax.set_title(f"{title}: confusion matrix (test set)")
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def classification_report(matrix, class_names):
    lines = [f"{'gesture':10s} {'precision':>9s} {'recall':>7s} {'f1':>6s} "
             f"{'images':>7s}"]
    for i, name in enumerate(class_names):
        tp = matrix[i, i]
        precision = tp / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        recall = tp / matrix[i, :].sum() if matrix[i, :].sum() else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        lines.append(f"{name:10s} {precision:9.3f} {recall:7.3f} {f1:6.3f} "
                     f"{matrix[i, :].sum():7d}")
    return "\n".join(lines)


def export_tflite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Dynamic-range quantisation: ~4x smaller and faster, tiny accuracy loss.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    path.write_bytes(converter.convert())


def evaluate_tflite(path, test_ds):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    correct = total = 0
    times = []
    for images, labels in test_ds.unbatch().batch(1):
        interpreter.set_tensor(input_index, images.numpy().astype(np.float32))
        start = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - start)
        pred = np.argmax(interpreter.get_tensor(output_index)[0])
        correct += int(pred == labels.numpy()[0])
        total += 1
    return correct / max(total, 1), 1000 * float(np.median(times))


def plot_comparison(results, path):
    names = [r["title"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    charts = (("Test accuracy (%)", [100 * r["tflite_accuracy"] for r in results]),
              ("TFLite size (MB)", [r["tflite_mb"] for r in results]),
              ("Time per image on laptop (ms)", [r["tflite_ms"] for r in results]))
    for ax, (title, values) in zip(axes, charts):
        bars = ax.bar(names, values, color=["#4C72B0", "#DD8452"][:len(values)])
        ax.bar_label(bars, fmt="%.1f")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(model_name, args, class_names, splits, split_mode):
    img_size = IMG_SIZES[model_name]
    out_dir = config.MODELS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(config.SEED)

    train_ds = make_dataset(*splits["train"], img_size, training=True)
    val_ds = make_dataset(*splits["val"], img_size, training=False)
    test_ds = make_dataset(*splits["test"], img_size, training=False)
    num_classes = len(class_names)
    weights = class_weights(np.array(splits["train"][1]), num_classes)

    start = time.time()
    if model_name == "cnn":
        title = "Own CNN (from scratch)"
        model = build_cnn(num_classes, img_size)
        model.summary()
        histories = train_cnn(model, train_ds, val_ds, weights, args.epochs_cnn)
    else:
        title = "MobileNetV2 (transfer learning)"
        model, base = build_mobilenet(num_classes, img_size,
                                      None if args.no_pretrained else "imagenet")
        histories = train_mobilenet(model, base, train_ds, val_ds, weights,
                                    args.epochs_frozen, args.epochs_fine_tune)
    train_minutes = (time.time() - start) / 60

    print(f"\n=== {title}: testing on photos it has never seen ===")
    true, pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        pred.extend(np.argmax(probs, axis=1))
        true.extend(labels.numpy())
    true, pred = np.array(true), np.array(pred)
    accuracy = float(np.mean(true == pred))
    matrix = confusion_matrix(true, pred, num_classes)
    report = classification_report(matrix, class_names)

    model.save(out_dir / "gesture_model.keras")
    (out_dir / "labels.txt").write_text("\n".join(class_names) + "\n")
    plot_history(histories, title, out_dir / "training_curves.png")
    plot_confusion_matrix(matrix, class_names, title,
                          out_dir / "confusion_matrix.png")

    print("Exporting TensorFlow Lite model for the Raspberry Pi "
          "(a long list of TensorSpec lines here is normal)...")
    tflite_path = out_dir / "gesture_model.tflite"
    export_tflite(model, tflite_path)
    tflite_accuracy, tflite_ms = evaluate_tflite(tflite_path, test_ds)

    result = {
        "model": model_name,
        "title": title,
        "input_size": img_size,
        "parameters": int(model.count_params()),
        "keras_accuracy": accuracy,
        "tflite_accuracy": tflite_accuracy,
        "tflite_ms": tflite_ms,
        "tflite_mb": tflite_path.stat().st_size / 1e6,
        "train_minutes": train_minutes,
        "split": split_mode,
        "images": {k: len(v[0]) for k, v in splits.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    summary = (
        f"{title}\n"
        f"Split: {split_mode}\n"
        f"Photos: train {result['images']['train']}, validation "
        f"{result['images']['val']}, test {result['images']['test']}\n"
        f"Parameters: {result['parameters']:,}   input {img_size}x{img_size}\n"
        f"Training time: {train_minutes:.1f} min\n"
        f"Test accuracy (Keras model):  {accuracy:.3f}\n"
        f"Test accuracy (TFLite model): {tflite_accuracy:.3f}\n"
        f"TFLite inference time on this computer: {tflite_ms:.1f} ms per image\n"
        f"TFLite model size: {result['tflite_mb']:.2f} MB\n\n"
        f"{report}\n\nConfusion matrix (rows = true, columns = predicted):\n"
        f"{class_names}\n{matrix}\n"
    )
    (out_dir / "report.txt").write_text(summary)
    print("\n" + summary)
    print(f"Saved everything for {title} in {out_dir}")
    return result


def deploy(model_name):
    """Copy models/<name>/ files to models/, where detect.py loads them."""
    src = config.MODELS_DIR / model_name
    for name in ("gesture_model.tflite", "gesture_model.keras", "labels.txt",
                 "report.txt", "training_curves.png", "confusion_matrix.png"):
        shutil.copy2(src / name, config.MODELS_DIR / name)
    (config.MODELS_DIR / "deployed_model.txt").write_text(model_name + "\n")
    print(f"\nDeployed '{model_name}': models/gesture_model.tflite is now that "
          "model. Send it to the Pi with send-model-to-pi.ps1.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["cnn", "mobilenet", "both"],
                        default="both")
    parser.add_argument("--deploy", choices=["none", "best", "cnn", "mobilenet"],
                        default="none",
                        help="copy a trained model to models/ for detect.py")
    parser.add_argument("--data-dir", default=str(config.DATASET_DIR),
                        help="folder with one sub-folder of images per gesture")
    parser.add_argument("--test-session", default=None,
                        help="session used as the test set (default: newest)")
    parser.add_argument("--epochs-cnn", type=int, default=config.EPOCHS_CNN)
    parser.add_argument("--epochs-frozen", type=int, default=config.EPOCHS_FROZEN)
    parser.add_argument("--epochs-fine-tune", type=int,
                        default=config.EPOCHS_FINE_TUNE)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="MobileNetV2 from random weights (testing only)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"No dataset at {data_dir}. Run collect_data.py first.")
    class_names, files = find_images(data_dir)
    if len(class_names) < 2:
        raise SystemExit("Need pictures for at least 2 gestures.")
    missing = [g for g in config.GESTURES if g not in class_names]
    if missing:
        print(f"WARNING: no pictures for {missing}; they will not be learned.")

    splits, sessions, split_mode = split_files(class_names, files,
                                               args.test_session)
    print("Photos per gesture and session:")
    for name in class_names:
        per_session = {s: sum(1 for it in files[name] if it[1] == s)
                       for s in sessions}
        print(f"  {name:10s} {len(files[name]):5d}   {per_session}")
    print(f"\n{split_mode}")
    print(f"train {len(splits['train'][0])}, validation {len(splits['val'][0])}, "
          f"test {len(splits['test'][0])} photos")
    if not splits["test"][0] or not splits["val"][0]:
        raise SystemExit("Not enough photos to make a validation and test set.")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    names = ["cnn", "mobilenet"] if args.model == "both" else [args.model]
    results = [run(n, args, class_names, splits, split_mode) for n in names]

    if len(results) > 1:
        plot_comparison(results, config.MODELS_DIR / "comparison.png")
        lines = ["Model comparison (same photos, same test session)", ""]
        for r in results:
            lines.append(f"{r['title']:34s} test accuracy "
                         f"{100 * r['tflite_accuracy']:5.1f}%   "
                         f"{r['tflite_mb']:5.2f} MB   {r['tflite_ms']:5.1f} ms   "
                         f"{r['parameters']:>10,} parameters")
        text = "\n".join(lines) + "\n"
        (config.MODELS_DIR / "comparison.txt").write_text(text)
        print("\n" + text)

    if args.deploy != "none":
        if args.deploy == "best":
            choice = max(results, key=lambda r: r["tflite_accuracy"])["model"]
        else:
            choice = args.deploy
        if not (config.MODELS_DIR / choice / "gesture_model.tflite").exists():
            raise SystemExit(f"No trained '{choice}' model to deploy yet.")
        deploy(choice)


if __name__ == "__main__":
    main()
