"""Fashion-MNIST: первичное обучение и последующее дообучение.

Запуск: python experiment.py baseline
Позднее: python experiment.py finetune --new-data new_images
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
# Все загрузки и кэши остаются внутри проекта.
os.environ.setdefault("KERAS_HOME", str(ROOT / "data" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "data" / "cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps

CLASSES = ["Футболка", "Брюки", "Пуловер", "Платье", "Пальто",
           "Сандалии", "Рубашка", "Кроссовки", "Сумка", "Ботильоны"]
FOLDERS = ["0_tshirt", "1_trouser", "2_pullover", "3_dress", "4_coat",
           "5_sandal", "6_shirt", "7_sneaker", "8_bag", "9_ankle_boot"]
SEED = 42


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_data():
    """Одинаковая стратифицированная выборка при каждом запуске."""
    (x, y), (xt, yt) = tf.keras.datasets.fashion_mnist.load_data()
    split_path = ROOT / "data" / "split.npz"
    if split_path.exists():
        with np.load(split_path) as split:
            train, val = split["train"], split["val"]
    else:
        rng = np.random.default_rng(SEED)
        train, val = [], []
        for label in range(10):
            indices = rng.permutation(np.flatnonzero(y == label))
            train.extend(indices[:5000])
            val.extend(indices[5000:])
        train = rng.permutation(train)
        val = rng.permutation(val)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(split_path, train=train, val=val)
    assert len(train) == 50000 and len(val) == 10000
    assert len(np.unique(np.concatenate([train, val]))) == 60000
    assert not np.intersect1d(train, val).size
    assert np.array_equal(np.bincount(y[train]), np.full(10, 5000))
    assert np.array_equal(np.bincount(y[val]), np.full(10, 1000))
    fingerprints = {"train_indices": digest(train), "validation_indices": digest(val),
                    "test_images": digest(xt), "test_labels": digest(yt)}
    normalize = lambda a: a.astype("float32")[..., None] / 255.0
    return (normalize(x[train]), y[train], normalize(x[val]), y[val],
            normalize(xt), yt, fingerprints)


def build_model():
    # Размеры совпадают с примером: 28 -> 14 -> 7 -> 588 -> 64 -> 10.
    return tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1), name="image"),
        tf.keras.layers.Conv2D(6, 3, padding="same", kernel_initializer="he_normal", name="conv1"),
        tf.keras.layers.ReLU(name="relu1"),
        tf.keras.layers.MaxPooling2D(2, name="pool1"),
        tf.keras.layers.Conv2D(12, 3, padding="same", kernel_initializer="he_normal", name="conv2"),
        tf.keras.layers.ReLU(name="relu2"),
        tf.keras.layers.MaxPooling2D(2, name="pool2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(64, kernel_initializer="he_normal", name="dense64"),
        tf.keras.layers.ReLU(name="relu3"),
        tf.keras.layers.Dense(10, name="logits"),
        tf.keras.layers.Softmax(name="probabilities"),
    ], name="fashion_cnn")


def dataset(x, y, batch, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(y), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    return ds.batch(batch).with_options(options).prefetch(1)


def plot_curves(history, out, selected_epoch):
    epochs = np.arange(1, len(history["loss"]) + 1)
    for key, title, ylabel in [("accuracy", "Точность распознавания", "Accuracy"),
                                ("loss", "Ошибка классификации", "Кросс-энтропия")]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(epochs, history[key], label="Обучение", marker="o", ms=3)
        ax.plot(epochs, history["val_" + key], label="Валидация", marker="o", ms=3)
        ax.axvline(selected_epoch, color="#777777", ls="--", label=f"Выбрана эпоха {selected_epoch}")
        ax.set(title=title, xlabel="Эпоха", ylabel=ylabel)
        ax.set_xticks(epochs)
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{key}.png", dpi=180)
        plt.close(fig)


def plot_examples(x, y, predicted, probs, indices, path, title):
    indices = indices[:16]
    if not len(indices):
        return
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, indices):
        ax.imshow(x[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Метка: {CLASSES[y[i]]}\nОтвет: {CLASSES[predicted[i]]}\n"
                     f"p = {probs[i, predicted[i]]:.2f}", fontsize=9,
                     color="#176a43" if y[i] == predicted[i] else "#a32929")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def export_architecture(model, out):
    with (out / "architecture.txt").open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    with (out / "layers.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Слой", "Тип", "Размер выхода", "Параметры"])
        for layer in model.layers:
            writer.writerow([layer.name, type(layer).__name__, str(tuple(layer.output.shape)), layer.count_params()])
    boxes = ["Вход\n28 × 28 × 1", "Conv 3×3, 6 + ReLU\n28 × 28 × 6",
             "MaxPool 2×2\n14 × 14 × 6", "Conv 3×3, 12 + ReLU\n14 × 14 × 12",
             "MaxPool 2×2\n7 × 7 × 12", "Flatten\n588", "Dense 64 + ReLU\n64",
             "Dense 10 + Softmax\n10 вероятностей"]
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for i, label in enumerate(boxes):
        y = .94 - i * .125
        ax.text(.5, y, label, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round,pad=.6", facecolor="#eef3fa", edgecolor="#6483a7"))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(.5, y - .083), xytext=(.5, y - .045),
                        arrowprops=dict(arrowstyle="->", color="#444444"))
    fig.tight_layout()
    fig.savefig(out / "architecture.png", dpi=160)
    plt.close(fig)


def evaluate_and_export(model, x, y, batch, out):
    scores = model.evaluate(dataset(x, y, batch), verbose=0, return_dict=True)
    probs = model.predict(dataset(x, y, batch), verbose=0)
    assert probs.shape == (10000, 10) and np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    predicted = probs.argmax(axis=1)
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, predicted), 1)
    assert cm.sum() == 10000
    assert abs(np.mean(predicted == y) - scores["accuracy"]) < 1e-6
    np.savez_compressed(out / "test_predictions.npz", labels=y, predicted=predicted, probabilities=probs)
    np.savetxt(out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    with (out / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Класс", "Количество", "Верно", "Recall", "Precision", "F1"])
        for i, name in enumerate(CLASSES):
            recall = float(cm[i, i] / cm[i].sum())
            precision = float(cm[i, i] / cm[:, i].sum()) if cm[:, i].sum() else 0.
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            writer.writerow([name, cm[i].sum(), cm[i, i], recall, precision, f1])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Ответ модели", ylabel="Правильный класс", title="Матрица ошибок на тестовой выборке")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    plot_examples(x, y, predicted, probs, np.arange(16), out / "predictions.png", "Первые 16 тестовых изображений")
    plot_examples(x, y, predicted, probs, np.flatnonzero(predicted != y), out / "errors.png", "Примеры ошибок модели")
    weights = model.get_layer("conv1").get_weights()[0][:, :, 0, :]
    fig, axes = plt.subplots(1, 6, figsize=(10, 2.5))
    scale = abs(weights).max()
    for i, ax in enumerate(axes):
        ax.imshow(weights[:, :, i], cmap="coolwarm", vmin=-scale, vmax=scale)
        ax.set_title(f"Фильтр {i + 1}")
        ax.axis("off")
    fig.suptitle("Обученные ядра первого свёрточного слоя, 3×3")
    fig.tight_layout()
    fig.savefig(out / "kernels.png", dpi=180)
    plt.close(fig)
    return {k: float(v) for k, v in scores.items()}, cm


def load_new_images(directory, invert):
    """Метки заданы папками. Автоинверсия не применяется: решение принимает пользователь."""
    images, labels, manifest = [], [], []
    seen = set()
    for i, folder in enumerate(FOLDERS):
        for path in sorted((directory / folder).glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("L")
                if invert:
                    im = ImageOps.invert(im)
                im = ImageOps.pad(im, (28, 28), method=Image.Resampling.LANCZOS, color=0)
                a = np.asarray(im, dtype="float32") / 255.0
            sha = digest(a)
            if sha in seen:
                raise ValueError(f"Дубликат нового изображения после обработки: {path}")
            seen.add(sha)
            images.append(a[..., None])
            labels.append(i)
            manifest.append({"path": str(path.resolve()), "label": i, "sha256_preprocessed": sha})
    if not images:
        raise ValueError(f"Нет изображений в {directory}. Разложите их по папкам классов из README.md.")
    return np.stack(images), np.asarray(labels, dtype="uint8"), manifest


def summary_text(metrics, cm):
    worst = int(np.argmin(np.diag(cm) / cm.sum(axis=1)))
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    actual, guessed = np.unravel_index(off_diagonal.argmax(), cm.shape)
    return ("Первичный эксперимент Fashion-MNIST\n" if metrics["stage"] == "baseline" else "Дообучение Fashion-MNIST\n") + (
        f"Обучение: {metrics['training_count']} изображений; валидация: 10000; тест: 10000.\n"
        f"Выполнено эпох: {metrics['epochs']}. Выбрана эпоха {metrics['selected_epoch']} по минимуму val_loss.\n"
        f"Тестовая точность: {metrics['test']['accuracy']:.2%}.\n"
        f"Тестовая кросс-энтропия: {metrics['test']['loss']:.6f}.\n"
        f"Правильно: {np.trace(cm)} из {cm.sum()}; ошибок: {cm.sum() - np.trace(cm)}.\n"
        f"Минимальная полнота у класса «{CLASSES[worst]}»: {cm[worst,worst]/cm[worst].sum():.2%}.\n"
        f"Самая частая направленная путаница: «{CLASSES[actual]}» → «{CLASSES[guessed]}», {cm[actual,guessed]} случаев.\n"
        "Это результаты одного запуска с seed=42, а не среднее по нескольким экспериментам.\n"
        "Графики показывают все эпохи; тест и сохранённая best.keras относятся к выбранной эпохе.\n")


def compare(before, after, out):
    assert before["fingerprints"] == after["fingerprints"], "Выборки изменились!"
    delta = (after["test"]["accuracy"] - before["test"]["accuracy"]) * 100
    text = (f"Точность до: {before['test']['accuracy']:.2%}\n"
            f"Точность после: {after['test']['accuracy']:.2%}\n"
            f"Изменение: {delta:+.2f} процентного пункта\n"
            f"Loss до: {before['test']['loss']:.6f}\nLoss после: {after['test']['loss']:.6f}\n"
            "Эффект включает дополнительное обучение и новые изображения. Чтобы отделить эффект новых данных,\n"
            "нужен контрольный запуск с тем же числом эпох без новых изображений.\n")
    (out / "comparison.txt").write_text(text, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Тестовая точность", "Тестовая ошибка"]):
        values = [before["test"][key], after["test"][key]]
        ax.bar(["До", "После"], values, color=["#6483a7", "#42957d"])
        if key == "accuracy":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    old = json.loads((ROOT / "results" / "before" / "history.json").read_text())
    new = json.loads((out / "history.json").read_text())
    n = len(old["loss"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in zip(axes, ["accuracy", "loss"]):
        for prefix, label, color in [("", "Обучение", "#6483a7"), ("val_", "Валидация", "#d28643")]:
            ax.plot(range(1, n + 1), old[prefix + key], color=color, label=label)
            ax.plot(range(n + 1, n + len(new[key]) + 1), new[prefix + key], color=color, ls="--")
        ax.axvline(n + .5, color="gray", ls=":", label="Начало дообучения")
        ax.set(xlabel="Эпоха", ylabel=key)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Сплошная линия — первый этап; пунктир — дообучение")
    fig.tight_layout()
    fig.savefig(out / "combined_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "finetune"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--new-data", type=Path, default=ROOT / "new_images")
    parser.add_argument("--invert", action="store_true", help="Инвертировать новые изображения с белым фоном")
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (15 if args.stage == "baseline" else 5)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Число эпох и размер батча должны быть положительными")
    out = ROOT / "results" / ("before" if args.stage == "baseline" else "after")
    if out.exists():
        parser.error(f"Папка уже существует: {out}. Сначала сохраните её под другим именем; перезапись запрещена.")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    x, y, xv, yv, xt, yt, fingerprints = load_data()
    before = None
    manifest = []
    if args.stage == "baseline":
        model = build_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        source = ROOT / "results" / "before"
        if not (source / "metrics.json").exists():
            parser.error("Сначала выполните baseline")
        before = json.loads((source / "metrics.json").read_text())
        if before["fingerprints"] != fingerprints:
            parser.error("Исходные выборки изменились. Сравнение невозможно.")
        extra_x, extra_y, manifest = load_new_images(args.new_data, args.invert)
        # Новые изображения не должны повторять исходные данные, в том числе тестовые.
        (raw_x, _), (raw_xt, _) = tf.keras.datasets.fashion_mnist.load_data()
        existing = {digest(a.astype("float32") / 255.) for a in np.concatenate([raw_x, raw_xt])}
        if any(item["sha256_preprocessed"] in existing for item in manifest):
            parser.error("Среди новых изображений есть копии исходного Fashion-MNIST")
        model = tf.keras.models.load_model(source / "best.keras")
        model.optimizer.learning_rate.assign(.0001)
        # Исходные обучающие примеры снижают риск забывания ранее изученных классов.
        x, y = np.concatenate([x, extra_x]), np.concatenate([y, extra_y])
    assert model.count_params() == 39066
    out.mkdir(parents=True)
    save_json(out / "new_images_manifest.json", manifest)
    export_architecture(model, out)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    print(f"Этап: {args.stage}; обучение: {len(y)}; валидация: {len(yv)}; тест: {len(yt)}", flush=True)
    history = model.fit(dataset(x, y, args.batch_size, True),
                        validation_data=dataset(xv, yv, args.batch_size),
                        epochs=args.epochs, callbacks=callbacks, verbose=2).history
    history = {k: [float(v) for v in values] for k, values in history.items()}
    assert all(np.all(np.isfinite(values)) for values in history.values())
    save_json(out / "history.json", history)
    model.save(out / "last.keras")
    selected = int(np.argmin(history["val_loss"])) + 1
    model = tf.keras.models.load_model(out / "best.keras")
    scores, cm = evaluate_and_export(model, xt, yt, args.batch_size, out)
    metrics = {"stage": args.stage, "seed": SEED, "epochs": args.epochs,
               "selected_epoch": selected, "selection_rule": "minimum validation loss",
               "training_count": len(y), "validation_count": len(yv), "test_count": len(yt),
               "new_images_count": len(manifest), "batch_size": args.batch_size,
               "optimizer": "Adam", "learning_rate": float(model.optimizer.learning_rate.numpy()),
               "parameters": model.count_params(), "fingerprints": fingerprints, "test": scores,
               "selected_validation": {k: values[selected - 1] for k, values in history.items()},
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                            "numpy": np.__version__, "keras": tf.keras.__version__}}
    save_json(out / "metrics.json", metrics)
    plot_curves(history, out, selected)
    text = summary_text(metrics, cm)
    (out / "results.txt").write_text(text, encoding="utf-8")
    if before:
        compare(before, metrics, out)
    print(text, flush=True)
    print(f"Сохранено в {out}", flush=True)


if __name__ == "__main__":
    main()
