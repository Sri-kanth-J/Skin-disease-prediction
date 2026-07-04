"""
EfficientNetV2S Training Script for Skin Disease Classification
Target: 70%+ accuracy
WSL2 Ubuntu 22.04 / Linux optimized
"""
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import json
import math
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
def setup_gpu():
    print("Configuring GPU...")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print(f"  Found {len(gpus)} GPU(s). Mixed precision enabled.")
            return True
        except RuntimeError as e:
            print(f"  GPU config error: {e}")
            return False
    print("  No GPU found. Training on CPU.")
    return False


# ---------------------------------------------------------------------------
# Warmup + cosine decay schedule
# ---------------------------------------------------------------------------
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup then cosine decay."""

    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr=1e-7):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        cos_progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cos_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * cos_progress)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cos_lr)

    def get_config(self):
        return dict(
            peak_lr=self.peak_lr,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            min_lr=self.min_lr,
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class SkinClassifier:
    def __init__(self, data_dir="balanced_dataset", input_shape=(224, 224, 3)):
        self.data_dir = Path(data_dir).resolve()
        self.input_shape = input_shape
        self.batch_size = 32        # safe for WSL2 NTFS mounts with limited RAM
        self.phase1_epochs = 35     # transfer learning
        self.phase2_epochs = 45     # fine-tuning

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def _validate_and_clean_files(self):
        """Remove zero-byte / missing image files to prevent mid-epoch crashes."""
        VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        removed = 0
        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
            for cls_dir in sorted(split_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in VALID_EXTS:
                        try:
                            if not f.exists() or f.stat().st_size == 0:
                                f.unlink(missing_ok=True)
                                removed += 1
                        except OSError:
                            pass
        if removed:
            print(f"  Removed {removed} broken files.")
        else:
            print("  All files OK.")

    def _count_class_samples(self, split, class_names):
        VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
        counts = {}
        for idx, cls in enumerate(class_names):
            cls_dir = self.data_dir / split / cls
            counts[idx] = (
                sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTS)
                if cls_dir.exists() else 0
            )
        return counts

    def _get_common_classes(self):
        split_classes = {}
        for split in ["train", "val", "test"]:
            p = self.data_dir / split
            if not p.exists():
                raise FileNotFoundError(f"Missing split: {p}")
            split_classes[split] = {d.name for d in p.iterdir() if d.is_dir()}
        common = split_classes["train"] & split_classes["val"] & split_classes["test"]
        if not common:
            raise ValueError("No common classes across train/val/test.")
        return sorted(common)

    def _save_class_mapping(self, class_names):
        os.makedirs("models", exist_ok=True)
        mapping = {idx: cls for idx, cls in enumerate(class_names)}
        with open("models/class_indices.json", "w") as f:
            json.dump(mapping, f, indent=2)
        print("  Saved class mapping -> models/class_indices.json")
        # also save human-readable names if dataset provides them
        names_path = Path("dataset") / "class_names.json"
        if names_path.exists():
            try:
                with open(names_path) as nf, open("models/class_names.json", "w") as mf:
                    mf.write(nf.read())
                print("  Copied class_names.json to models/")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Augmentation (GPU-side, [0,255] input — EfficientNetV2S preprocesses internally)
    # ------------------------------------------------------------------
    def _build_augmentation(self):
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"), # Skin lesions can appear at any rotation/orientation
            layers.RandomRotation(0.20),                 # Higher rotation freedom
            layers.RandomZoom((-0.15, 0.15)),
            layers.RandomTranslation(0.15, 0.15),
            layers.RandomBrightness(0.25),                # Robustness against inconsistent clinical lighting
            layers.RandomContrast(0.25),
        ], name="augmentation")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self):
        print(f"\nLoading dataset from: {self.data_dir}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_dir}. Run simple_rebalance.py first.")

        print("Validating files...")
        self._validate_and_clean_files()

        class_names = self._get_common_classes()
        print(f"Classes ({len(class_names)}): {class_names}")

        # EfficientNetV2S has include_preprocessing=True by default,
        # so images should be passed as uint8 / float32 in [0, 255].
        load_kwargs = dict(
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=self.input_shape[:2],
            batch_size=self.batch_size,
        )

        train_ds_raw = tf.keras.utils.image_dataset_from_directory(
            str(self.data_dir / "train"), shuffle=True, seed=42, **load_kwargs
        )
        val_ds_raw = tf.keras.utils.image_dataset_from_directory(
            str(self.data_dir / "val"), shuffle=False, **load_kwargs
        )

        aug = self._build_augmentation()
        train_ds = (
            train_ds_raw
            .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(4)  # Hard bound to 4 instead of AUTOTUNE to prevent OOM memory leaks in WSL2
        )
        val_ds = val_ds_raw.prefetch(4)

        train_counts = self._count_class_samples("train", class_names)
        val_counts = self._count_class_samples("val", class_names)
        # Load human-readable names from dataset if available
        human_names = None
        for names_path in [Path("dataset") / "class_names.json", Path("models") / "class_names.json"]:
            if names_path.exists():
                try:
                    human_names = json.load(open(names_path))
                    break
                except Exception:
                    pass

        print(f"  Train: {sum(train_counts.values())} images | "
              f"Val: {sum(val_counts.values())} images | "
              f"Batch: {self.batch_size}")
        for idx, cls in enumerate(class_names):
            # Map folder name (e.g. "11") to disease name (e.g. "Melanoma")
            folder_idx = int(cls) if cls.isdigit() else idx
            label = human_names[folder_idx] if human_names and folder_idx < len(human_names) else cls
            print(f"    Class {idx} ({label}): train={train_counts[idx]}, val={val_counts[idx]}")
        self._save_class_mapping(class_names)
        return train_ds, val_ds, class_names, len(class_names), train_counts

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def build_model(self, num_classes):
        backbone = EfficientNetV2S(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
            include_preprocessing=True,
        )
        backbone.trainable = False

        inputs = Input(shape=self.input_shape)
        x = backbone(inputs, training=False)

        # Robust classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(
            512, activation="swish",  # Swish matches EfficientNet internals better than ReLU
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

        model = Model(inputs, outputs)
        print(f"  Parameters: {model.count_params():,}")
        return model, backbone

    # ------------------------------------------------------------------
    # Class weights — no hard cap; allow full rebalancing
    # ------------------------------------------------------------------
    def _compute_class_weights(self, train_counts):
        all_labels = []
        for idx, count in train_counts.items():
            all_labels.extend([idx] * count)
        all_labels = np.array(all_labels)

        counts = np.array([train_counts[i] for i in sorted(train_counts)])
        ratio = counts.max() / max(counts.min(), 1)

        if ratio <= 1.2:
            print("  Near-perfect balance — no class weights applied.")
            return None

        weights = compute_class_weight("balanced", classes=np.unique(all_labels), y=all_labels)
        weights = np.clip(weights, 0.5, 6.0) # Adjusted soft cap slightly for structural security
        cw = dict(enumerate(weights))
        print(f"  Balance ratio {ratio:.1f}:1 → class weights: "
              + ", ".join(f"cls{k}={v:.2f}" for k, v in cw.items()))
        return cw
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    def train(self):
        train_ds, val_ds, class_names, num_classes, train_counts = self.load_data()
        model, backbone = self.build_model(num_classes)
        class_weight_dict = self._compute_class_weights(train_counts)

        os.makedirs("models/checkpoints", exist_ok=True)

        total_train = sum(train_counts.values())
        steps_per_epoch = math.ceil(total_train / self.batch_size)

        # ── Phase 1: Frozen backbone ─────────────────────────────────
        print("\n=== Phase 1: Transfer learning (backbone frozen) ===")

        p1_total_steps = steps_per_epoch * self.phase1_epochs
        p1_warmup = steps_per_epoch * 3
        schedule_p1 = WarmupCosineDecay(
        peak_lr=1e-3, total_steps=p1_total_steps,  # Reduced from 3e-3 for better early stability
        warmup_steps=p1_warmup, min_lr=1e-6
        )
        optimizer_p1 = AdamW(learning_rate=schedule_p1, weight_decay=1e-4)

        model.compile(
        optimizer=optimizer_p1,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
        )

        ckpt_p1 = "models/checkpoints/efficientnetv2s_transfer.keras"
        cbs_p1 = [
        EarlyStopping(
        monitor="val_accuracy", patience=10,
        restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
        ckpt_p1, monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1,
        ),
        ]

        model.fit(
             train_ds,
             epochs=self.phase1_epochs,
             validation_data=val_ds,
             callbacks=cbs_p1,
             class_weight=class_weight_dict,
             verbose=1,
        )
        print(f"  Best phase-1 model saved to {ckpt_p1}")

        # ── Phase 2: Fine-tuning ────────────────────────────────────
        print("\n=== Phase 2: Fine-tuning (top 150 backbone layers unfrozen) ===")
        backbone.trainable = True
        # Unfreeze top 150 layers to give the model room to capture specialized skin lesion semantics
        freeze_until = max(0, len(backbone.layers) - 150)

        for layer in backbone.layers[:freeze_until]:
           layer.trainable = False

        for layer in backbone.layers[freeze_until:]:
                # Keep BatchNorm stats protected to prevent degradation of pre-trained parameters
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        p2_total_steps = steps_per_epoch * self.phase2_epochs
        p2_warmup = steps_per_epoch * 3  # Increased warmup for structural normalization safety
        schedule_p2 = WarmupCosineDecay(
            peak_lr=4e-5, total_steps=p2_total_steps,
            warmup_steps=p2_warmup, min_lr=1e-7
            )
        optimizer_p2 = AdamW(learning_rate=schedule_p2,
            weight_decay=1e-3)  # Raised weight decay to combat overfitting

        model.compile(
                optimizer=optimizer_p2,
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                metrics=["accuracy"],
            )

        ckpt_p2 = "models/checkpoints/efficientnetv2s_finetuned.keras"
        cbs_p2 = [
                EarlyStopping(
                    monitor="val_accuracy", patience=12,
                    # Extended patience to give the deeper networks room to converge
                    restore_best_weights=True, verbose=1,
                ),
               ModelCheckpoint(
                    ckpt_p2, monitor="val_accuracy", mode="max",
                    save_best_only=True, verbose=1,
                ),
            ]

        model.fit(
                train_ds,
                epochs=self.phase2_epochs,
                validation_data=val_ds,
                callbacks=cbs_p2,
                class_weight=class_weight_dict,
                verbose=1,
        )
        print(f"  Best phase-2 model saved to {ckpt_p2}")

        model.save("models/gpu_trained_model.keras")
        print("\nTraining complete. Final model saved to models/gpu_trained_model.keras")
        return model

    # ---------------------------------------------------------------------------
    # Entry point
    # ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_gpu()

    data_dir = Path(__file__).resolve().parent / "balanced_dataset"
    if not data_dir.exists():
        print(f"balanced_dataset not found at {data_dir}")
        print("Run: python simple_rebalance.py")
        raise SystemExit(1)

    classifier = SkinClassifier(data_dir=str(data_dir))
    classifier.train()