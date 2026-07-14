import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive: works in WSL2 without a display
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent


class ModelTester:
    def __init__(self, model_path=None, dataset_path=None):
        self.model_path = (
            Path(model_path).resolve() if model_path
            else SCRIPT_DIR / "models" / "checkpoints" / "mobilenetv3_finetuned.keras"
        )
        self.dataset_path = (
            Path(dataset_path).resolve() if dataset_path
            else SCRIPT_DIR / "datasets" / "val"
        )
        self.input_shape = (224, 224)
        self.batch_size = 32

    def _load_class_mapping(self):
        mapping_path = SCRIPT_DIR / "models" / "class_indices.json"
        if not mapping_path.exists():
            print("⚠️ class_indices.json not found. Using directory class order.")
            return None
        with open(mapping_path, 'r', encoding='utf-8') as file:
            mapping = json.load(file)
        ordered_classes = [mapping[str(i)] for i in sorted(int(k) for k in mapping.keys())]
        print(f"✅ Loaded class mapping from {mapping_path}")
        return ordered_classes
        
    def load_model_and_data(self):
        if not self.model_path.exists():
            print(f"Model not found at {self.model_path}")
            ckpt_dir = SCRIPT_DIR / "models" / "checkpoints"
            if ckpt_dir.exists():
                available = list(ckpt_dir.glob("*.keras"))
                if available:
                    print("Available checkpoints:")
                    for f in available:
                        print(f"  {f}")
            raise FileNotFoundError("Run train.py first to create the model.")
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {self.dataset_path}. Run train.py first."
            )

        print(f"Loading model: {self.model_path}")
        model = tf.keras.models.load_model(str(self.model_path), compile=False)

        mapped_classes = self._load_class_mapping()
        print(f"Class order: {mapped_classes}")

        raw_ds = tf.keras.utils.image_dataset_from_directory(
            str(self.dataset_path),
            labels="inferred",
            label_mode="categorical",
            class_names=mapped_classes,
            image_size=self.input_shape,
            batch_size=self.batch_size,
            shuffle=False,
        )

        val_batches = tf.data.experimental.cardinality(raw_ds).numpy()
        split_idx = val_batches // 2

        test_ds = raw_ds.skip(split_idx).prefetch(2)
        print(f"📊 Isolated {val_batches - split_idx} holdout evaluation batches from validation directory.")
        
        true_labels = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
        true_classes = np.argmax(true_labels, axis=1)

        return model, test_ds, true_classes, mapped_classes

    def evaluate(self):
        model, test_ds, true_classes, class_labels = self.load_model_and_data()
        names_path = SCRIPT_DIR / "models" / "class_names.json"
        human_names = None
        if names_path.exists():
            try:
                human_names = json.load(open(names_path))
            except Exception:
                human_names = None
        print("\nTest set distribution:")
        class_counts = {}

        def get_disease_name(cls):
            if human_names:
                folder_idx = int(cls) if cls.isdigit() else None
                if folder_idx is not None and folder_idx < len(human_names):
                    return human_names[folder_idx]
            return cls
        
        for i, cls in enumerate(class_labels):
            count = int(np.sum(true_classes == i))
            name = get_disease_name(cls)
            class_counts[cls] = count
            print(f"  Class {i} ({name}): {count} samples")
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        balance_ratio = max_count / max(min_count, 1)
        print(f"  Balance ratio: {balance_ratio:.1f}:1")

        print("\nGenerating predictions...")
        predictions = model.predict(test_ds, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(true_classes, predicted_classes)
        macro_f1 = f1_score(true_classes, predicted_classes, average='macro')
        weighted_f1 = f1_score(true_classes, predicted_classes, average='weighted')
        
        print("\n" + "="*60)
        print(f"🎯  DATASET TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"🎯 MACRO F1 SCORE: {macro_f1:.4f}")
        print(f"🎯 WEIGHTED F1 SCORE: {weighted_f1:.4f}")
        print("="*60)
        
        print("\n📋 PER-CLASS PERFORMANCE ( Dataset):")
        print("-" * 50)
        for i, class_name in enumerate(class_labels):
            class_mask = (true_classes == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(true_classes[class_mask], predicted_classes[class_mask])
                class_count = np.sum(class_mask)
                status = "🟢" if class_acc > 0.7 else "🟡" if class_acc > 0.5 else "🔴"
                display_name = get_disease_name(class_name)
                print(f"   {status} Class {display_name}: {class_acc:.3f} ({class_count} samples)")
        
        print("\nDETAILED CLASSIFICATION REPORT:")
        report_names = [get_disease_name(cls) for cls in class_labels]
        report_labels = list(range(len(class_labels)))
        print(classification_report(true_classes, predicted_classes, labels=report_labels, target_names=report_names, zero_division=0))

        self.plot_confusion_matrix(true_classes, predicted_classes, class_labels, get_disease_name)

        out_dir = SCRIPT_DIR / "models"
        out_dir.mkdir(exist_ok=True)
        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "num_classes": int(len(class_labels)),
            "num_samples": int(len(true_classes)),
            "classes": class_labels,
        }
        metrics_path = out_dir / "test_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        
        print(f"\n💡  DATASET RESULTS SUMMARY:")
        print(f"   • Total accuracy: {accuracy*100:.1f}%")
        print(f"   • Macro F1: {macro_f1:.3f}")
        print(f"   • Test set balance: {balance_ratio:.1f}:1")
        print(f"   • Classes tested: {len(class_labels)}")
        print(f"   • Total test samples: {len(true_classes)}")
        
        if accuracy > 0.8:
            print("   🏆 Excellent performance!  dataset worked well.")
        elif accuracy > 0.6:
            print("   ✅ Good improvement with  dataset!")
        elif accuracy > 0.4:
            print("   📈 Better than original im dataset.")
        else:
            print("   ⚠️ Performance still low - check data quality.")

    def plot_confusion_matrix(self, true_classes, predicted_classes, class_labels, get_name_fn=None):
        cm = confusion_matrix(true_classes, predicted_classes, labels=list(range(len(class_labels))))
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)

        display_labels = [get_name_fn(c) if get_name_fn else c for c in class_labels]

        plt.figure(figsize=(12, 10))
        plt.imshow(cm_normalized, cmap="Blues")
        plt.colorbar(label="Recall per class")
        plt.xticks(range(len(display_labels)), display_labels, rotation=45, ha="right")
        plt.yticks(range(len(display_labels)), display_labels)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i,
                    f"{cm[i,j]}\n({cm_normalized[i,j]:.2f})",
                    ha="center", va="center", fontsize=8, color="black",
                )

        overall_acc = np.trace(cm) / np.sum(cm)
        plt.title(f"Confusion Matrix  (accuracy={overall_acc:.3f})", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted", fontsize=11)
        plt.ylabel("True", fontsize=11)
        plt.tight_layout()

        out_path = SCRIPT_DIR / "models" / "_confusion_matrix.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {out_path}")

if __name__ == "__main__":
    print(" Dataset Model Testing")
    print("=" * 60)

    test_dir = SCRIPT_DIR / "datasets" / "val"
    if not test_dir.exists():
        print(f"Test data not found at {test_dir}")
        print("Run: python simple_rebalance.py")
        raise SystemExit(1)

    tester = ModelTester()
    print(f"Model   : {tester.model_path}")
    print(f"Test dir: {tester.dataset_path}")
    tester.evaluate()
    print("\nTesting complete. Check models/_confusion_matrix.png for the confusion matrix.")