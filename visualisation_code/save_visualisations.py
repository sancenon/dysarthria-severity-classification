import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score, precision_score, recall_score
)

def save_visualisations(df_test, base_out, trainer=None):
    """
    Create comprehensive visualizations and metrics for model evaluation.
    Saves plots and reports to base_out directory.
    """
    
    # Create training curves if trainer data is available
    if trainer:
        logs = pd.DataFrame(trainer.state.log_history)

        # Plot loss curves
        train_loss = logs[logs.get("loss").notna()] if "loss" in logs else pd.DataFrame()
        val_loss = logs[logs.get("eval_loss").notna()] if "eval_loss" in logs else pd.DataFrame()

        plt.figure(figsize=(9, 4))
        if not train_loss.empty:
            plt.plot(train_loss["epoch"], train_loss["loss"], marker="o", label="Train Loss")
        if not val_loss.empty:
            plt.plot(val_loss["epoch"], val_loss["eval_loss"], marker="o", label="Val Loss")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss per Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_out / "loss_curve.png")
        plt.show()
        plt.close()

        # Plot accuracy curves
        val_acc = logs[logs.get("eval_accuracy").notna()] if "eval_accuracy" in logs else pd.DataFrame()
        train_acc = logs[logs.get("train_accuracy").notna()] if "train_accuracy" in logs else pd.DataFrame()

        plt.figure(figsize=(9, 4))
        if not train_acc.empty:
            plt.plot(train_acc["epoch"], train_acc["train_accuracy"], marker="o", label="Train Accuracy")
        if not val_acc.empty:
            plt.plot(val_acc["epoch"], val_acc["eval_accuracy"], marker="o", label="Val Accuracy")
        
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_out / "accuracy_curve.png")
        plt.show()
        plt.close()

    # Make sure we have the required columns
    if not {"pred", "true"}.issubset(df_test.columns):
        raise ValueError("df_test must have columns: 'pred' and 'true' (integer labels 0-3).")

    # Set up our class labels - these map to severity levels
    label_map = {0: "normal/mild", 1: "moderate", 2: "severe", 3: "profound"}
    classes = [0, 1, 2, 3]
    id2label = label_map
    display_names = [id2label[i] for i in classes]

    y_true = df_test["true"].to_numpy()
    y_pred = df_test["pred"].to_numpy()

    # Generate overall classification report
    report_txt = classification_report(
        y_true, y_pred,
        labels=classes,
        target_names=display_names,
        zero_division=0
    )
    report_path = base_out / "classification_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")
    print("\n=== Classification Report (Overall) ===")
    print(report_txt)

    # Create confusion matrix visualization
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    plt.figure(figsize=(6, 6))
    disp.plot(values_format='d', cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    cm_path = base_out / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.show()
    plt.close()

    # Print confusion matrix as tables
    print("\n=== Confusion Matrix (Raw Counts) ===")
    print(pd.DataFrame(
        cm,
        index=[f"true_{id2label[i]}" for i in classes],
        columns=[f"pred_{id2label[i]}" for i in classes]
    ).to_string())

    # Show row-normalized version (percentages)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

    print("\n=== Confusion Matrix (Percentages) ===")
    print(pd.DataFrame(
        np.nan_to_num(cm_norm),
        index=[f"true_{id2label[i]}" for i in classes],
        columns=[f"pred_{id2label[i]}" for i in classes]
    ).round(3).to_string())

    # Overall test accuracy
    final_test_acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Final Test Accuracy: {final_test_acc:.4f} ===")

    # Helper function to clean up source names for filenames
    def slugify(s: str):
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")

    # Break down metrics by data source (if available)
    if "source" in df_test.columns:
        print("\n=== Metrics by Data Source ===")
        by_src_rows = []
        
        for src, g in df_test.groupby("source", dropna=False):
            src_name = "NA" if pd.isna(src) else str(src)
            y_t = g["true"].to_numpy()
            y_p = g["pred"].to_numpy()

            # Calculate metrics for this source
            acc = accuracy_score(y_t, y_p)
            f1m = f1_score(y_t, y_p, average="macro", zero_division=0)
            prec = precision_score(y_t, y_p, average="macro", zero_division=0)
            rec = recall_score(y_t, y_p, average="macro", zero_division=0)

            # Save detailed report for this source
            src_report = classification_report(
                y_t, y_p,
                labels=classes,
                target_names=display_names,
                zero_division=0
            )
            (base_out / f"classification_report_source_{slugify(src_name)}.txt").write_text(src_report, encoding="utf-8")

            # Create confusion matrix for this source
            cm_s = confusion_matrix(y_t, y_p, labels=classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_s, display_labels=display_names)
            plt.figure(figsize=(6, 6))
            disp.plot(values_format='d', cmap="Blues")
            plt.title(f"Confusion Matrix (Test) — source={src_name}")
            plt.tight_layout()
            plt.savefig(base_out / f"confusion_matrix_source_{slugify(src_name)}.png")
            plt.show()
            plt.close()

            by_src_rows.append({
                "source": src_name,
                "n_samples": len(g),
                "accuracy": acc,
                "macro_f1": f1m,
                "precision_macro": prec,
                "recall_macro": rec,
            })

        by_src_df = pd.DataFrame(by_src_rows).sort_values(["source"])
        by_src_df.to_csv(base_out / "metrics_by_source.csv", index=False)
        print(by_src_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\n(No 'source' column found - skipping source-specific metrics)")

    # Speaker-level analysis using majority voting
    if "speaker_id" in df_test.columns:
        print("\n=== Speaker-level Analysis (Majority Vote) ===")
        speaker_results = []
        
        # For each speaker, use majority vote to get final prediction
        for session_code, group in df_test.groupby("session_code"):
            speaker_pred = group["pred"].mode()[0]  # Most common prediction
            speaker_true = group["true"].iloc[0]    # Actual label (should be same for all clips)
            speaker_id = group["speaker_id"].iloc[0]
            
            speaker_results.append({
                "speaker_id": speaker_id,
                "session_code": session_code,
                "speaker_pred": speaker_pred,
                "speaker_true": speaker_true,
                "is_correct": speaker_pred == speaker_true,
                "n_clips": len(group),
                "pred_name": id2label.get(int(speaker_pred), str(speaker_pred)),
                "true_name": id2label.get(int(speaker_true), str(speaker_true)),
            })

        df_speaker = pd.DataFrame(speaker_results)
        speaker_accuracy = df_speaker["is_correct"].mean()
        
        # Also calculate per-speaker clip-level accuracy
        speaker_clip_accuracy = df_test.groupby('session_code').apply(
            lambda x: (x['pred'] == x['true']).mean()
        ).sort_index()
        per_speaker_acc = speaker_clip_accuracy.mean()

        # Generate speaker-level classification report
        y_true_speaker = df_speaker["speaker_true"]
        y_pred_speaker = df_speaker["speaker_pred"]

        speaker_report_txt = classification_report(
            y_true_speaker, y_pred_speaker,
            labels=classes,
            target_names=display_names,
            zero_division=0
        )
        speaker_report_path = base_out / "speaker_classification_report.txt"
        speaker_report_path.write_text(speaker_report_txt, encoding="utf-8")
        print("\n=== Speaker Classification Report ===")
        print(speaker_report_txt)

        # Save speaker results
        df_speaker_csv_path = base_out / "metrics_by_speaker.csv"
        df_speaker.to_csv(df_speaker_csv_path, index=False)

        speaker_summary_path = base_out / "speaker_accuracy_report.txt"
        with open(speaker_summary_path, "w", encoding="utf-8") as f:
            f.write("=== Speaker-level Accuracy (Majority Vote) on TEST set ===\n\n")
            f.write(df_speaker.to_string(index=False))
            f.write("\n\n")
            f.write(f"Overall Speaker Accuracy (Majority Vote): {speaker_accuracy:.4f}\n")
            f.write(f"Per-Speaker Accuracy (Majority Vote): {per_speaker_acc:.4f}\n")

        print(df_speaker.to_string(index=False))
        print(f"\nOverall Speaker Accuracy (Majority Vote): {speaker_accuracy:.4f}")

        # Create detailed per-speaker accuracy visualization
        def per_speaker_stats(x):
            """Calculate detailed stats for each speaker"""
            n = len(x)
            correct = (x["pred"] == x["true"]).sum()
            acc = correct / n
            # Track what the model predicted when it was wrong
            misclassified = x.loc[x["pred"] != x["true"], "pred"].value_counts().to_dict()
            return pd.Series({
                "accuracy": acc,
                "n_clips": n,
                "correct": correct,
                "label": x["true"].iloc[0],
                "misclassified": misclassified
            })

        speaker_stats = df_test.groupby("session_code").apply(per_speaker_stats).reset_index()
        speaker_stats = speaker_stats.sort_values(["label", "session_code"])

        # Color scheme for different severity levels
        label_colors = {
            0: '#90EE90',  # Light green for Normal/Mild
            1: '#FFFF99',  # Light yellow for Moderate
            2: '#FFB74D',  # Orange for Severe
            3: '#FF6B6B',  # Red for Profound
        }

        # Create vertical stacked bar chart
        fig, ax = plt.subplots(figsize=(max(8, len(speaker_stats) * 0.6), 6))

        for bar_pos, (i, row) in enumerate(speaker_stats.iterrows()):
            total = row["n_clips"]
            correct = row["correct"]
            base = 0
            
            # Correct predictions (solid color)
            ax.bar(bar_pos, correct,
                   color=label_colors[row["label"]],
                   edgecolor="black", linewidth=0.8)
            base += correct
            
            # Incorrect predictions (hatched, colored by what model predicted)
            for pred_label, count in row["misclassified"].items():
                ax.bar(bar_pos, count, bottom=base,
                       color=label_colors[pred_label],
                       edgecolor="black", linewidth=0.8,
                       hatch="///")  # Hatching indicates incorrect
                base += count
            
            # Show accuracy percentage on top of each bar
            ax.text(bar_pos, total + 0.2, f"{row['accuracy']*100:.0f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="black")

        # Styling
        ax.set_title("Per-Speaker Clip Accuracy", fontsize=14, fontweight="bold")
        ax.set_xlabel("Speaker ID (grouped by severity)", fontsize=12)
        ax.set_ylabel("Number of Clips", fontsize=12)
        ax.set_xticks(range(len(speaker_stats)))
        ax.set_xticklabels(speaker_stats["session_code"], rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0, speaker_stats["n_clips"].max() + 1)

        # Legend explaining colors and patterns
        legend_elements = [
            Patch(facecolor=label_colors[0], edgecolor="black", label="Normal/Mild"),
            Patch(facecolor=label_colors[1], edgecolor="black", label="Moderate"),
            Patch(facecolor=label_colors[2], edgecolor="black", label="Severe"),
            Patch(facecolor=label_colors[3], edgecolor="black", label="Profound"),
            Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="Incorrect (hatched)")
        ]
        ax.legend(handles=legend_elements, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)

        plt.tight_layout()
        speaker_plot_path = base_out / "speaker_clip_accuracy_fixed.png"
        plt.savefig(speaker_plot_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

        # Create horizontal version (better for many speakers)
        fig, ax = plt.subplots(figsize=(10, max(6, len(speaker_stats) * 0.5)))

        # Add severity info to speaker labels for clarity
        severity_names = {0: "Normal", 1: "Moderate", 2: "Severe", 3: "Profound"}
        speaker_labels = [f"{row['session_code']} ({severity_names[row['label']]})" 
                          for _, row in speaker_stats.iterrows()]

        y_positions = range(len(speaker_stats))

        for bar_pos, (i, row) in enumerate(speaker_stats.iterrows()):
            total = row["n_clips"]
            correct = row["correct"]
            base = 0
            
            # Correct predictions
            ax.barh(bar_pos, correct,
                    color=label_colors[row["label"]],
                    edgecolor="black", linewidth=0.8)
            base += correct
            
            # Incorrect predictions
            for pred_label, count in row["misclassified"].items():
                ax.barh(bar_pos, count, left=base,
                        color=label_colors[pred_label],
                        edgecolor="black", linewidth=0.8,
                        hatch="///")
                base += count
            
            # Accuracy percentage at end of bar
            ax.text(total + 0.2, bar_pos, f"{row['accuracy']*100:.0f}%",
                    ha="left", va="center", fontsize=10, fontweight="bold",
                    color="black")

        # Styling for horizontal chart
        ax.set_title("Per-Speaker Clip Accuracy", fontsize=14, fontweight="bold", pad=20)
        ax.set_ylabel("Speaker ID (grouped by severity)", fontsize=12)
        ax.set_xlabel("Number of Clips", fontsize=12)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(speaker_labels)
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, speaker_stats["n_clips"].max() + 2)
        ax.invert_yaxis()  # Put first speaker at top

        # Same legend as before
        ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)

        plt.tight_layout()
        speaker_plot_horizontal_path = base_out / "speaker_clip_accuracy_horizontal.png"
        plt.savefig(speaker_plot_horizontal_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()
    else:
        print("\n(No 'speaker_id' column found - skipping speaker-level analysis)")

    # Summary of what was saved
    print(f"\n=== Files Saved ===")
    print(f"Output directory: {base_out}")
    print(f"Confusion matrix: {cm_path}")
    print(f"Classification report: {report_path}")
    print(f"Training curves: {base_out / 'loss_curve.png'} and {base_out / 'accuracy_curve.png'}")
    
    if "source" in df_test.columns:
        print(f"Source breakdown: {base_out / 'metrics_by_source.csv'}")
    if "speaker_id" in df_test.columns:
        print(f"Speaker analysis: {base_out / 'metrics_by_speaker.csv'}")
        print(f"Speaker plots: {base_out / 'speaker_clip_accuracy_vertical.png'} and horizontal version")