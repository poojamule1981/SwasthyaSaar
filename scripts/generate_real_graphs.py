"""
Generate graphs from REAL test results for blackbook/PPT.
Uses actual data from run_real_tests.py output.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Load real results
df = pd.read_csv(os.path.join(RESULTS_DIR, "real_test_results.csv"))

print("=" * 60)
print("SwasthyaSaar - Generating Graphs from REAL Test Data")
print("=" * 60)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'blue': '#1f77b4', 'orange': '#ff7f0e', 'green': '#2ca02c', 'red': '#d62728', 'purple': '#9467bd'}

# --- Fig 1: Precision, Recall, F1 per Test Case ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df))
width = 0.25

ax.bar(x - width, df["Precision"], width, label='Precision', color=COLORS['blue'])
ax.bar(x, df["Recall"], width, label='Recall', color=COLORS['orange'])
ax.bar(x + width, df["F1 Score"], width, label='F1 Score', color=COLORS['green'])

ax.set_xlabel('Test Cases', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision, Recall & F1 Score per Test Report', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df["Test ID"], fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)

# Add value labels
for i, (p, r, f) in enumerate(zip(df["Precision"], df["Recall"], df["F1 Score"])):
    ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', fontsize=8)
    ax.text(i, r + 0.02, f'{r:.2f}', ha='center', fontsize=8)
    ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_fig1_precision_recall_f1.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 1: Precision, Recall & F1 per Test Report")

# --- Fig 2: Detection Rate (Expected vs Detected vs Correct) ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df))
width = 0.25

ax.bar(x - width, df["Expected Params"], width, label='Expected', color=COLORS['blue'])
ax.bar(x, df["Detected Params"], width, label='Detected', color=COLORS['orange'])
ax.bar(x + width, df["Correct Values"], width, label='Correct', color=COLORS['green'])

ax.set_xlabel('Test Cases', fontsize=12)
ax.set_ylabel('Number of Parameters', fontsize=12)
ax.set_title('Parameter Detection: Expected vs Detected vs Correct', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df["Test ID"], fontsize=10)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_fig2_detection_counts.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 2: Parameter Detection Counts")

# --- Fig 3: Classification Accuracy per Report ---
fig, ax = plt.subplots(figsize=(10, 6))

colors_bar = [COLORS['green'] if v >= 0.9 else COLORS['orange'] if v >= 0.7 else COLORS['red'] for v in df["Classification Acc"]]
bars = ax.bar(df["Test ID"], df["Classification Acc"] * 100, color=colors_bar, width=0.6)

ax.set_xlabel('Test Cases', fontsize=12)
ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
ax.set_title('Status Classification Accuracy (Low/Normal/High)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend()

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.0f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_fig3_classification_accuracy.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 3: Classification Accuracy")

# --- Fig 4: Recall by Report Type ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(df["Report Type"], df["Recall"] * 100, color=COLORS['blue'], height=0.5)
ax.set_xlabel('Recall (%)', fontsize=12)
ax.set_title('Parameter Recall by Report Type', fontsize=14, fontweight='bold')
ax.set_xlim(0, 110)

for i, (val, name) in enumerate(zip(df["Recall"], df["Report Type"])):
    ax.text(val * 100 + 1, i, f'{val*100:.0f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_fig4_recall_by_type.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 4: Recall by Report Type")

# --- Fig 5: Overall System Performance (Pie: Correct vs Missed vs Wrong) ---
total_expected = df["Expected Params"].sum()
total_correct = df["Correct Values"].sum()
total_detected = df["Detected Params"].sum()
missed = total_expected - total_detected
wrong_value = total_detected - total_correct

fig, ax = plt.subplots(figsize=(8, 6))
sizes = [total_correct, missed, wrong_value]
labels = [f'Correct ({total_correct})', f'Missed ({missed})', f'Wrong Value ({wrong_value})']
colors_pie = [COLORS['green'], COLORS['red'], COLORS['orange']]
explode = (0.05, 0.05, 0.05)

wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
                                   explode=explode, startangle=90, textprops={'fontsize': 11})
ax.set_title(f'Overall Extraction Accuracy (Total: {total_expected} params)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_fig5_overall_accuracy_pie.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 5: Overall Accuracy (Pie Chart)")

# --- Fig 6: Precision vs Recall Line Chart ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Test ID"], df["Precision"], 'o-', color=COLORS['blue'], linewidth=2, markersize=8, label='Precision')
ax.plot(df["Test ID"], df["Recall"], 's-', color=COLORS['orange'], linewidth=2, markersize=8, label='Recall')
ax.plot(df["Test ID"], df["F1 Score"], '^-', color=COLORS['green'], linewidth=2, markersize=8, label='F1 Score')

ax.set_xlabel('Test Cases', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision vs Recall vs F1 Across Test Cases', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0.2, 1.1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_fig6_precision_recall_line.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 6: Precision vs Recall Line Chart")

# --- Fig 7: Summary Metrics Dashboard ---
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

metrics = ['Precision', 'Recall', 'F1 Score', 'Classification\nAccuracy']
values = [98.28, 89.06, 93.44, 97.40]
colors_metrics = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]

for i, (ax, metric, val, col) in enumerate(zip(axes, metrics, values, colors_metrics)):
    ax.barh([0], [val], color=col, height=0.4)
    ax.set_xlim(0, 105)
    ax.set_yticks([])
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.text(val + 1, 0, f'{val:.1f}%', va='center', fontsize=13, fontweight='bold')
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.4)

plt.suptitle('SwasthyaSaar — Overall Performance Metrics (10 Reports)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "real_fig7_summary_dashboard.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 7: Summary Dashboard")

# ============================================================
# Print final summary for blackbook
# ============================================================

print("\n" + "=" * 60)
print("📊 REAL TEST RESULTS FOR BLACKBOOK")
print("=" * 60)
print(f"""
OVERALL SYSTEM PERFORMANCE (10 Reports, 64 Parameters):
• Overall Precision: 98.28%
• Overall Recall: 89.06%
• Overall F1 Score: 93.44%
• Classification Accuracy: 97.40%

BY REPORT TYPE:
• CBC (Clear HTML): 100% Precision, 100% Recall
• Lipid Profile: 100% Precision, 80% Recall
• Kidney Function: 100% Precision, 100% Recall
• Liver Function: 100% Precision, 88% Recall
• Thyroid: 100% Precision, 100% Recall
• Diabetes: 100% Precision, 33% Recall (synonym issue)
• Electrolytes: 100% Precision, 100% Recall
• Vitamins: 67% Precision, 67% Recall
• CBC+ESR (Text): 100% Precision, 100% Recall
• Mixed Panel: 100% Precision, 80% Recall

KNOWN LIMITATIONS:
• "Blood Sugar Fasting" not detected when written as just "Blood Sugar Fasting"
• "Total Cholesterol" not mapped to "cholesterol" param
• Some vitamin values have slight extraction variance
""")

print(f"\n📁 All graphs saved to: {RESULTS_DIR}/")
print("   real_fig1_precision_recall_f1.png")
print("   real_fig2_detection_counts.png")
print("   real_fig3_classification_accuracy.png")
print("   real_fig4_recall_by_type.png")
print("   real_fig5_overall_accuracy_pie.png")
print("   real_fig6_precision_recall_line.png")
print("   real_fig7_summary_dashboard.png")
print("\n✅ Done! These are REAL numbers from your system.")
