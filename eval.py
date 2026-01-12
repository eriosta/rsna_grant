import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL PLOT STYLING - Grant-appropriate: clear, readable, compact
# ============================================================================
plt.rcParams.update({
    # Font sizes - scaled for ~7 inch wide figures
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    # Font weights
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.titleweight': 'bold',
    # Line widths
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    # Marker edge
    'lines.markeredgewidth': 1,
})

# ============================================================================
# CONFIGURATION - OUTPUT PATHS
# ============================================================================
OUTPUT_DIR = '/Users/eri/rsna_grant/outputs/'

# Excel output files
PATIENT_CONCORDANCE_FILE = OUTPUT_DIR + 'patient_concordance_metrics.xlsx'
POOLED_CONCORDANCE_FILE = OUTPUT_DIR + 'pooled_concordance_summary.xlsx'
LESION_TYPE_CONCORDANCE_FILE = OUTPUT_DIR + 'lesion_type_concordance.xlsx'
PATIENT1_ENHANCED_FILE = OUTPUT_DIR + 'patient1_enhanced.xlsx'
PATIENT5_ENHANCED_FILE = OUTPUT_DIR + 'patient5_enhanced.xlsx'
PATIENT6_ENHANCED_FILE = OUTPUT_DIR + 'patient6_enhanced.xlsx'

# Main concordance figures
CONCORDANCE_FIGURE_SVG = OUTPUT_DIR + 'concordance_analysis.svg'

# Patient trajectory figures
PATIENT1_TRAJ_SVG = OUTPUT_DIR + 'patient1_trajectories.svg'
PATIENT5_TRAJ_SVG = OUTPUT_DIR + 'patient5_trajectories.svg'
PATIENT6_TRAJ_SVG = OUTPUT_DIR + 'patient6_trajectories.svg'

# Grant application figures
WATERFALL_PLOT_SVG = OUTPUT_DIR + 'waterfall_plot.svg'
BLAND_ALTMAN_SVG = OUTPUT_DIR + 'bland_altman_plots.svg'
DETECTION_BY_SIZE_SVG = OUTPUT_DIR + 'detection_by_size.svg'
SPIDER_PLOT_SVG = OUTPUT_DIR + 'spider_plot.svg'
# Individual patient spider plots (use .format(patient_number) to fill in)
SPIDER_PLOT_PATIENT_SVG = OUTPUT_DIR + 'spider_plot_patient{}.svg'
TUMOR_BURDEN_SVG = OUTPUT_DIR + 'total_tumor_burden.svg'
PERCIST_BURDEN_SVG = OUTPUT_DIR + 'percist_metabolic_burden.svg'

# ============================================================================
# LOAD DATA
# ============================================================================

# Load the data
file_path = '/Users/eri/rsna_grant/lesion_trajectories_gt_vs_pred.xlsx'
summary_df = pd.read_excel(file_path, sheet_name='Summary')

# Get all patient sheets
xl = pd.ExcelFile(file_path)
patient_sheets = [sheet for sheet in xl.sheet_names if sheet.startswith('MRN_')]

print("="*80)
print("LESION TRAJECTORY CONCORDANCE ANALYSIS")
print("="*80)
print(f"\nTotal Patients: {len(patient_sheets)}")
print(f"Total Lesion Trajectories: {len(summary_df)}")

# ============================================================================
# CONCORDANCE CORRELATION COEFFICIENT (CCC)
# ============================================================================
def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate Lin's Concordance Correlation Coefficient"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return np.nan, 0
    
    # Pearson correlation coefficient
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    
    # CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    ccc = numerator / denominator if denominator != 0 else 0
    
    return ccc, len(y_true)

# ============================================================================
# INDIVIDUAL PATIENT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("INDIVIDUAL PATIENT METRICS")
print("="*80)

patient_results = []

for patient_id in patient_sheets:
    patient_df = pd.read_excel(file_path, sheet_name=patient_id)
    
    # Remove empty rows
    patient_df = patient_df.dropna(subset=['Date'])
    
    if len(patient_df) == 0:
        continue
    
    # Calculate detection rate (HRA match exists, not "no_hra")
    # A detection is when the HRA_Group doesn't contain "no_hra"
    detected = patient_df['HRA_Group'].notna() & ~patient_df['HRA_Group'].astype(str).str.contains('no_hra', case=False, na=False)
    detection_rate = (detected.sum() / len(patient_df)) * 100
    
    # Calculate metrics
    location_accuracy = patient_df['Location_Match'].mean() * 100
    laterality_accuracy = patient_df['Laterality_Match'].mean() * 100
    
    # Measurement errors (excluding NaN)
    long_axis_error = patient_df['Long_Axis_Error_mm'].mean()
    short_axis_error = patient_df['Short_Axis_Error_mm'].mean()
    suv_error = patient_df['SUV_Error'].mean()
    
    # CCC for continuous measurements
    ccc_long_axis, n_long = concordance_correlation_coefficient(
        patient_df['GT_Long_Axis_mm'], 
        patient_df['Pred_Long_Axis_mm']
    )
    
    ccc_short_axis, n_short = concordance_correlation_coefficient(
        patient_df['GT_Short_Axis_mm'], 
        patient_df['Pred_Short_Axis_mm']
    )
    
    ccc_suv, n_suv = concordance_correlation_coefficient(
        patient_df['GT_SUV'], 
        patient_df['Pred_SUV']
    )
    
    # Unique lesion trajectories for this patient
    n_trajectories = patient_df['HRA_Group'].nunique()
    
    patient_results.append({
        'Patient_ID': patient_id.replace('MRN_', ''),
        'N_Observations': len(patient_df),
        'N_Trajectories': n_trajectories,
        'Detection_Rate_%': detection_rate,
        'Location_Accuracy_%': location_accuracy,
        'Laterality_Accuracy_%': laterality_accuracy,
        'Mean_Long_Axis_Error_mm': long_axis_error,
        'Mean_Short_Axis_Error_mm': short_axis_error,
        'Mean_SUV_Error': suv_error,
        'CCC_Long_Axis': ccc_long_axis,
        'CCC_Short_Axis': ccc_short_axis,
        'CCC_SUV': ccc_suv,
        'N_Long_Axis_Pairs': n_long,
        'N_Short_Axis_Pairs': n_short,
        'N_SUV_Pairs': n_suv
    })
    
    print(f"\n{patient_id}")
    print(f"  Observations: {len(patient_df)} | Trajectories: {n_trajectories}")
    print(f"  Detection Rate: {detection_rate:.1f}%")
    print(f"  Location Accuracy: {location_accuracy:.1f}%")
    print(f"  Laterality Accuracy: {laterality_accuracy:.1f}%")
    print(f"  Long Axis Error: {long_axis_error:.2f} mm (CCC: {ccc_long_axis:.3f}, n={n_long})")
    print(f"  Short Axis Error: {short_axis_error:.2f} mm (CCC: {ccc_short_axis:.3f}, n={n_short})")
    print(f"  SUV Error: {suv_error:.2f} (CCC: {ccc_suv:.3f}, n={n_suv})")

patient_results_df = pd.DataFrame(patient_results)

# ============================================================================
# POOLED ANALYSIS ACROSS ALL PATIENTS
# ============================================================================
print("\n" + "="*80)
print("POOLED METRICS (ALL PATIENTS COMBINED)")
print("="*80)

# Combine all patient data
all_patient_data = []
for patient_id in patient_sheets:
    patient_df = pd.read_excel(file_path, sheet_name=patient_id)
    patient_df = patient_df.dropna(subset=['Date'])
    patient_df['Patient'] = patient_id
    all_patient_data.append(patient_df)

pooled_df = pd.concat(all_patient_data, ignore_index=True)

# Overall accuracy
detected_pooled = pooled_df['HRA_Group'].notna() & ~pooled_df['HRA_Group'].astype(str).str.contains('no_hra', case=False, na=False)
overall_detection_rate = (detected_pooled.sum() / len(pooled_df)) * 100
overall_location_acc = pooled_df['Location_Match'].mean() * 100
overall_laterality_acc = pooled_df['Laterality_Match'].mean() * 100

print(f"\nTotal Observations: {len(pooled_df)}")
print(f"Overall Detection Rate: {overall_detection_rate:.2f}%")
print(f"Overall Location Accuracy: {overall_location_acc:.2f}%")
print(f"Overall Laterality Accuracy: {overall_laterality_acc:.2f}%")

# Pooled CCC for measurements
pooled_ccc_long, n_long_pooled = concordance_correlation_coefficient(
    pooled_df['GT_Long_Axis_mm'], 
    pooled_df['Pred_Long_Axis_mm']
)

pooled_ccc_short, n_short_pooled = concordance_correlation_coefficient(
    pooled_df['GT_Short_Axis_mm'], 
    pooled_df['Pred_Short_Axis_mm']
)

pooled_ccc_suv, n_suv_pooled = concordance_correlation_coefficient(
    pooled_df['GT_SUV'], 
    pooled_df['Pred_SUV']
)

print(f"\nPooled Concordance Correlation Coefficients:")
print(f"  Long Axis CCC: {pooled_ccc_long:.3f} (n={n_long_pooled})")
print(f"  Short Axis CCC: {pooled_ccc_short:.3f} (n={n_short_pooled})")
print(f"  SUV CCC: {pooled_ccc_suv:.3f} (n={n_suv_pooled})")

# Pooled measurement errors
pooled_long_error = pooled_df['Long_Axis_Error_mm'].mean()
pooled_short_error = pooled_df['Short_Axis_Error_mm'].mean()
pooled_suv_error = pooled_df['SUV_Error'].mean()

pooled_long_std = pooled_df['Long_Axis_Error_mm'].std()
pooled_short_std = pooled_df['Short_Axis_Error_mm'].std()
pooled_suv_std = pooled_df['SUV_Error'].std()

print(f"\nPooled Measurement Errors (Mean ± SD):")
print(f"  Long Axis: {pooled_long_error:.2f} ± {pooled_long_std:.2f} mm")
print(f"  Short Axis: {pooled_short_error:.2f} ± {pooled_short_std:.2f} mm")
print(f"  SUV: {pooled_suv_error:.2f} ± {pooled_suv_std:.2f}")

# Cohen's Kappa for location matching (categorical)
# Create binary vectors for perfect location match vs not
location_gt = pooled_df['GT_Location'].astype(str)
location_pred = pooled_df['HRA_Group'].astype(str)

# For locations that match exactly
exact_matches = (pooled_df['Location_Match'] == 1).astype(int)
kappa_location = cohen_kappa_score(
    [1]*len(exact_matches), 
    exact_matches
) if len(exact_matches) > 0 else 0

print(f"\nCohen's Kappa for Location Agreement: {kappa_location:.3f}")

# ============================================================================
# SUMMARY BY LESION TYPE
# ============================================================================
print("\n" + "="*80)
print("METRICS BY LESION TYPE (FROM SUMMARY)")
print("="*80)

# Calculate detection rate for each row in summary
summary_df['Detected'] = summary_df['HRA_Group'].notna() & ~summary_df['HRA_Group'].astype(str).str.contains('no_hra', case=False, na=False)

lesion_type_metrics = summary_df.groupby('GT_Lesion_Type').agg({
    'Num_Observations': 'sum',
    'Location_Match_Count': 'sum',
    'Laterality_Match_Count': 'sum',
    'Avg_Long_Axis_Error_mm': 'mean',
    'Avg_Short_Axis_Error_mm': 'mean',
    'Avg_SUV_Error': 'mean',
    'Detected': 'sum'
}).reset_index()

# Calculate accuracies
lesion_type_metrics['Detection_Rate_%'] = (
    lesion_type_metrics['Detected'] / 
    lesion_type_metrics['Num_Observations'] * 100
)

lesion_type_metrics['Location_Accuracy_%'] = (
    lesion_type_metrics['Location_Match_Count'] / 
    lesion_type_metrics['Num_Observations'] * 100
)

lesion_type_metrics['Laterality_Accuracy_%'] = (
    lesion_type_metrics['Laterality_Match_Count'] / 
    lesion_type_metrics['Num_Observations'] * 100
)

print(lesion_type_metrics.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Save individual patient results
patient_results_df.to_excel(PATIENT_CONCORDANCE_FILE, index=False)

# Create comprehensive summary
summary_stats = {
    'Metric': [
        'Total Patients',
        'Total Observations',
        'Total Trajectories',
        'Overall Detection Rate (%)',
        'Overall Location Accuracy (%)',
        'Overall Laterality Accuracy (%)',
        'Pooled Long Axis CCC',
        'Pooled Short Axis CCC',
        'Pooled SUV CCC',
        'Mean Long Axis Error (mm)',
        'Mean Short Axis Error (mm)',
        'Mean SUV Error',
        'Cohen\'s Kappa (Location)'
    ],
    'Value': [
        len(patient_sheets),
        len(pooled_df),
        summary_df['HRA_Group'].nunique(),
        f"{overall_detection_rate:.2f}",
        f"{overall_location_acc:.2f}",
        f"{overall_laterality_acc:.2f}",
        f"{pooled_ccc_long:.3f}",
        f"{pooled_ccc_short:.3f}",
        f"{pooled_ccc_suv:.3f}",
        f"{pooled_long_error:.2f} ± {pooled_long_std:.2f}",
        f"{pooled_short_error:.2f} ± {pooled_short_std:.2f}",
        f"{pooled_suv_error:.2f} ± {pooled_suv_std:.2f}",
        f"{kappa_location:.3f}"
    ]
}

summary_stats_df = pd.DataFrame(summary_stats)
summary_stats_df.to_excel(POOLED_CONCORDANCE_FILE, index=False)

# Save lesion type metrics
lesion_type_metrics.to_excel(LESION_TYPE_CONCORDANCE_FILE, index=False)

print("\n" + "="*80)
print("Results saved to:")
print("  Excel files:")
print("    - patient_concordance_metrics.xlsx")
print("    - pooled_concordance_summary.xlsx")
print("    - lesion_type_concordance.xlsx")
print("  Figures:")
print("    - concordance_analysis.svg")
print("    - patient1_trajectories.svg")
print("    - patient5_trajectories.svg")
print("    - patient6_trajectories.svg")
print("="*80)

# ============================================================================
# CREATE VISUALIZATIONS - MAIN CONCORDANCE FIGURE
# ============================================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

# 1. Detection and Location Accuracy by Patient
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(patient_results_df))
width = 0.35
bars1 = ax1.bar(x - width/2, patient_results_df['Detection_Rate_%'], 
                width, label='Detection Rate', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, patient_results_df['Location_Accuracy_%'], 
                width, label='Location Accuracy', color='#F18F01', alpha=0.8)
ax1.set_xlabel('Patient', fontweight='bold', fontsize=9)
ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=9)
ax1.set_title('A. Detection and Location Accuracy by Patient', fontweight='bold', fontsize=10, loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels([f"P{i+1}" for i in range(len(patient_results_df))], rotation=0)
ax1.legend(loc='lower right', fontsize=8, frameon=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, 115])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if not np.isnan(height):
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    height = bar.get_height()
    if not np.isnan(height):
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=7)

# 2. CCC Distribution with individual points
ax2 = fig.add_subplot(gs[0, 2])
ccc_data = []
ccc_labels = []
ccc_colors = ['#F18F01', '#C73E1D', '#6A994E']

for metric in ['CCC_Long_Axis', 'CCC_Short_Axis', 'CCC_SUV']:
    vals = patient_results_df[metric].dropna()
    if len(vals) > 0:
        ccc_data.append(vals)
        ccc_labels.append(metric.replace('CCC_', '').replace('_', ' '))

bp = ax2.boxplot(ccc_data, labels=ccc_labels, patch_artist=True,
                 showmeans=True, meanline=True, widths=0.5)
for patch, color in zip(bp['boxes'], ccc_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)

# Overlay individual datapoints
for i, (data, color) in enumerate(zip(ccc_data, ccc_colors)):
    # Add jitter to x-coordinates for better visibility
    x_jitter = np.random.normal(i+1, 0.04, size=len(data))
    ax2.scatter(x_jitter, data, alpha=0.7, s=60, color=color, 
               edgecolors='black', linewidths=0.7, zorder=3)

ax2.set_ylabel('CCC', fontweight='bold', fontsize=9)
ax2.set_title('B. CCC Distribution (with Individual Points)', fontweight='bold', fontsize=10, loc='left')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 1.05])
ax2.tick_params(axis='x', rotation=15, labelsize=11)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Bland-Altman Plot - Long Axis
ax3 = fig.add_subplot(gs[1, 0])
gt_long = pooled_df['GT_Long_Axis_mm'].dropna()
pred_long = pooled_df['Pred_Long_Axis_mm'].dropna()
mask = ~(pooled_df['GT_Long_Axis_mm'].isna() | pooled_df['Pred_Long_Axis_mm'].isna())
gt_long = pooled_df.loc[mask, 'GT_Long_Axis_mm']
pred_long = pooled_df.loc[mask, 'Pred_Long_Axis_mm']

if len(gt_long) > 0:
    mean_vals = (gt_long + pred_long) / 2
    diff_vals = pred_long - gt_long
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()
    
    ax3.scatter(mean_vals, diff_vals, alpha=0.5, s=40, color='#2E86AB')
    ax3.axhline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.2f}')
    ax3.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5, 
                label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    ax3.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5,
                label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    ax3.set_xlabel('Mean of GT and Pred (mm)', fontweight='bold', fontsize=8)
    ax3.set_ylabel('Difference (Pred - GT) mm', fontweight='bold', fontsize=8)
    ax3.set_title('C. Bland-Altman: Long Axis', fontweight='bold', fontsize=10, loc='left')
    ax3.legend(fontsize=6, frameon=True, fancybox=True, loc='upper left')
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

# 4. Bland-Altman Plot - Short Axis
ax4 = fig.add_subplot(gs[1, 1])
mask = ~(pooled_df['GT_Short_Axis_mm'].isna() | pooled_df['Pred_Short_Axis_mm'].isna())
gt_short = pooled_df.loc[mask, 'GT_Short_Axis_mm']
pred_short = pooled_df.loc[mask, 'Pred_Short_Axis_mm']

if len(gt_short) > 0:
    mean_vals = (gt_short + pred_short) / 2
    diff_vals = pred_short - gt_short
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()
    
    ax4.scatter(mean_vals, diff_vals, alpha=0.5, s=40, color='#A23B72')
    ax4.axhline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.2f}')
    ax4.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5,
                label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    ax4.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5,
                label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    ax4.set_xlabel('Mean of GT and Pred (mm)', fontweight='bold', fontsize=8)
    ax4.set_ylabel('Difference (Pred - GT) mm', fontweight='bold', fontsize=8)
    ax4.set_title('D. Bland-Altman: Short Axis', fontweight='bold', fontsize=10, loc='left')
    ax4.legend(fontsize=6, frameon=True, fancybox=True, loc='upper left')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

# 5. Bland-Altman Plot - SUV
ax5 = fig.add_subplot(gs[1, 2])
mask = ~(pooled_df['GT_SUV'].isna() | pooled_df['Pred_SUV'].isna())
gt_suv = pooled_df.loc[mask, 'GT_SUV']
pred_suv = pooled_df.loc[mask, 'Pred_SUV']

if len(gt_suv) > 0:
    mean_vals = (gt_suv + pred_suv) / 2
    diff_vals = pred_suv - gt_suv
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()
    
    ax5.scatter(mean_vals, diff_vals, alpha=0.5, s=40, color='#F18F01')
    ax5.axhline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.2f}')
    ax5.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5,
                label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    ax5.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle=':', linewidth=1.5,
                label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    ax5.set_xlabel('Mean of GT and Pred', fontweight='bold', fontsize=8)
    ax5.set_ylabel('Difference (Pred - GT)', fontweight='bold', fontsize=8)
    ax5.set_title('E. Bland-Altman: SUV Max', fontweight='bold', fontsize=10, loc='left')
    ax5.legend(fontsize=6, frameon=True, fancybox=True, loc='lower left')
    ax5.grid(alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

fig.suptitle('Lesion Trajectory Concordance Analysis', fontsize=12, fontweight='bold', y=1.0)

plt.savefig(CONCORDANCE_FIGURE_SVG, bbox_inches='tight', facecolor='white')
plt.close()

print("Main concordance figure saved!")

# ============================================================================
# CREATE SEPARATE TRAJECTORY FIGURES FOR P5 AND P6
# ============================================================================
print("Generating individual patient trajectory figures...")

# Load P5 data
p5_id = patient_sheets[4]  # 5th patient (0-indexed)
p5_df = pd.read_excel(file_path, sheet_name=p5_id)
p5_df = p5_df.dropna(subset=['Date'])
p5_df['Date'] = pd.to_datetime(p5_df['Date'])
# Map dates to sequential timepoints for P5
p5_unique_dates = sorted(p5_df['Date'].unique())
p5_date_to_timepoint = {date: idx + 1 for idx, date in enumerate(p5_unique_dates)}
p5_df['Timepoint'] = p5_df['Date'].map(p5_date_to_timepoint)
p5_metrics = patient_results_df[patient_results_df['Patient_ID'] == p5_id.replace('MRN_', '')].iloc[0]

# Load P6 data
p6_id = patient_sheets[5]  # 6th patient (0-indexed)
p6_df = pd.read_excel(file_path, sheet_name=p6_id)
p6_df = p6_df.dropna(subset=['Date'])
p6_df['Date'] = pd.to_datetime(p6_df['Date'])
# Map dates to sequential timepoints for P6
p6_unique_dates = sorted(p6_df['Date'].unique())
p6_date_to_timepoint = {date: idx + 1 for idx, date in enumerate(p6_unique_dates)}
p6_df['Timepoint'] = p6_df['Date'].map(p6_date_to_timepoint)
p6_metrics = patient_results_df[patient_results_df['Patient_ID'] == p6_id.replace('MRN_', '')].iloc[0]

# ============================================================================
# LESION TRACKING AND AREA CALCULATION
# ============================================================================

def create_unique_lesion_ids(df):
    """
    Create unique lesion IDs within each HRA_Group by tracking lesions across time.
    Uses size-based ranking to identify individual lesions within the same anatomical location.
    
    For example, "right lung_3" might have 4 separate lesions:
    - L0 (largest), L1 (2nd largest), L2 (3rd largest), L3 (smallest)
    """
    df = df.copy()
    df['Lesion_ID'] = ''
    
    for hra_group in df['HRA_Group'].unique():
        if pd.isna(hra_group):
            continue
            
        mask = df['HRA_Group'] == hra_group
        group_df = df[mask].copy().sort_values('Date')
        
        # Calculate area for matching
        group_df['GT_Area_temp'] = group_df['GT_Long_Axis_mm'] * group_df['GT_Short_Axis_mm']
        
        # For each unique date, assign lesion indices based on size ranking
        for date in sorted(group_df['Date'].unique()):
            date_mask = group_df['Date'] == date
            date_obs = group_df[date_mask].copy()
            
            # Sort by area (largest first) - this ranks lesions by size
            date_obs = date_obs.sort_values('GT_Area_temp', ascending=False, na_position='last')
            
            # Assign lesion numbers based on size ranking (0=largest, 1=2nd largest, etc.)
            for rank, (idx, row) in enumerate(date_obs.iterrows()):
                lesion_id = f"{hra_group}_L{rank}"
                df.loc[idx, 'Lesion_ID'] = lesion_id
    
    return df

print("\n" + "="*80)
print("CREATING UNIQUE LESION IDENTIFIERS AND CALCULATING AREAS")
print("="*80)

# Create unique lesion IDs for both patients
p5_df = create_unique_lesion_ids(p5_df)
p6_df = create_unique_lesion_ids(p6_df)

# Calculate area measurements (bidimensional product - WHO/RECIST standard)
p5_df['GT_Area_mm2'] = p5_df['GT_Long_Axis_mm'] * p5_df['GT_Short_Axis_mm']
p5_df['Pred_Area_mm2'] = p5_df['Pred_Long_Axis_mm'] * p5_df['Pred_Short_Axis_mm']

p6_df['GT_Area_mm2'] = p6_df['GT_Long_Axis_mm'] * p6_df['GT_Short_Axis_mm']
p6_df['Pred_Area_mm2'] = p6_df['Pred_Long_Axis_mm'] * p6_df['Pred_Short_Axis_mm']

# Print summary of lesion tracking
print(f"\nPatient 5 ({p5_id}):")
print(f"  Total observations: {len(p5_df)}")
print(f"  Unique lesions identified: {p5_df['Lesion_ID'].nunique()}")
print(f"  HRA groups with multiple lesions:")
lesion_counts_p5 = p5_df.groupby('HRA_Group')['Lesion_ID'].nunique()
multi_lesion_p5 = lesion_counts_p5[lesion_counts_p5 > 1]
for hra, count in multi_lesion_p5.items():
    print(f"    - {hra}: {count} lesions")

print(f"\nPatient 6 ({p6_id}):")
print(f"  Total observations: {len(p6_df)}")
print(f"  Unique lesions identified: {p6_df['Lesion_ID'].nunique()}")
print(f"  HRA groups with multiple lesions:")
lesion_counts_p6 = p6_df.groupby('HRA_Group')['Lesion_ID'].nunique()
multi_lesion_p6 = lesion_counts_p6[lesion_counts_p6 > 1]
for hra, count in multi_lesion_p6.items():
    print(f"    - {hra}: {count} lesions")

# Save enhanced data with lesion IDs and area calculations
p5_df.to_excel(PATIENT5_ENHANCED_FILE, index=False)
p6_df.to_excel(PATIENT6_ENHANCED_FILE, index=False)
print(f"\nEnhanced data saved:")
print(f"  - patient5_enhanced.xlsx (with Lesion_ID and Area_mm2 columns)")
print(f"  - patient6_enhanced.xlsx (with Lesion_ID and Area_mm2 columns)")

# ============================================================================
# ============================================================================
# ============================================================================
# PATIENT 5 TRAJECTORY FIGURE - 2-COLUMN GRID LAYOUT WITH AREA MEASUREMENTS
# ============================================================================

# Get unique HRA groups (anatomical locations)
p5_hra_groups = []
for hra in p5_df['HRA_Group'].unique():
    if pd.notna(hra):
        hra_data = p5_df[p5_df['HRA_Group'] == hra].sort_values('Date')
        if len(hra_data) > 0:
            p5_hra_groups.append((hra, hra_data))

n_hra_p5 = len(p5_hra_groups)

# Calculate grid dimensions (2 columns)
n_cols = 2
n_rows = (n_hra_p5 + n_cols - 1) // n_cols

# Create figure with grid layout
fig_p5 = plt.figure(figsize=(10, 3.5 * n_rows))
gs_p5 = fig_p5.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)

colors_hra = ['#2E86AB', '#F18F01', '#A23B72', '#6A994E', '#C73E1D']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers per lesion
linestyles = ['-', '--', '-.', ':']  # Different line styles per lesion

# Get overall timepoint range for P5 to align all panels
p5_tp_min = p5_df['Timepoint'].min()
p5_tp_max = p5_df['Timepoint'].max()
p5_xlim = [p5_tp_min - 0.3, p5_tp_max + 0.3]

# Error thresholds
area_error_threshold = 10.0  # mm²
suv_error_threshold = 0.5

for hra_idx, (hra_name, hra_data) in enumerate(p5_hra_groups):
    row = hra_idx // n_cols
    col = hra_idx % n_cols
    
    ax = fig_p5.add_subplot(gs_p5[row, col])
    color = colors_hra[hra_idx % len(colors_hra)]
    ax_suv = ax.twinx()
    
    # Get unique lesions within this HRA group
    lesions = hra_data['Lesion_ID'].unique()
    n_lesions = len(lesions)
    
    # ============ AREA MEASUREMENTS (Left Y-axis) ============
    for lesion_idx, lesion_id in enumerate(sorted(lesions)):
        lesion_data = hra_data[hra_data['Lesion_ID'] == lesion_id].sort_values('Date')
        
        marker = markers[lesion_idx % len(markers)]
        linestyle = linestyles[lesion_idx % len(linestyles)]
        
        # Plot area predictions
        pred_area_mask = lesion_data['Pred_Area_mm2'].notna()
        if pred_area_mask.sum() > 0:
            dates_pred = lesion_data.loc[pred_area_mask, 'Timepoint']
            area_pred = lesion_data.loc[pred_area_mask, 'Pred_Area_mm2']
            
            lesion_label = lesion_id.split('_L')[1]  # Extract "0", "1", "2", etc.
            ax.plot(dates_pred, area_pred, marker=marker, linestyle=linestyle,
                   linewidth=1.5, markersize=7, color=color, alpha=0.5,
                   markeredgecolor='black', markeredgewidth=1.2, zorder=5,
                   label=f'Lesion {lesion_label}')
        
        # Add area error arrows
        for date_idx in lesion_data.index:
            if pd.notna(lesion_data.loc[date_idx, 'GT_Area_mm2']) and pd.notna(lesion_data.loc[date_idx, 'Pred_Area_mm2']):
                date = lesion_data.loc[date_idx, 'Timepoint']
                gt_area = lesion_data.loc[date_idx, 'GT_Area_mm2']
                pred_area = lesion_data.loc[date_idx, 'Pred_Area_mm2']
                if abs(gt_area - pred_area) > area_error_threshold:
                    ax.annotate('', xy=(date, gt_area), xytext=(date, pred_area),
                              arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                              zorder=10)
                    ax.scatter(date, gt_area, color='red', s=100, marker='x', linewidths=3, zorder=11)
    
    # ============ SUV MEASUREMENTS (Right Y-axis) ============
    # Track if SUV data is plotted
    suv_data_plotted = False
    # Plot SUV per lesion
    for lesion_idx, lesion_id in enumerate(sorted(lesions)):
        lesion_data = hra_data[hra_data['Lesion_ID'] == lesion_id].sort_values('Date')
        
        pred_suv_mask = lesion_data['Pred_SUV'].notna()
        if pred_suv_mask.sum() > 0:
            dates_suv = lesion_data.loc[pred_suv_mask, 'Timepoint']
            suv_pred = lesion_data.loc[pred_suv_mask, 'Pred_SUV']
            ax_suv.plot(dates_suv, suv_pred, marker='D', linestyle='--', linewidth=1.5,
                       markersize=6, color='darkviolet', alpha=0.5,
                       markeredgecolor='black', markeredgewidth=1.8, zorder=4)
            suv_data_plotted = True
        
        # Add SUV error arrows
        for date_idx in lesion_data.index:
            if pd.notna(lesion_data.loc[date_idx, 'GT_SUV']) and pd.notna(lesion_data.loc[date_idx, 'Pred_SUV']):
                date = lesion_data.loc[date_idx, 'Timepoint']
                gt_suv = lesion_data.loc[date_idx, 'GT_SUV']
                pred_suv = lesion_data.loc[date_idx, 'Pred_SUV']
                if abs(gt_suv - pred_suv) > suv_error_threshold:
                    ax_suv.annotate('', xy=(date, gt_suv), xytext=(date, pred_suv),
                                  arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                                  zorder=10)
                    ax_suv.scatter(date, gt_suv, color='red', s=100, marker='x', linewidths=3, zorder=11)
    
    # Formatting
    ax.set_ylabel('Area (mm²)', fontweight='bold', fontsize=10, color=color)
    ax_suv.set_ylabel('SUV Max', fontweight='bold', fontsize=10, color='darkviolet')
    ax.set_title(f'{chr(65+hra_idx)}. {hra_name} ({n_lesions} lesion{"s" if n_lesions > 1 else ""}, n={len(hra_data)} obs)',
                fontweight='bold', fontsize=10, loc='left')
    
    # Combined legend - optimized placement
    lines1, labels1 = ax.get_legend_handles_labels()
    
    # If only SUV data is plotted (no area data), create legend with just SUV Max
    if not lines1 and suv_data_plotted:
        lines1 = [plt.Line2D([0], [0], marker='D', color='darkviolet', linestyle='--',
                            markersize=10, markeredgecolor='black', alpha=0.5)]
        labels1 = ['SUV Max']
    
    # If area data exists, optionally add SUV Max if it was plotted (only if not already present)
    if lines1:
        if suv_data_plotted and 'SUV Max' not in labels1:
            lines1.append(plt.Line2D([0], [0], marker='D', color='darkviolet', linestyle='--',
                                    markersize=10, markeredgecolor='black', alpha=0.5))
            labels1.append('SUV Max')
        # Use 'best' location which automatically finds the least cluttered position
        if lines1:  # Only create legend if there are items to show
            ax.legend(lines1, labels1, fontsize=13, loc='best', frameon=True, 
                     fancybox=True, shadow=True, ncol=min(2, len(lines1)))
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14, labelcolor=color)
    ax_suv.tick_params(axis='y', labelsize=14, labelcolor='darkviolet')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(color)
    ax_suv.spines['right'].set_color('darkviolet')
    ax.set_xlim(p5_xlim)
    
    # Only show x-label on bottom row
    if row == n_rows - 1:
        ax.set_xlabel('Timepoint', fontweight='bold', fontsize=10)
    else:
        ax.set_xlabel('')

fig_p5.suptitle(f'Patient 5 Lesion Trajectories (n={len(p5_df)} observations, {p5_df["Lesion_ID"].nunique()} unique lesions)',
               fontsize=10, fontweight='bold', y=0.98)

plt.savefig(PATIENT5_TRAJ_SVG, bbox_inches='tight', facecolor='white')
plt.close()
# ============================================================================
# PATIENT 6 TRAJECTORY FIGURE - 2-COLUMN GRID LAYOUT WITH AREA MEASUREMENTS
# ============================================================================

# Get unique HRA groups (anatomical locations)
p6_hra_groups = []
for hra in p6_df['HRA_Group'].unique():
    if pd.notna(hra):
        hra_data = p6_df[p6_df['HRA_Group'] == hra].sort_values('Date')
        if len(hra_data) > 0:
            p6_hra_groups.append((hra, hra_data))

n_hra_p6 = len(p6_hra_groups)
n_rows = (n_hra_p6 + n_cols - 1) // n_cols

# Create figure with grid layout
fig_p6 = plt.figure(figsize=(10, 3.5 * n_rows))
gs_p6 = fig_p6.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)

# Get overall timepoint range for P6
p6_tp_min = p6_df['Timepoint'].min()
p6_tp_max = p6_df['Timepoint'].max()
p6_xlim = [p6_tp_min - 0.3, p6_tp_max + 0.3]

for hra_idx, (hra_name, hra_data) in enumerate(p6_hra_groups):
    row = hra_idx // n_cols
    col = hra_idx % n_cols
    
    ax = fig_p6.add_subplot(gs_p6[row, col])
    color = colors_hra[hra_idx % len(colors_hra)]
    ax_suv = ax.twinx()
    
    # Get unique lesions within this HRA group
    lesions = hra_data['Lesion_ID'].unique()
    n_lesions = len(lesions)
    
    # ============ AREA MEASUREMENTS (Left Y-axis) ============
    for lesion_idx, lesion_id in enumerate(sorted(lesions)):
        lesion_data = hra_data[hra_data['Lesion_ID'] == lesion_id].sort_values('Date')
        
        marker = markers[lesion_idx % len(markers)]
        linestyle = linestyles[lesion_idx % len(linestyles)]
        
        # Plot area predictions
        pred_area_mask = lesion_data['Pred_Area_mm2'].notna()
        if pred_area_mask.sum() > 0:
            dates_pred = lesion_data.loc[pred_area_mask, 'Timepoint']
            area_pred = lesion_data.loc[pred_area_mask, 'Pred_Area_mm2']
            
            lesion_label = lesion_id.split('_L')[1]
            ax.plot(dates_pred, area_pred, marker=marker, linestyle=linestyle,
                   linewidth=1.5, markersize=7, color=color, alpha=0.5,
                   markeredgecolor='black', markeredgewidth=1.2, zorder=5,
                   label=f'Lesion {lesion_label}')
        
        # Add area error arrows
        for date_idx in lesion_data.index:
            if pd.notna(lesion_data.loc[date_idx, 'GT_Area_mm2']) and pd.notna(lesion_data.loc[date_idx, 'Pred_Area_mm2']):
                date = lesion_data.loc[date_idx, 'Timepoint']
                gt_area = lesion_data.loc[date_idx, 'GT_Area_mm2']
                pred_area = lesion_data.loc[date_idx, 'Pred_Area_mm2']
                if abs(gt_area - pred_area) > area_error_threshold:
                    ax.annotate('', xy=(date, gt_area), xytext=(date, pred_area),
                              arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                              zorder=10)
                    ax.scatter(date, gt_area, color='red', s=100, marker='x', linewidths=3, zorder=11)
    
    # ============ SUV MEASUREMENTS (Right Y-axis) ============
    # Track if SUV data is plotted
    suv_data_plotted = False
    for lesion_idx, lesion_id in enumerate(sorted(lesions)):
        lesion_data = hra_data[hra_data['Lesion_ID'] == lesion_id].sort_values('Date')
        
        pred_suv_mask = lesion_data['Pred_SUV'].notna()
        if pred_suv_mask.sum() > 0:
            dates_suv = lesion_data.loc[pred_suv_mask, 'Timepoint']
            suv_pred = lesion_data.loc[pred_suv_mask, 'Pred_SUV']
            ax_suv.plot(dates_suv, suv_pred, marker='D', linestyle='--', linewidth=1.5,
                       markersize=6, color='darkviolet', alpha=0.5,
                       markeredgecolor='black', markeredgewidth=1.8, zorder=4)
            suv_data_plotted = True
        
        # Add SUV error arrows
        for date_idx in lesion_data.index:
            if pd.notna(lesion_data.loc[date_idx, 'GT_SUV']) and pd.notna(lesion_data.loc[date_idx, 'Pred_SUV']):
                date = lesion_data.loc[date_idx, 'Timepoint']
                gt_suv = lesion_data.loc[date_idx, 'GT_SUV']
                pred_suv = lesion_data.loc[date_idx, 'Pred_SUV']
                if abs(gt_suv - pred_suv) > suv_error_threshold:
                    ax_suv.annotate('', xy=(date, gt_suv), xytext=(date, pred_suv),
                                  arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                                  zorder=10)
                    ax_suv.scatter(date, gt_suv, color='red', s=100, marker='x', linewidths=3, zorder=11)
    
    # Formatting
    ax.set_ylabel('Area (mm²)', fontweight='bold', fontsize=10, color=color)
    ax_suv.set_ylabel('SUV Max', fontweight='bold', fontsize=10, color='darkviolet')
    ax.set_title(f'{chr(65+hra_idx)}. {hra_name} ({n_lesions} lesion{"s" if n_lesions > 1 else ""}, n={len(hra_data)} obs)',
                fontweight='bold', fontsize=10, loc='left')
    
    # Combined legend - optimized placement
    lines1, labels1 = ax.get_legend_handles_labels()
    
    # If only SUV data is plotted (no area data), create legend with just SUV Max
    if not lines1 and suv_data_plotted:
        lines1 = [plt.Line2D([0], [0], marker='D', color='darkviolet', linestyle='--',
                            markersize=10, markeredgecolor='black', alpha=0.5)]
        labels1 = ['SUV Max']
    
    # If area data exists, optionally add SUV Max if it was plotted (only if not already present)
    if lines1:
        if suv_data_plotted and 'SUV Max' not in labels1:
            lines1.append(plt.Line2D([0], [0], marker='D', color='darkviolet', linestyle='--',
                                    markersize=10, markeredgecolor='black', alpha=0.5))
            labels1.append('SUV Max')
        # Use 'best' location which automatically finds the least cluttered position
        if lines1:  # Only create legend if there are items to show
            ax.legend(lines1, labels1, fontsize=13, loc='best', frameon=True, 
                     fancybox=True, shadow=True, ncol=min(2, len(lines1)))
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14, labelcolor=color)
    ax_suv.tick_params(axis='y', labelsize=14, labelcolor='darkviolet')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(color)
    ax_suv.spines['right'].set_color('darkviolet')
    ax.set_xlim(p6_xlim)
    
    # Only show x-label on bottom row
    if row == n_rows - 1:
        ax.set_xlabel('Timepoint', fontweight='bold', fontsize=10)
    else:
        ax.set_xlabel('')

fig_p6.suptitle(f'Patient 6 Lesion Trajectories (n={len(p6_df)} observations, {p6_df["Lesion_ID"].nunique()} unique lesions)',
               fontsize=10, fontweight='bold', y=0.98)

plt.savefig(PATIENT6_TRAJ_SVG, bbox_inches='tight', facecolor='white')
plt.close()

print("Patient trajectory figures saved!")

# ============================================================================
# PATIENT 1 TRAJECTORY FIGURE - 2-COLUMN GRID LAYOUT WITH AREA MEASUREMENTS
# ============================================================================
print("Generating patient 1 trajectory figure...")

# Find patient 1 by MRN
p1_mrn = 'MRN_3a17da316461461a0856'
p1_id = None
for sheet in patient_sheets:
    if sheet == p1_mrn:
        p1_id = sheet
        break

if p1_id is None:
    print(f"Warning: Patient {p1_mrn} not found in patient sheets. Available sheets: {patient_sheets[:5]}...")
else:
    # Load P1 data
    p1_df = pd.read_excel(file_path, sheet_name=p1_id)
    p1_df = p1_df.dropna(subset=['Date'])
    p1_df['Date'] = pd.to_datetime(p1_df['Date'])
    # Map dates to sequential timepoints for P1
    p1_unique_dates = sorted(p1_df['Date'].unique())
    p1_date_to_timepoint = {date: idx + 1 for idx, date in enumerate(p1_unique_dates)}
    p1_df['Timepoint'] = p1_df['Date'].map(p1_date_to_timepoint)
    p1_metrics = patient_results_df[patient_results_df['Patient_ID'] == p1_id.replace('MRN_', '')].iloc[0] if len(patient_results_df[patient_results_df['Patient_ID'] == p1_id.replace('MRN_', '')]) > 0 else None

    # Create unique lesion IDs for P1
    p1_df = create_unique_lesion_ids(p1_df)

    # Calculate area measurements
    p1_df['GT_Area_mm2'] = p1_df['GT_Long_Axis_mm'] * p1_df['GT_Short_Axis_mm']
    p1_df['Pred_Area_mm2'] = p1_df['Pred_Long_Axis_mm'] * p1_df['Pred_Short_Axis_mm']

    # Print summary of lesion tracking
    print(f"\nPatient 1 ({p1_id}):")
    print(f"  Total observations: {len(p1_df)}")
    print(f"  Unique lesions identified: {p1_df['Lesion_ID'].nunique()}")
    print(f"  HRA groups with multiple lesions:")
    lesion_counts_p1 = p1_df.groupby('HRA_Group')['Lesion_ID'].nunique()
    multi_lesion_p1 = lesion_counts_p1[lesion_counts_p1 > 1]
    for hra, count in multi_lesion_p1.items():
        print(f"    - {hra}: {count} lesions")

    # Save enhanced data with lesion IDs and area calculations
    p1_df.to_excel(PATIENT1_ENHANCED_FILE, index=False)
    print(f"\nEnhanced data saved:")
    print(f"  - patient1_enhanced.xlsx (with Lesion_ID and Area_mm2 columns)")

    # Get unique HRA groups (anatomical locations)
    p1_hra_groups = []
    for hra in p1_df['HRA_Group'].unique():
        if pd.notna(hra):
            hra_data = p1_df[p1_df['HRA_Group'] == hra].sort_values('Date')
            if len(hra_data) > 0:
                p1_hra_groups.append((hra, hra_data))

    n_hra_p1 = len(p1_hra_groups)

    # Calculate grid dimensions (2 columns)
    n_cols = 2
    n_rows = (n_hra_p1 + n_cols - 1) // n_cols

    # Create figure with grid layout
    fig_p1 = plt.figure(figsize=(10, 3.5 * n_rows))
    gs_p1 = fig_p1.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.35)

    colors_hra = ['#2E86AB', '#F18F01', '#A23B72', '#6A994E', '#C73E1D']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']  # Different markers per lesion
    linestyles = ['-', '--', '-.', ':']  # Different line styles per lesion

    # Get overall timepoint range for P1 to align all panels
    p1_tp_min = p1_df['Timepoint'].min()
    p1_tp_max = p1_df['Timepoint'].max()
    p1_xlim = [p1_tp_min - 0.3, p1_tp_max + 0.3]

    # Error thresholds
    area_error_threshold = 10.0  # mm²
    suv_error_threshold = 0.5

    for hra_idx, (hra_name, hra_data) in enumerate(p1_hra_groups):
        row = hra_idx // n_cols
        col = hra_idx % n_cols
        
        ax = fig_p1.add_subplot(gs_p1[row, col])
        color = colors_hra[hra_idx % len(colors_hra)]
        ax_suv = ax.twinx()
        
        # Get unique lesions within this HRA group
        lesions = hra_data['Lesion_ID'].unique()
        n_lesions = len(lesions)
        
        # ============ AREA MEASUREMENTS (Left Y-axis) ============
        for lesion_idx, lesion_id in enumerate(sorted(lesions)):
            lesion_data = hra_data[hra_data['Lesion_ID'] == lesion_id].sort_values('Date')
            
            marker = markers[lesion_idx % len(markers)]
            linestyle = linestyles[lesion_idx % len(linestyles)]
            
            # Plot area predictions
            pred_area_mask = lesion_data['Pred_Area_mm2'].notna()
            if pred_area_mask.sum() > 0:
                dates_pred = lesion_data.loc[pred_area_mask, 'Timepoint']
                area_pred = lesion_data.loc[pred_area_mask, 'Pred_Area_mm2']
                
                lesion_label = lesion_id.split('_L')[1]  # Extract "0", "1", "2", etc.
                ax.plot(dates_pred, area_pred, marker=marker, linestyle=linestyle,
                       linewidth=1.5, markersize=6, color=color, alpha=0.5,
                       markeredgecolor='black', markeredgewidth=1, zorder=5,
                       label=f'Lesion {lesion_label}')
            
            # Add area error arrows
            for date_idx in lesion_data.index:
                if pd.notna(lesion_data.loc[date_idx, 'GT_Area_mm2']) and pd.notna(lesion_data.loc[date_idx, 'Pred_Area_mm2']):
                    date = lesion_data.loc[date_idx, 'Timepoint']
                    gt_area = lesion_data.loc[date_idx, 'GT_Area_mm2']
                    pred_area = lesion_data.loc[date_idx, 'Pred_Area_mm2']
                    if abs(gt_area - pred_area) > area_error_threshold:
                        ax.annotate('', xy=(date, gt_area), xytext=(date, pred_area),
                                  arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                                  zorder=10)
                        ax.scatter(date, gt_area, color='red', s=100, marker='x', linewidths=3, zorder=11)
        
        # ============ SUV MEASUREMENTS (Right Y-axis) ============
        # Track if SUV data is plotted
        suv_data_plotted = False
        # Plot SUV per lesion
        for lesion_idx, lesion_id in enumerate(sorted(lesions)):
            lesion_data = hra_data[hra_data['Lesion_ID'] == lesion_id].sort_values('Date')
            
            pred_suv_mask = lesion_data['Pred_SUV'].notna()
            if pred_suv_mask.sum() > 0:
                dates_suv = lesion_data.loc[pred_suv_mask, 'Timepoint']
                suv_pred = lesion_data.loc[pred_suv_mask, 'Pred_SUV']
                ax_suv.plot(dates_suv, suv_pred, marker='D', linestyle='--', linewidth=2.5,
                           markersize=10, color='darkviolet', alpha=0.5,
                           markeredgecolor='black', markeredgewidth=1.5, zorder=4)
                suv_data_plotted = True
            
            # Add SUV error arrows
            for date_idx in lesion_data.index:
                if pd.notna(lesion_data.loc[date_idx, 'GT_SUV']) and pd.notna(lesion_data.loc[date_idx, 'Pred_SUV']):
                    date = lesion_data.loc[date_idx, 'Timepoint']
                    gt_suv = lesion_data.loc[date_idx, 'GT_SUV']
                    pred_suv = lesion_data.loc[date_idx, 'Pred_SUV']
                    if abs(gt_suv - pred_suv) > suv_error_threshold:
                        ax_suv.annotate('', xy=(date, gt_suv), xytext=(date, pred_suv),
                                      arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.9),
                                      zorder=10)
                        ax_suv.scatter(date, gt_suv, color='red', s=100, marker='x', linewidths=3, zorder=11)
        
        # Formatting
        ax.set_ylabel('Area (mm²)', fontweight='bold', fontsize=10, color=color)
        ax_suv.set_ylabel('SUV Max', fontweight='bold', fontsize=10, color='darkviolet')
        ax.set_title(f'{chr(65+hra_idx)}. {hra_name} ({n_lesions} lesion{"s" if n_lesions > 1 else ""}, n={len(hra_data)} obs)',
                    fontweight='bold', fontsize=10, loc='left')
        
        # Combined legend - optimized placement
        lines1, labels1 = ax.get_legend_handles_labels()
        
        # If only SUV data is plotted (no area data), create legend with just SUV Max
        if not lines1 and suv_data_plotted:
            lines1 = [plt.Line2D([0], [0], marker='D', color='darkviolet', linestyle='--',
                                markersize=10, markeredgecolor='black', alpha=0.5)]
            labels1 = ['SUV Max']
        
        # If area data exists, optionally add SUV Max if it was plotted (only if not already present)
        if lines1:
            if suv_data_plotted and 'SUV Max' not in labels1:
                lines1.append(plt.Line2D([0], [0], marker='D', color='darkviolet', linestyle='--',
                                        markersize=10, markeredgecolor='black', alpha=0.5))
                labels1.append('SUV Max')
            # Use 'best' location which automatically finds the least cluttered position
            if lines1:  # Only create legend if there are items to show
                ax.legend(lines1, labels1, fontsize=13, loc='best', frameon=True, 
                         fancybox=True, shadow=True, ncol=min(2, len(lines1)))
        
        ax.grid(alpha=0.3, linestyle='--')
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.tick_params(axis='x', rotation=0, labelsize=14)
        ax.tick_params(axis='y', labelsize=14, labelcolor=color)
        ax_suv.tick_params(axis='y', labelsize=14, labelcolor='darkviolet')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color(color)
        ax_suv.spines['right'].set_color('darkviolet')
        ax.set_xlim(p1_xlim)
        
        # Only show x-label on bottom row
        if row == n_rows - 1:
            ax.set_xlabel('Timepoint', fontweight='bold', fontsize=10)
        else:
            ax.set_xlabel('')

    fig_p1.suptitle(f'Patient 1 Lesion Trajectories (n={len(p1_df)} observations, {p1_df["Lesion_ID"].nunique()} unique lesions)',
                   fontsize=10, fontweight='bold', y=0.98)

    plt.savefig(PATIENT1_TRAJ_SVG, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Patient 1 trajectory figure saved!")

print("\nAnalysis complete!")

# ============================================================================
# GRANT APPLICATION FIGURES
# ============================================================================

print("\n" + "="*80)
print("GENERATING GRANT APPLICATION FIGURES")
print("="*80)

# Combine all patient data for comprehensive analysis
all_patient_data = []
print(f"\nProcessing {len(patient_sheets)} patients for grant figures...")

for patient_id in patient_sheets:
    patient_df = pd.read_excel(file_path, sheet_name=patient_id)
    patient_df = patient_df.dropna(subset=['Date'])
    patient_df['Date'] = pd.to_datetime(patient_df['Date'])
    patient_df['Patient_ID'] = patient_id
    
    # Add lesion IDs and areas
    patient_df = create_unique_lesion_ids(patient_df)
    patient_df['GT_Area_mm2'] = patient_df['GT_Long_Axis_mm'] * patient_df['GT_Short_Axis_mm']
    patient_df['Pred_Area_mm2'] = patient_df['Pred_Long_Axis_mm'] * patient_df['Pred_Short_Axis_mm']
    
    # Normalize dates to timepoints for this patient
    unique_dates = sorted(patient_df['Date'].unique())
    date_to_timepoint = {date: idx + 1 for idx, date in enumerate(unique_dates)}
    patient_df['Timepoint'] = patient_df['Date'].map(date_to_timepoint)
    
    all_patient_data.append(patient_df)
    print(f"  ✓ {patient_id}: {len(patient_df)} observations, {patient_df['Lesion_ID'].nunique()} lesions, {len(unique_dates)} timepoints")

combined_df = pd.concat(all_patient_data, ignore_index=True)
print(f"\nTotal combined: {len(combined_df)} observations across {combined_df['Patient_ID'].nunique()} patients")
print(f"Total unique lesions: {combined_df['Lesion_ID'].nunique()}")


# ============================================================================
# FIGURE 1: WATERFALL PLOT (Response Classification)
# ============================================================================
print("\n1. Creating Waterfall Plot...")

# Calculate percent change from baseline for each lesion
lesion_changes = []
for lesion_id in combined_df['Lesion_ID'].unique():
    if not lesion_id:
        continue
    
    lesion_data = combined_df[combined_df['Lesion_ID'] == lesion_id].sort_values('Date')
    
    # Get baseline (first non-NaN measurement)
    baseline_area = lesion_data['GT_Area_mm2'].dropna().iloc[0] if len(lesion_data['GT_Area_mm2'].dropna()) > 0 else None
    
    if baseline_area and baseline_area > 0:
        # Get nadir (minimum) and peak (maximum) measurements
        measurements = lesion_data['GT_Area_mm2'].dropna()
        if len(measurements) > 1:
            nadir = measurements.min()
            peak = measurements.max()
            
            # Calculate percent changes
            pct_change_nadir = ((nadir - baseline_area) / baseline_area) * 100
            pct_change_peak = ((peak - baseline_area) / baseline_area) * 100
            
            # Use the maximum absolute change (could be growth or shrinkage)
            max_change = pct_change_peak if abs(pct_change_peak) > abs(pct_change_nadir) else pct_change_nadir
            
            # RECIST 1.1 classification
            if max_change <= -30:
                response = 'PR (Partial Response)'
                color = '#2E86AB'  # Blue
            elif max_change >= 20:
                response = 'PD (Progressive Disease)'
                color = '#C73E1D'  # Red
            else:
                response = 'SD (Stable Disease)'
                color = '#6A994E'  # Green
            
            lesion_changes.append({
                'Lesion_ID': lesion_id,
                'Patient_ID': lesion_data['Patient_ID'].iloc[0],
                'HRA_Group': lesion_data['HRA_Group'].iloc[0],
                'Baseline_Area': baseline_area,
                'Max_Change_Percent': max_change,
                'Response': response,
                'Color': color
            })

waterfall_df = pd.DataFrame(lesion_changes)
waterfall_df = waterfall_df.sort_values('Max_Change_Percent', ascending=False)

# Create waterfall plot
fig_waterfall = plt.figure(figsize=(7, 5))
ax_waterfall = fig_waterfall.add_subplot(111)

x_pos = np.arange(len(waterfall_df))
bars = ax_waterfall.bar(x_pos, waterfall_df['Max_Change_Percent'],
                        color=waterfall_df['Color'], alpha=0.85, edgecolor='black', linewidth=2)

# Add RECIST threshold lines
ax_waterfall.axhline(y=-30, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, label='PR threshold (-30%)')
ax_waterfall.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='PD threshold (+20%)')
ax_waterfall.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.4)

# Formatting
ax_waterfall.set_xlabel('Lesion Index', fontweight='bold', fontsize=11)
ax_waterfall.set_ylabel('Maximum Change from Baseline (%)', fontweight='bold', fontsize=11)
ax_waterfall.set_title('Waterfall Plot: Lesion Response by RECIST 1.1 Criteria',
                       fontweight='bold', fontsize=12, pad=20)
ax_waterfall.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
ax_waterfall.spines['top'].set_visible(False)
ax_waterfall.spines['right'].set_visible(False)
ax_waterfall.tick_params(axis='both', labelsize=9, width=1, length=4)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E86AB', edgecolor='black', linewidth=2, label=f'PR: Partial Response (n={len(waterfall_df[waterfall_df["Response"]=="PR (Partial Response)"])})'),
    Patch(facecolor='#6A994E', edgecolor='black', linewidth=2, label=f'SD: Stable Disease (n={len(waterfall_df[waterfall_df["Response"]=="SD (Stable Disease)"])})'),
    Patch(facecolor='#C73E1D', edgecolor='black', linewidth=2, label=f'PD: Progressive Disease (n={len(waterfall_df[waterfall_df["Response"]=="PD (Progressive Disease)"])})')
]
ax_waterfall.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True)

# Add stats text
total_lesions = len(waterfall_df)
pr_rate = (len(waterfall_df[waterfall_df['Response'] == 'PR (Partial Response)']) / total_lesions) * 100
pd_rate = (len(waterfall_df[waterfall_df['Response'] == 'PD (Progressive Disease)']) / total_lesions) * 100
sd_rate = (len(waterfall_df[waterfall_df['Response'] == 'SD (Stable Disease)']) / total_lesions) * 100

stats_text = f'Total Lesions: {total_lesions}\nPR: {pr_rate:.1f}% | SD: {sd_rate:.1f}% | PD: {pd_rate:.1f}%'
ax_waterfall.text(0.02, 0.98, stats_text, transform=ax_waterfall.transAxes,
                 fontsize=10, fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

plt.tight_layout()
plt.savefig(WATERFALL_PLOT_SVG, bbox_inches='tight', facecolor='white')
plt.close()
print("   Waterfall plot saved")


# ============================================================================
# FIGURE 2: BLAND-ALTMAN PLOTS (Agreement Analysis)
# ============================================================================
print("2. Creating Bland-Altman Plots...")

fig_ba = plt.figure(figsize=(10, 4))
gs_ba = fig_ba.add_gridspec(1, 2, wspace=0.25)

# Bland-Altman for Area
ax_ba_area = fig_ba.add_subplot(gs_ba[0, 0])
area_data = combined_df[['GT_Area_mm2', 'Pred_Area_mm2']].dropna()

if len(area_data) > 0:
    gt_area = area_data['GT_Area_mm2'].values
    pred_area = area_data['Pred_Area_mm2'].values
    
    mean_area = (gt_area + pred_area) / 2
    diff_area = pred_area - gt_area
    
    mean_diff = np.mean(diff_area)
    std_diff = np.std(diff_area)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # Scatter plot
    ax_ba_area.scatter(mean_area, diff_area, alpha=0.7, s=40, edgecolors='black', linewidths=2)

    # Mean difference line
    ax_ba_area.axhline(mean_diff, color='blue', linestyle='--', linewidth=1.5, label=f'Mean Bias: {mean_diff:.1f} mm²')

    # Limits of agreement
    ax_ba_area.axhline(upper_loa, color='red', linestyle='--', linewidth=1.5, label=f'Upper LoA: {upper_loa:.1f} mm²')
    ax_ba_area.axhline(lower_loa, color='red', linestyle='--', linewidth=1.5, label=f'Lower LoA: {lower_loa:.1f} mm²')
    ax_ba_area.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.4)

    # Fill between LoA
    ax_ba_area.fill_between([mean_area.min(), mean_area.max()], lower_loa, upper_loa, alpha=0.15, color='red')

    ax_ba_area.set_xlabel('Mean of GT and Predicted Area (mm²)', fontweight='bold', fontsize=11)
    ax_ba_area.set_ylabel('Difference (Predicted - GT) (mm²)', fontweight='bold', fontsize=11)
    ax_ba_area.set_title('A. Bland-Altman Plot: Lesion Area', fontweight='bold', fontsize=11, loc='left')
    ax_ba_area.legend(loc='upper left', fontsize=9, frameon=True)
    ax_ba_area.grid(alpha=0.3, linestyle='--', linewidth=1.5)
    ax_ba_area.spines['top'].set_visible(False)
    ax_ba_area.spines['right'].set_visible(False)
    ax_ba_area.tick_params(axis='both', labelsize=9, width=1, length=4)

# Bland-Altman for SUV
ax_ba_suv = fig_ba.add_subplot(gs_ba[0, 1])
suv_data = combined_df[['GT_SUV', 'Pred_SUV']].dropna()

if len(suv_data) > 0:
    gt_suv = suv_data['GT_SUV'].values
    pred_suv = suv_data['Pred_SUV'].values
    
    mean_suv = (gt_suv + pred_suv) / 2
    diff_suv = pred_suv - gt_suv
    
    mean_diff_suv = np.mean(diff_suv)
    std_diff_suv = np.std(diff_suv)
    upper_loa_suv = mean_diff_suv + 1.96 * std_diff_suv
    lower_loa_suv = mean_diff_suv - 1.96 * std_diff_suv
    
    # Scatter plot
    ax_ba_suv.scatter(mean_suv, diff_suv, alpha=0.7, s=40, edgecolors='black', linewidths=2, color='purple')

    # Mean difference line
    ax_ba_suv.axhline(mean_diff_suv, color='blue', linestyle='--', linewidth=1.5, label=f'Mean Bias: {mean_diff_suv:.2f}')

    # Limits of agreement
    ax_ba_suv.axhline(upper_loa_suv, color='red', linestyle='--', linewidth=1.5, label=f'Upper LoA: {upper_loa_suv:.2f}')
    ax_ba_suv.axhline(lower_loa_suv, color='red', linestyle='--', linewidth=1.5, label=f'Lower LoA: {lower_loa_suv:.2f}')
    ax_ba_suv.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.4)

    # Fill between LoA
    ax_ba_suv.fill_between([mean_suv.min(), mean_suv.max()], lower_loa_suv, upper_loa_suv, alpha=0.15, color='red')

    ax_ba_suv.set_xlabel('Mean of GT and Predicted SUV Max', fontweight='bold', fontsize=11)
    ax_ba_suv.set_ylabel('Difference (Predicted - GT)', fontweight='bold', fontsize=11)
    ax_ba_suv.set_title('B. Bland-Altman Plot: SUV Max', fontweight='bold', fontsize=11, loc='left')
    ax_ba_suv.legend(loc='upper left', fontsize=9, frameon=True)
    ax_ba_suv.grid(alpha=0.3, linestyle='--', linewidth=1.5)
    ax_ba_suv.spines['top'].set_visible(False)
    ax_ba_suv.spines['right'].set_visible(False)
    ax_ba_suv.tick_params(axis='both', labelsize=9, width=1, length=4)

fig_ba.suptitle('Bland-Altman Analysis: Measurement Agreement', fontsize=12, fontweight='bold', y=0.98)
plt.savefig(BLAND_ALTMAN_SVG, bbox_inches='tight', facecolor='white')
plt.close()
print("   Bland-Altman plots saved")


# ============================================================================
# FIGURE 3: DETECTION RATE BY LESION SIZE
# ============================================================================
print("3. Creating Detection Rate by Size...")

# Define size categories
def categorize_size(area_mm2):
    if pd.isna(area_mm2):
        return 'Unknown'
    elif area_mm2 < 25:  # ~5mm × 5mm
        return '<5mm'
    elif area_mm2 < 100:  # ~10mm × 10mm
        return '5-10mm'
    elif area_mm2 < 400:  # ~20mm × 20mm
        return '10-20mm'
    else:
        return '>20mm'

combined_df['Size_Category'] = combined_df['GT_Area_mm2'].apply(categorize_size)

# Calculate detection rates
size_categories = ['<5mm', '5-10mm', '10-20mm', '>20mm']
detection_stats = []

for category in size_categories:
    cat_data = combined_df[combined_df['Size_Category'] == category]
    total = len(cat_data)
    
    if total > 0:
        # Detected = has both GT and Pred measurements
        detected = cat_data[(cat_data['GT_Area_mm2'].notna()) & (cat_data['Pred_Area_mm2'].notna())]
        detection_rate = (len(detected) / total) * 100
        
        # Accuracy: within 20% of GT area
        accurate = detected[abs((detected['Pred_Area_mm2'] - detected['GT_Area_mm2']) / detected['GT_Area_mm2']) <= 0.20]
        accuracy_rate = (len(accurate) / len(detected) * 100) if len(detected) > 0 else 0
        
        detection_stats.append({
            'Category': category,
            'Total': total,
            'Detection_Rate': detection_rate,
            'Accuracy_Rate': accuracy_rate
        })

detection_df = pd.DataFrame(detection_stats)

# Create figure
fig_detection = plt.figure(figsize=(7, 5))
ax_det = fig_detection.add_subplot(111)

x_pos = np.arange(len(detection_df))
width = 0.35

bars1 = ax_det.bar(x_pos - width/2, detection_df['Detection_Rate'], width,
                   label='Detection Rate', color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=2)
bars2 = ax_det.bar(x_pos + width/2, detection_df['Accuracy_Rate'], width,
                   label='Accuracy Rate (±20%)', color='#F18F01', alpha=0.85, edgecolor='black', linewidth=2)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_det.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add sample size labels
for i, (idx, row) in enumerate(detection_df.iterrows()):
    ax_det.text(i, -8, f'n={row["Total"]}', ha='center', fontsize=10, fontweight='bold', style='italic')

ax_det.set_xlabel('Lesion Size Category', fontweight='bold', fontsize=11)
ax_det.set_ylabel('Rate (%)', fontweight='bold', fontsize=11)
ax_det.set_title('Detection and Accuracy Rates by Lesion Size', fontweight='bold', fontsize=12, pad=20)
ax_det.set_xticks(x_pos)
ax_det.set_xticklabels(detection_df['Category'], fontsize=10, fontweight='bold')
ax_det.set_ylim(0, 105)
ax_det.legend(loc='lower right', fontsize=8, frameon=True)
ax_det.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
ax_det.spines['top'].set_visible(False)
ax_det.spines['right'].set_visible(False)
ax_det.tick_params(axis='both', labelsize=9, width=1, length=4)
ax_det.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5)

plt.tight_layout()
plt.savefig(DETECTION_BY_SIZE_SVG, bbox_inches='tight', facecolor='white')
plt.close()
print("   Detection by size plot saved")


# ============================================================================
# FIGURE 4: SPIDER PLOT (Individual Lesion Trajectories)
# ============================================================================
print("4. Creating Spider Plot...")

# ============================================================================
# DIAGNOSTIC: SPIDER PLOT REQUIREMENTS AND PATIENT 4 ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SPIDER PLOT REQUIREMENTS DIAGNOSTIC")
print("="*80)
print("\nRequirements for a lesion to be visualized in the spider plot:")
print("  1. Lesion_ID must not be empty")
print("  2. Must have at least one valid GT_Area_mm2 measurement that is:")
print("     - Not NaN")
print("     - Greater than 0")
print("  3. Baseline is defined as the first timepoint with valid data")
print("     (lesions can appear at any timepoint, not just timepoint 1)")
print("\nLesion trajectory types:")
print("  - Full trajectory: Lesion spans from first to last patient timepoint")
print("  - Appears later: Lesion first appears after patient's first timepoint (dashed line, ^ marker)")
print("  - Disappears early: Lesion disappears before patient's last timepoint (dotted line, v marker)")
print("  - Single point: Lesion only has data at one timepoint (square marker, no line)")
print("\n" + "-"*80)
print("PATIENT 4 TIMEPOINT ANALYSIS")
print("-"*80)

# Find patient 4 (assuming it's the 4th patient in sorted order)
all_patient_ids_sorted = sorted(combined_df['Patient_ID'].unique())
if len(all_patient_ids_sorted) >= 4:
    patient_4_id = all_patient_ids_sorted[3]  # 0-indexed, so 3 = 4th patient
    patient_4_data = combined_df[combined_df['Patient_ID'] == patient_4_id]
    
    print(f"\nPatient 4 ID: {patient_4_id}")
    print(f"Total observations: {len(patient_4_data)}")
    
    # Check unique dates/timepoints
    unique_dates_p4 = sorted(patient_4_data['Date'].unique())
    unique_timepoints_p4 = sorted(patient_4_data['Timepoint'].unique())
    print(f"Unique dates: {len(unique_dates_p4)}")
    print(f"  Dates: {[str(d.date()) for d in unique_dates_p4]}")
    print(f"Unique timepoints: {len(unique_timepoints_p4)}")
    print(f"  Timepoints: {unique_timepoints_p4}")
    
    # Check lesions
    unique_lesions_p4 = patient_4_data['Lesion_ID'].unique()
    print(f"\nUnique lesions: {len([l for l in unique_lesions_p4 if l])}")
    
    # Analyze each lesion
    print("\nLesion-by-lesion analysis:")
    for lesion_id in unique_lesions_p4:
        if not lesion_id:
            continue
        
        lesion_data = patient_4_data[patient_4_data['Lesion_ID'] == lesion_id].sort_values('Timepoint')
        
        # Check baseline
        baseline = lesion_data['GT_Area_mm2'].dropna().iloc[0] if len(lesion_data['GT_Area_mm2'].dropna()) > 0 else None
        
        # Calculate percent change
        lesion_data_copy = lesion_data.copy()
        lesion_data_copy['Pct_Change'] = ((lesion_data_copy['GT_Area_mm2'] - baseline) / baseline) * 100 if baseline and baseline > 0 else None
        
        mask = lesion_data_copy['Pct_Change'].notna()
        valid_timepoints = mask.sum()
        
        print(f"\n  Lesion: {lesion_id}")
        print(f"    Total observations: {len(lesion_data)}")
        print(f"    Timepoints with data: {sorted(lesion_data['Timepoint'].unique())}")
        print(f"    Baseline GT_Area_mm2: {baseline if baseline else 'MISSING'}")
        print(f"    Valid percent change timepoints: {valid_timepoints}")
        if valid_timepoints > 0:
            print(f"    Timepoint values: {lesion_data_copy.loc[mask, 'Timepoint'].tolist()}")
            print(f"    Percent changes: {lesion_data_copy.loc[mask, 'Pct_Change'].tolist()}")
        
        # Check if it meets requirements (now includes single timepoints)
        meets_requirements = (
            lesion_id and 
            baseline and baseline > 0
        )
        print(f"    ✓ Meets spider plot requirements: {meets_requirements}")
        if not meets_requirements:
            if not lesion_id:
                print(f"      ✗ Reason: Lesion_ID is empty")
            elif not baseline or baseline <= 0:
                print(f"      ✗ Reason: No valid baseline (baseline={baseline})")
        else:
            # Determine trajectory type
            first_tp = sorted(lesion_data['Timepoint'].unique())[0]
            last_tp = sorted(lesion_data['Timepoint'].unique())[-1]
            patient_tps = sorted(patient_4_data['Timepoint'].unique())
            patient_min_tp = min(patient_tps)
            patient_max_tp = max(patient_tps)
            
            if first_tp > patient_min_tp:
                traj_type = "appears later"
            elif last_tp < patient_max_tp:
                traj_type = "disappears early"
            elif valid_timepoints == 1:
                traj_type = "single point"
            else:
                traj_type = "full trajectory"
            print(f"      Trajectory type: {traj_type}")

print("\n" + "="*80)

fig_spider = plt.figure(figsize=(7, 5))
ax_spider = fig_spider.add_subplot(111)

# Generate distinct colors for patients
n_patients = combined_df['Patient_ID'].nunique()
patient_colors = plt.cm.tab10(np.linspace(0, 1, n_patients))

# Track which lesions are plotted per patient
plotted_lesions_summary = []
patient_idx_map = {pid: idx for idx, pid in enumerate(sorted(combined_df['Patient_ID'].unique()))}

# Collect data ranges for balanced axis limits
all_timepoints = []
all_pct_changes = []

# Get overall timepoint range across all patients
all_patient_timepoints = []
for patient_id in sorted(combined_df['Patient_ID'].unique()):
    patient_data = combined_df[combined_df['Patient_ID'] == patient_id]
    all_patient_timepoints.extend(patient_data['Timepoint'].unique())
overall_min_tp = min(all_patient_timepoints) if all_patient_timepoints else 1
overall_max_tp = max(all_patient_timepoints) if all_patient_timepoints else 12

for patient_id in sorted(combined_df['Patient_ID'].unique()):
    patient_data = combined_df[combined_df['Patient_ID'] == patient_id]
    patient_idx = patient_idx_map[patient_id]
    color = patient_colors[patient_idx]
    patient_label = f'Patient {patient_idx + 1}'
    
    # Get patient's timepoint range
    patient_timepoints = sorted(patient_data['Timepoint'].unique())
    patient_min_tp = min(patient_timepoints) if patient_timepoints else 1
    patient_max_tp = max(patient_timepoints) if patient_timepoints else 1
    
    patient_lesion_count = 0
    
    for lesion_id in patient_data['Lesion_ID'].unique():
        if not lesion_id:
            continue
        
        lesion_data = patient_data[patient_data['Lesion_ID'] == lesion_id].sort_values('Timepoint')
        
        # Get all valid area measurements
        valid_area_data = lesion_data[lesion_data['GT_Area_mm2'].notna() & (lesion_data['GT_Area_mm2'] > 0)]
        
        if len(valid_area_data) == 0:
            continue
        
        # Use first timepoint with valid data as baseline (not necessarily timepoint 1)
        baseline = valid_area_data.iloc[0]['GT_Area_mm2']
        baseline_timepoint = valid_area_data.iloc[0]['Timepoint']
        
        # Calculate percent change from baseline
        lesion_data_copy = lesion_data.copy()
        lesion_data_copy['Pct_Change'] = ((lesion_data_copy['GT_Area_mm2'] - baseline) / baseline) * 100
        
        # Get all valid timepoints with percent change
        mask = lesion_data_copy['Pct_Change'].notna()
        valid_data = lesion_data_copy.loc[mask].sort_values('Timepoint')
        
        if len(valid_data) == 0:
            continue
        
        timepoints = valid_data['Timepoint'].values
        pct_changes = valid_data['Pct_Change'].values
        
        # Determine lesion trajectory type
        first_tp = timepoints[0]
        last_tp = timepoints[-1]
        n_tps = len(timepoints)
        
        # Classify lesion appearance/disappearance - all use square markers
        if first_tp > patient_min_tp:
            lesion_type = 'appears_later'  # Lesion appears after first timepoint
            linestyle = '--'  # Dashed line
            marker = 's'  # Square
        elif last_tp < patient_max_tp:
            lesion_type = 'disappears_early'  # Lesion disappears before last timepoint
            linestyle = ':'  # Dotted line
            marker = 's'  # Square
        elif n_tps == 1:
            lesion_type = 'single_point'  # Only one timepoint
            linestyle = ''  # No line
            marker = 's'  # Square
        else:
            lesion_type = 'full_trajectory'  # Spans available timepoints
            linestyle = '-'  # Solid line
            marker = 's'  # Square
        
        # Collect for axis limit calculation
        all_timepoints.extend(timepoints)
        all_pct_changes.extend(pct_changes)
        
        # Plot the trajectory
        if n_tps > 1:
            # Multi-point trajectory with line
            if patient_lesion_count == 0:
                ax_spider.plot(timepoints, pct_changes,
                             marker=marker, linestyle=linestyle, linewidth=2, markersize=6,
                             alpha=0.8, color=color, label=patient_label,
                             markeredgecolor='black', markeredgewidth=1)
            else:
                ax_spider.plot(timepoints, pct_changes,
                             marker=marker, linestyle=linestyle, linewidth=2, markersize=6,
                             alpha=0.8, color=color,
                             markeredgecolor='black', markeredgewidth=1)
        else:
            # Single point - just marker, no line
            if patient_lesion_count == 0:
                ax_spider.scatter(timepoints, pct_changes,
                                marker=marker, s=50, alpha=0.8, color=color,
                                edgecolors='black', linewidths=2, label=patient_label, zorder=5)
            else:
                ax_spider.scatter(timepoints, pct_changes,
                                marker=marker, s=50, alpha=0.8, color=color,
                                edgecolors='black', linewidths=2, zorder=5)
        
        patient_lesion_count += 1
        plotted_lesions_summary.append({
            'Patient': patient_label,
            'Patient_ID': patient_id,
            'Lesion_ID': lesion_id,
            'HRA_Group': lesion_data['HRA_Group'].iloc[0],
            'Timepoints': n_tps,
            'First_Timepoint': first_tp,
            'Last_Timepoint': last_tp,
            'Baseline_Area': baseline,
            'Baseline_Timepoint': baseline_timepoint,
            'Lesion_Type': lesion_type,
            'Max_Change_Pct': pct_changes.max(),
            'Min_Change_Pct': pct_changes.min()
        })

# Add RECIST reference lines
ax_spider.axhline(y=-30, color='blue', linestyle='--', linewidth=1.5, alpha=0.6, label='PR threshold (-30%)')
ax_spider.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='PD threshold (+20%)')
ax_spider.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5)

ax_spider.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
ax_spider.set_ylabel('Change from Baseline (%)', fontweight='bold', fontsize=11)
ax_spider.set_title('Lesion Trajectories: Appearing/Disappearing Lesions',
                    fontsize=11, fontweight='bold', pad=15)
ax_spider.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=True, ncol=4)
ax_spider.grid(alpha=0.3, linestyle='--', linewidth=1.5)
ax_spider.spines['top'].set_visible(False)
ax_spider.spines['right'].set_visible(False)

# Set integer x-axis ticks
ax_spider.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_spider.tick_params(axis='both', labelsize=9, width=1, length=4)

# Set fixed x-axis range and balance y-axis for 3x3" plot
ax_spider.set_xlim(0, 12)

# Adjust y-axis to balance the plot area
if len(all_pct_changes) > 0:
    y_min, y_max = min(all_pct_changes), max(all_pct_changes)
    
    # Add small padding
    y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 5
    
    # Make y-axis range similar to x-axis range (12) for balanced appearance
    x_range = 12
    y_range = (y_max + y_padding) - (y_min - y_padding)
    
    if y_range < x_range:
        # Y range is smaller, expand Y to match X range
        y_center = (y_max + y_min) / 2
        ax_spider.set_ylim(y_center - x_range/2, y_center + x_range/2)
    else:
        # Y range is larger, use data range with padding
        ax_spider.set_ylim(y_min - y_padding, y_max + y_padding)

plt.tight_layout()
plt.savefig(SPIDER_PLOT_SVG, bbox_inches='tight', facecolor='white')
plt.close()

# Print summary table of plotted lesions
print("   Combined spider plot saved")

# ============================================================================
# CREATE INDIVIDUAL SPIDER PLOTS PER PATIENT
# ============================================================================
print("\n   Creating individual patient spider plots...")

# Generate distinct colors for lesions within each patient
lesion_colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Up to 12 different colors per patient

for patient_id in sorted(combined_df['Patient_ID'].unique()):
    patient_data = combined_df[combined_df['Patient_ID'] == patient_id]
    patient_idx = patient_idx_map[patient_id]
    patient_label = f'Patient {patient_idx + 1}'
    
    # Get patient's timepoint range
    patient_timepoints = sorted(patient_data['Timepoint'].unique())
    patient_min_tp = min(patient_timepoints) if patient_timepoints else 1
    patient_max_tp = max(patient_timepoints) if patient_timepoints else 1
    
    # Create figure with GridSpec: main plot on top, legend subplot below
    # Patient 2 gets slightly larger size since it has many lesions
    if patient_idx == 1:  # Patient 2 (0-indexed, so idx 1 = Patient 2)
        fig_patient = plt.figure(figsize=(7, 6))
        gs_patient = fig_patient.add_gridspec(2, 1, height_ratios=[4, 1.5], hspace=0.25)
    else:
        fig_patient = plt.figure(figsize=(6, 5))
        gs_patient = fig_patient.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25)
    
    ax_patient = fig_patient.add_subplot(gs_patient[0])
    ax_legend = fig_patient.add_subplot(gs_patient[1])
    
    # Collect data for this patient
    patient_timepoints_list = []
    patient_pct_changes = []
    patient_lesion_count = 0
    
    # Track lesions by location to explain multiple lesions in same location
    lesions_by_location = {}
    
    # Track lesion trajectory types for diagnostic output
    lesion_trajectory_info = []
    
    for lesion_idx, lesion_id in enumerate(sorted(patient_data['Lesion_ID'].unique())):
        if not lesion_id:
            continue
        
        lesion_data = patient_data[patient_data['Lesion_ID'] == lesion_id].sort_values('Timepoint')
        
        # Get all valid area measurements
        valid_area_data = lesion_data[lesion_data['GT_Area_mm2'].notna() & (lesion_data['GT_Area_mm2'] > 0)]
        
        if len(valid_area_data) == 0:
            continue
        
        # Use first timepoint with valid data as baseline
        baseline = valid_area_data.iloc[0]['GT_Area_mm2']
        baseline_timepoint = valid_area_data.iloc[0]['Timepoint']
        
        # Calculate percent change from baseline
        lesion_data_copy = lesion_data.copy()
        lesion_data_copy['Pct_Change'] = ((lesion_data_copy['GT_Area_mm2'] - baseline) / baseline) * 100
        
        # Get all valid timepoints with percent change
        mask = lesion_data_copy['Pct_Change'].notna()
        valid_data = lesion_data_copy.loc[mask].sort_values('Timepoint')
        
        if len(valid_data) == 0:
            continue
        
        timepoints = valid_data['Timepoint'].values
        pct_changes = valid_data['Pct_Change'].values
        
        # Determine lesion trajectory type
        first_tp = timepoints[0]
        last_tp = timepoints[-1]
        n_tps = len(timepoints)
        
        # Classify lesion appearance/disappearance - all use square markers
        if first_tp > patient_min_tp:
            lesion_type = 'appears_later'
            linestyle = '--'
            marker = 's'  # Square
        elif last_tp < patient_max_tp:
            lesion_type = 'disappears_early'
            linestyle = ':'
            marker = 's'  # Square
        elif n_tps == 1:
            lesion_type = 'single_point'
            linestyle = ''
            marker = 's'  # Square
        else:
            lesion_type = 'full_trajectory'
            linestyle = '-'
            marker = 's'  # Square
        
        # Collect for axis limits
        patient_timepoints_list.extend(timepoints)
        patient_pct_changes.extend(pct_changes)
        
        # Get lesion color (cycle through colors)
        lesion_color = lesion_colors[lesion_idx % len(lesion_colors)]
        
        # Create lesion label - use grant-friendly names for readability
        # Define readable name mappings for common anatomical locations
        readable_names = {
            'gastric lymph node_3': 'Gastric LN',
            'left supraclavicular lymph node_3': 'L. Supraclavicular LN',
            'right supraclavicular lymph node_3': 'R. Supraclavicular LN',
            'liver_no_hra': 'Liver',
            'liver_1': 'Liver',
            'liver_3': 'Liver',
            'lumbar lymph node_3': 'Lumbar LN',
            'lymph_node_no_hra': 'Lymph Node',
            'pelvic lymph node_3': 'Pelvic LN',
            'peritoneal mesentery_3': 'Peritoneal',
            'right lung_3': 'R. Lung',
            'left lung_3': 'L. Lung',
            'lung_3': 'Lung',
            'subcarinal lymph node_3': 'Subcarinal LN',
            'axillary lymph node_3': 'Axillary LN',
            'internal iliac lymph node_3': 'Internal Iliac LN',
            'bladder_no_hra': 'Bladder',
            'prostate_no_hra': 'Prostate',
            'bone_no_hra': 'Bone',
            'soft_tissue_no_hra': 'Soft Tissue',
        }
        
        if '_L' in lesion_id:
            base_name = lesion_id.split('_L')[0]
            lesion_num = lesion_id.split('_L')[1]  # "0", "1", "2", etc.
            
            # Track lesions by location for diagnostic info
            if base_name not in lesions_by_location:
                lesions_by_location[base_name] = []
            lesions_by_location[base_name].append('_L' + lesion_num)
            
            # Get readable base name
            readable_base = readable_names.get(base_name, base_name.replace('_', ' ').title())
            
            # Add number suffix if there are multiple lesions in same location
            # Check if this location has multiple lesions by looking ahead
            same_location_lesions = [lid for lid in patient_data['Lesion_ID'].unique() 
                                    if lid and lid.startswith(base_name + '_L')]
            if len(same_location_lesions) > 1:
                lesion_label = f"{readable_base} #{int(lesion_num) + 1}"
            else:
                lesion_label = readable_base
        else:
            lesion_label = readable_names.get(lesion_id, lesion_id.replace('_', ' ').title())
        
        # Plot the trajectory
        if n_tps > 1:
            ax_patient.plot(timepoints, pct_changes,
                          marker=marker, linestyle=linestyle, linewidth=2, markersize=7,
                          alpha=0.85, color=lesion_color, label=lesion_label,
                          markeredgecolor='black', markeredgewidth=1)
        else:
            ax_patient.scatter(timepoints, pct_changes,
                             marker=marker, s=50, alpha=0.85, color=lesion_color,
                             edgecolors='black', linewidths=2, label=lesion_label, zorder=5)
        
        # Store lesion trajectory info for diagnostic output
        lesion_trajectory_info.append({
            'label': lesion_label,
            'type': lesion_type,
            'first_tp': first_tp,
            'last_tp': last_tp,
            'patient_min_tp': patient_min_tp,
            'patient_max_tp': patient_max_tp
        })
        
        patient_lesion_count += 1
    
    # Add baseline reference line (0% change)
    ax_patient.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.4)

    # Formatting
    ax_patient.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
    ax_patient.set_ylabel('Change from Baseline (%)', fontweight='bold', fontsize=11)
    ax_patient.set_title(f'{patient_label} Lesion Trajectories',
                        fontsize=11, fontweight='bold', pad=15)
    
    # Set axis limits based on patient data
    if len(patient_timepoints_list) > 0:
        x_min, x_max = min(patient_timepoints_list), max(patient_timepoints_list)
        # Patient 2 gets more padding for wider axes
        x_padding = 1.0 if patient_idx == 1 else 0.5
        ax_patient.set_xlim(max(0, x_min - x_padding), min(12, x_max + x_padding))
    else:
        ax_patient.set_xlim(0, 12)
    
    if len(patient_pct_changes) > 0:
        y_min, y_max = min(patient_pct_changes), max(patient_pct_changes)
        # Patient 2 gets more padding for wider axes
        y_padding = (y_max - y_min) * 0.15 if patient_idx == 1 else (y_max - y_min) * 0.1
        if y_padding < 5:
            y_padding = 10 if patient_idx == 1 else 5
        ax_patient.set_ylim(y_min - y_padding, y_max + y_padding)
    
    ax_patient.grid(alpha=0.3, linestyle='--', linewidth=1.5)
    ax_patient.spines['top'].set_visible(False)
    ax_patient.spines['right'].set_visible(False)
    ax_patient.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_patient.tick_params(axis='both', labelsize=9, width=1, length=4)

    # Legend - in separate subplot below the main plot
    # Get handles and labels from the main plot
    handles, labels = ax_patient.get_legend_handles_labels()

    # Clear the legend subplot and turn off its axes
    ax_legend.axis('off')

    # Calculate number of columns based on number of items
    ncol_legend = 3 if patient_lesion_count > 6 else 2
    if patient_lesion_count <= 4:
        ncol_legend = 2

    # Place legend in the dedicated subplot
    ax_legend.legend(handles, labels, loc='center', fontsize=8,
                    frameon=True, ncol=ncol_legend, fancybox=True, shadow=True,
                    borderaxespad=0)
    
    # Save individual patient plot
    patient_plot_svg = SPIDER_PLOT_PATIENT_SVG.format(patient_idx + 1)

    plt.tight_layout()
    plt.savefig(patient_plot_svg, bbox_inches='tight', facecolor='white')
    plt.close()

    # Print diagnostic info about lesion trajectories
    print(f"      {patient_label}: {patient_lesion_count} lesions")
    
    # Check for appearing/disappearing lesions
    appears_later = [l for l in lesion_trajectory_info if l['type'] == 'appears_later']
    disappears_early = [l for l in lesion_trajectory_info if l['type'] == 'disappears_early']
    single_points = [l for l in lesion_trajectory_info if l['type'] == 'single_point']
    full_trajectories = [l for l in lesion_trajectory_info if l['type'] == 'full_trajectory']
    
    if appears_later:
        print(f"        📈 NEW LESIONS (appear after TP 1):")
        for l in appears_later:
            print(f"           - {l['label']}: first seen at TP {l['first_tp']}")
    
    if disappears_early:
        print(f"        📉 RESOLVED LESIONS (disappear before last TP):")
        for l in disappears_early:
            print(f"           - {l['label']}: last seen at TP {l['last_tp']} (patient has data until TP {l['patient_max_tp']})")
    
    if single_points:
        print(f"        📍 SINGLE OBSERVATION:")
        for l in single_points:
            print(f"           - {l['label']}: only at TP {l['first_tp']}")
    
    if full_trajectories:
        print(f"        ✓ Full trajectories: {len(full_trajectories)} lesion(s)")
    
    # Multi-lesion locations
    multi_lesion_locations = {loc: nums for loc, nums in lesions_by_location.items() if len(nums) > 1}
    if multi_lesion_locations:
        print(f"        Note: Multiple lesions in same location:")
        for loc, nums in multi_lesion_locations.items():
            print(f"          - {loc}: {len(nums)} lesions")
print("\n   Spider Plot Lesion Summary:")
print("   " + "="*80)
spider_summary_df = pd.DataFrame(plotted_lesions_summary)
if len(spider_summary_df) > 0:
    for patient in spider_summary_df['Patient'].unique():
        patient_lesions = spider_summary_df[spider_summary_df['Patient'] == patient]
        print(f"\n   {patient}: {len(patient_lesions)} lesions")
        for _, lesion in patient_lesions.iterrows():
            print(f"      • {lesion['Lesion_ID']}")
            print(f"        Location: {lesion['HRA_Group']}")
            print(f"        Type: {lesion['Lesion_Type']}")
            print(f"        Timepoints: {lesion['Timepoints']} (TP {lesion['First_Timepoint']}-{lesion['Last_Timepoint']})")
            print(f"        Baseline: {lesion['Baseline_Area']:.1f} mm² at TP {lesion['Baseline_Timepoint']}")
            if lesion['Timepoints'] > 1:
                print(f"        Change range: {lesion['Min_Change_Pct']:.1f}% to {lesion['Max_Change_Pct']:.1f}%")
            else:
                print(f"        Single point: {lesion['Max_Change_Pct']:.1f}% change")
    
    # Summary statistics
    print("\n   " + "-"*80)
    print("   Summary Statistics:")
    print(f"   Total lesions plotted: {len(spider_summary_df)}")
    print(f"   Full trajectories: {len(spider_summary_df[spider_summary_df['Lesion_Type']=='full_trajectory'])}")
    print(f"   Appear later: {len(spider_summary_df[spider_summary_df['Lesion_Type']=='appears_later'])}")
    print(f"   Disappear early: {len(spider_summary_df[spider_summary_df['Lesion_Type']=='disappears_early'])}")
    print(f"   Single point: {len(spider_summary_df[spider_summary_df['Lesion_Type']=='single_point'])}")
    print("   " + "="*80)



# ============================================================================
# FIGURE 5: TOTAL TUMOR BURDEN OVER TIME
# ============================================================================
print("5. Creating Total Tumor Burden Plot...")

# Use a broken y-axis so very large burdens don't compress the rest
fig_burden, (ax_burden_top, ax_burden_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 5),
    gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05}
)

# Only include Patients 1, 2, 4, and 5 (based on sorted patient order)
all_patient_ids_sorted = sorted(combined_df['Patient_ID'].unique())
global_idx_map = {pid: idx for idx, pid in enumerate(all_patient_ids_sorted)}
included_patient_numbers = {1, 2, 4, 5}
selected_patient_ids = [
    pid for pid in all_patient_ids_sorted
    if (global_idx_map[pid] + 1) in included_patient_numbers
]

n_patients = len(selected_patient_ids)
colors_patients_list = plt.cm.tab10(np.linspace(0, 1, n_patients)) if n_patients > 0 else []

# Collect all y values to set sensible limits
all_burden_values = []

for idx, patient_id in enumerate(selected_patient_ids):
    patient_data = combined_df[combined_df['Patient_ID'] == patient_id]
    
    # Calculate total tumor burden per timepoint
    burden_by_timepoint = patient_data.groupby('Timepoint').agg({
        'GT_Area_mm2': 'sum',
        'Pred_Area_mm2': 'sum'
    }).reset_index()
    
    burden_by_timepoint = burden_by_timepoint.sort_values('Timepoint')
    all_burden_values.extend(burden_by_timepoint['GT_Area_mm2'].tolist())
    all_burden_values.extend(burden_by_timepoint['Pred_Area_mm2'].tolist())
    
    # Use global index for patient labeling to stay consistent with earlier figures
    global_idx = global_idx_map[patient_id]
    patient_label = f'Patient {global_idx + 1}'
    color = colors_patients_list[idx]
    
    # Plot GT burden
    ax_burden_top.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['GT_Area_mm2'],
                       marker='o', linewidth=2, markersize=8,
                       label=f'{patient_label} (GT)',
                       color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=1)
    ax_burden_bottom.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['GT_Area_mm2'],
                          marker='o', linewidth=2, markersize=8,
                          label=f'{patient_label} (GT)',
                          color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=1)

    # Plot Predicted burden
    ax_burden_top.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['Pred_Area_mm2'],
                       marker='s', linewidth=2, markersize=7, linestyle='--',
                       label=f'{patient_label} (Pred)',
                       color=color, alpha=0.7, markeredgecolor='black', markeredgewidth=1)
    ax_burden_bottom.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['Pred_Area_mm2'],
                          marker='s', linewidth=2, markersize=7, linestyle='--',
                          label=f'{patient_label} (Pred)',
                          color=color, alpha=0.7, markeredgecolor='black', markeredgewidth=1)

# Set y-limits with a break
if len(all_burden_values) > 0:
    y_max = max(all_burden_values)
else:
    y_max = 1.0

lower_max = 1200  # top of the lower panel
gap_factor = 2.0  # size of the gap between panels, relative to lower_max

ax_burden_bottom.set_ylim(0, lower_max)
ax_burden_top.set_ylim(lower_max * gap_factor, y_max * 1.05)

# Hide the spines between ax_burden_top and ax_burden_bottom
ax_burden_top.spines['bottom'].set_visible(False)
ax_burden_bottom.spines['top'].set_visible(False)
ax_burden_top.tick_params(labeltop=False)  # don't put tick labels at the top subplot's bottom
ax_burden_bottom.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Labels, title, and legend
ax_burden_bottom.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
ax_burden_bottom.set_ylabel('Total Tumor Burden (mm²)', fontweight='bold', fontsize=11)
fig_burden.suptitle(f'Total Tumor Burden Over Time (n={n_patients} patients)',
                    fontweight='bold', fontsize=13, y=0.97)

ax_burden_bottom.grid(alpha=0.3, linestyle='--', linewidth=1.5)
ax_burden_top.grid(alpha=0.3, linestyle='--', linewidth=1.5)

for ax in (ax_burden_top, ax_burden_bottom):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9, width=1, length=4)

ax_burden_top.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, frameon=True, ncol=1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(TUMOR_BURDEN_SVG, bbox_inches='tight', facecolor='white')
plt.close()
print("   Total tumor burden plot saved")

# ============================================================================
# FIGURE 6: PERCIST TOTAL METABOLIC TUMOR BURDEN OVER TIME
# ============================================================================
print("6. Creating PERCIST Metabolic Tumor Burden Plot...")

# Single panel - no breaks needed
fig_percist = plt.figure(figsize=(7, 5))
ax_percist = fig_percist.add_subplot(111)

# Include same patients as RECIST analysis
all_patient_ids_sorted = sorted(combined_df['Patient_ID'].unique())
global_idx_map_percist = {pid: idx for idx, pid in enumerate(all_patient_ids_sorted)}
included_patient_numbers_percist = {1, 2, 4, 5}
selected_patient_ids_percist = [
    pid for pid in all_patient_ids_sorted
    if (global_idx_map_percist[pid] + 1) in included_patient_numbers_percist
]

n_patients_percist = len(selected_patient_ids_percist)
colors_patients_percist = plt.cm.tab10(np.linspace(0, 1, n_patients_percist)) if n_patients_percist > 0 else []

# Collect all y values to set sensible limits for metabolic burden
all_metabolic_burden_values = []

for idx, patient_id in enumerate(selected_patient_ids_percist):
    patient_data = combined_df[combined_df['Patient_ID'] == patient_id]
    
    # Calculate total metabolic burden per timepoint (sum of SUV across all lesions)
    metabolic_by_timepoint = patient_data.groupby('Timepoint').agg({
        'GT_SUV': 'sum',
        'Pred_SUV': 'sum'
    }).reset_index()
    
    metabolic_by_timepoint = metabolic_by_timepoint.sort_values('Timepoint')
    all_metabolic_burden_values.extend(metabolic_by_timepoint['GT_SUV'].tolist())
    all_metabolic_burden_values.extend(metabolic_by_timepoint['Pred_SUV'].tolist())
    
    # Use global index for patient labeling
    global_idx = global_idx_map_percist[patient_id]
    patient_label = f'Patient {global_idx + 1}'
    color = colors_patients_percist[idx]
    
    # Plot GT metabolic burden
    ax_percist.plot(metabolic_by_timepoint['Timepoint'], metabolic_by_timepoint['GT_SUV'],
                    marker='o', linewidth=2, markersize=8,
                    label=f'{patient_label} (GT)',
                    color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=1)

    # Plot Predicted metabolic burden
    ax_percist.plot(metabolic_by_timepoint['Timepoint'], metabolic_by_timepoint['Pred_SUV'],
                    marker='s', linewidth=2, markersize=7, linestyle='--',
                    label=f'{patient_label} (Pred)',
                    color=color, alpha=0.7, markeredgecolor='black', markeredgewidth=1)

# Set y-limits to 0-30
ax_percist.set_ylim(0, 30)
ax_percist.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Labels, title, and legend
ax_percist.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
ax_percist.set_ylabel('Total Metabolic Burden (SUV Sum)', fontweight='bold', fontsize=11)
fig_percist.suptitle(f'PERCIST: Total Metabolic Tumor Burden Over Time (n={n_patients_percist} patients)',
                     fontweight='bold', fontsize=13, y=0.97)

ax_percist.grid(alpha=0.3, linestyle='--', linewidth=1.5)
ax_percist.spines['right'].set_visible(False)
ax_percist.spines['top'].set_visible(False)
ax_percist.tick_params(axis='both', labelsize=9, width=1, length=4)

ax_percist.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, frameon=True, ncol=1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(PERCIST_BURDEN_SVG, bbox_inches='tight', facecolor='white')
plt.close()
print("   PERCIST metabolic burden plot saved")

# ============================================================================
# COMBINED FIGURE: RECIST + PERCIST (Aligned X-axes)
# ============================================================================
print("7. Creating Combined RECIST + PERCIST Plot...")

# Create figure with GridSpec for better control
from matplotlib.gridspec import GridSpec
fig_combined = plt.figure(figsize=(7, 9))
gs_combined = GridSpec(3, 1, figure=fig_combined, hspace=0.15, height_ratios=[1, 3, 4])

# RECIST subplot with broken y-axis (top 2 panels)
ax_recist_top_combined = fig_combined.add_subplot(gs_combined[0])
ax_recist_bottom_combined = fig_combined.add_subplot(gs_combined[1], sharex=ax_recist_top_combined)

# PERCIST subplot (bottom panel)
ax_percist_combined = fig_combined.add_subplot(gs_combined[2], sharex=ax_recist_bottom_combined)

# Re-plot RECIST data on combined figure
for idx, patient_id in enumerate(selected_patient_ids):
    patient_data = combined_df[combined_df['Patient_ID'] == patient_id]
    
    burden_by_timepoint = patient_data.groupby('Timepoint').agg({
        'GT_Area_mm2': 'sum',
        'Pred_Area_mm2': 'sum'
    }).reset_index()
    
    burden_by_timepoint = burden_by_timepoint.sort_values('Timepoint')
    
    global_idx = global_idx_map[patient_id]
    patient_label = f'Patient {global_idx + 1}'
    color = colors_patients_list[idx]
    
    # Plot GT burden
    ax_recist_top_combined.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['GT_Area_mm2'],
                               marker='o', linewidth=2, markersize=8,
                               label=f'{patient_label} (GT)',
                               color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=1)
    ax_recist_bottom_combined.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['GT_Area_mm2'],
                                  marker='o', linewidth=2, markersize=8,
                                  label=f'{patient_label} (GT)',
                                  color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=1)

    # Plot Predicted burden
    ax_recist_top_combined.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['Pred_Area_mm2'],
                               marker='s', linewidth=2, markersize=7, linestyle='--',
                               label=f'{patient_label} (Pred)',
                               color=color, alpha=0.7, markeredgecolor='black', markeredgewidth=1)
    ax_recist_bottom_combined.plot(burden_by_timepoint['Timepoint'], burden_by_timepoint['Pred_Area_mm2'],
                                  marker='s', linewidth=2, markersize=7, linestyle='--',
                                  label=f'{patient_label} (Pred)',
                                  color=color, alpha=0.7, markeredgecolor='black', markeredgewidth=1)

# Set RECIST y-limits
if len(all_burden_values) > 0:
    y_max = max(all_burden_values)
else:
    y_max = 1.0

lower_max = 1200
gap_factor = 2.0

ax_recist_bottom_combined.set_ylim(0, lower_max)
ax_recist_top_combined.set_ylim(lower_max * gap_factor, y_max * 1.05)

# Hide spines for RECIST
ax_recist_top_combined.spines['bottom'].set_visible(False)
ax_recist_bottom_combined.spines['top'].set_visible(False)
ax_recist_top_combined.tick_params(labeltop=False, labelbottom=False)
ax_recist_bottom_combined.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# RECIST labels - hide x-axis labels on top panel
ax_recist_bottom_combined.set_ylabel('Total Tumor Burden (mm²)', fontweight='bold', fontsize=11)
ax_recist_bottom_combined.set_xlabel('')  # Hide x-label on RECIST bottom (will show on PERCIST)

ax_recist_bottom_combined.grid(alpha=0.3, linestyle='--', linewidth=1.5)
ax_recist_top_combined.grid(alpha=0.3, linestyle='--', linewidth=1.5)

for ax in (ax_recist_top_combined, ax_recist_bottom_combined):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9, width=1, length=4)

ax_recist_top_combined.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, frameon=True, ncol=1)

# Re-plot PERCIST data on combined figure
for idx, patient_id in enumerate(selected_patient_ids_percist):
    patient_data = combined_df[combined_df['Patient_ID'] == patient_id]

    metabolic_by_timepoint = patient_data.groupby('Timepoint').agg({
        'GT_SUV': 'sum',
        'Pred_SUV': 'sum'
    }).reset_index()

    metabolic_by_timepoint = metabolic_by_timepoint.sort_values('Timepoint')

    global_idx = global_idx_map_percist[patient_id]
    patient_label = f'Patient {global_idx + 1}'
    color = colors_patients_percist[idx]

    # Plot GT metabolic burden
    ax_percist_combined.plot(metabolic_by_timepoint['Timepoint'], metabolic_by_timepoint['GT_SUV'],
                            marker='o', linewidth=2, markersize=8,
                            label=f'{patient_label} (GT)',
                            color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=1)

    # Plot Predicted metabolic burden
    ax_percist_combined.plot(metabolic_by_timepoint['Timepoint'], metabolic_by_timepoint['Pred_SUV'],
                            marker='s', linewidth=2, markersize=7, linestyle='--',
                            label=f'{patient_label} (Pred)',
                            color=color, alpha=0.7, markeredgecolor='black', markeredgewidth=1)

# Set PERCIST y-limits
ax_percist_combined.set_ylim(0, 30)
ax_percist_combined.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# PERCIST labels
ax_percist_combined.set_xlabel('Timepoint', fontweight='bold', fontsize=11)
ax_percist_combined.set_ylabel('Total Metabolic Burden (SUV Sum)', fontweight='bold', fontsize=11)

ax_percist_combined.grid(alpha=0.3, linestyle='--', linewidth=1.5)
ax_percist_combined.spines['right'].set_visible(False)
ax_percist_combined.spines['top'].set_visible(False)
ax_percist_combined.tick_params(axis='both', labelsize=9, width=1, length=4)

ax_percist_combined.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, frameon=True, ncol=1)

# Overall title
fig_combined.suptitle(f'Tumor Burden Over Time: RECIST vs PERCIST (n={n_patients} patients)',
                     fontweight='bold', fontsize=13, y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.97])
COMBINED_BURDEN_SVG = OUTPUT_DIR + 'combined_recist_percist_burden.svg'
plt.savefig(COMBINED_BURDEN_SVG, bbox_inches='tight', facecolor='white')
plt.close()
print("   Combined RECIST + PERCIST plot saved")

print("\n" + "="*80)
print("ALL GRANT APPLICATION FIGURES COMPLETED!")
print("="*80)
print("\nGenerated files:")
print("  1. Waterfall Plot - Response classification by RECIST 1.1")
print("  2. Bland-Altman Plots - Measurement agreement (Area & SUV)")
print("  3. Detection by Size - Performance across size categories")
print("  4. Spider Plot - Combined individual lesion trajectories (all patients)")
print("     + Individual patient spider plots (spider_plot_patient1.svg, etc.)")
print("  5. Total Tumor Burden - Disease burden over time")
print("  6. PERCIST Metabolic Burden - Metabolic disease burden over time")
print("  7. Combined RECIST + PERCIST - Aligned tumor burden comparison")
print("\nDataset Summary:")
print(f"  - Total Patients: {combined_df['Patient_ID'].nunique()}")
print(f"  - Total Observations: {len(combined_df)}")
print(f"  - Total Unique Lesions: {combined_df['Lesion_ID'].nunique()}")
print(f"  - Date Range: {combined_df['Date'].min().strftime('%Y-%m-%d')} to {combined_df['Date'].max().strftime('%Y-%m-%d')}")
print("="*80)