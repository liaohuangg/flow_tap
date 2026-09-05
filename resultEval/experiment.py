# \begin{table*}[tbp] 
#   \centering
#   \caption{Performance Comparison of Different Matching Methods}
#   \label{tab:matching_compare}
  
#   % 自定义固定宽度列（自己改数字就行）
#   \newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}} 
  
#   \begin{tabular}{
#     C{1.0cm}  % Case 列（现在也居中了）
#     C{1.0cm} C{0.9cm} C{1.0cm} C{0.8cm}  % Flow Match
#     C{1.0cm} C{0.9cm} C{1.0cm} C{0.8cm}  % Thermal-Guided FM
#     C{1.0cm} C{0.9cm} C{1.0cm} C{0.8cm}  % TAP2.5D
#   }
#     \toprule
#     \multirow{2}{*}{Cases}
#     & \multicolumn{4}{c}{FM}
#     & \multicolumn{4}{c}{TFM}
#     & \multicolumn{4}{c}{TAP2.5D} \\
#     \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}
#     & BA($\mathrm{mm}^2$) & WL(mm) & maxT($^\circ$C) & RT(s)
#     & BA($\mathrm{mm}^2$) & WL(mm) & maxT($^\circ$C) & RT(s)
#     & BA($\mathrm{mm}^2$) & WL(mm) & maxT($^\circ$C) & RT(s) \\
#     \midrule
#             Case 1  & 1499.30 & 45717.60 & 102.75  & 10.88 & 1575.17 & 52462.70 & 97.71 & 59.98 & 1600.00 & 44797.81 & 99.35 & 7550 \\
#             Case 2  & 1752.18 & 48507.23 & 75.81 & 11.17 & 1935.28 & 39650.76 & 75.74 & 61.79 & 1790.80 & 33033.66 & 76.23 & 8127 \\
#             Case 3  &  774.47 & 83030.00& 120.79 & 10.77 & 1118.05 & 116463.40 & 105.95 & 55.90& 1295.75 & 209893.36& 103.16& 6611 \\
#             Case 4  & 2642.18 & 146021.99& 128.55 & 20.84 & 3615.45 & 102988.06& 120.25 & 56.71 & 3249.00 & 128161.25& 112.07& 11385 \\
#             Case 5  & 1362.65 & 74017.52& 135.18  & 10.42 & 2428.42 & 140437.60& 115.43 & 55.14 & 2068.00 & 216750.70& 113.37& 9198 \\
#             Case 6  & 1772.79 & 113376.66&  108.08 & 13.43 & 1988.73 & 120103.93& 101.95 & 56.54 & 2550.00 & 119125.50& 97.41 & 11277 \\
#             Case 7  &  603.49 & 36289.27 &   81.84 & 27.60 &  2077.77 &63508.26 & 70.81 & 63.85 & 1381.98 & 28319.78 & 74.46 & 13771 \\
#             Case 8  &  353.47 & 29269.80 &  81.39 & 37.55 &  1138.91 & 60531.09 & 70.70 & 73.91 & 1239.50 & 28164.88 & 73.15 & 20614 \\
#             Case 9  & 2833.52 & 198294.03& 133.71 & 43.68 &4843.34 & 279503.32 & 110.86 & 90.13 & 5094.38 & 219844.09& 108.16& 62034 \\
#             Case 10 & 1894.14 & 93648.21& 115.54 & 100.30&  4546.08 &  150607.63& 95.14 & 148.98 & 5054.96 & 231451.30& 97.13 & 150536 \\
#             Avg & 1548.82 & 86817.23 & - & 28.66 & 2526.72 & 112625.68 & - & 72.29 & 2532.44 & 125954.23 & - & 30110 \\
#     \bottomrule
#   \end{tabular}
# \end{table*}
DATA = [
    {
        "case": "Case 1",
        "FM": {"BA": 1499.30, "WL": 45717.60, "peakT": 102.75, "RT": 10.88},
        "TFM": {"BA": 1575.17, "WL": 52462.70, "peakT": 97.71, "RT": 59.98},
        "TAP2.5D": {"BA": 1600.00, "WL": 44797.81, "peakT": 99.35, "RT": 7550.00},
    },
    {
        "case": "Case 2",
        "FM": {"BA": 1752.18, "WL": 48507.23, "peakT": 75.81, "RT": 11.17},
        "TFM": {"BA": 1935.28, "WL": 39650.76, "peakT": 75.74, "RT": 61.79},
        "TAP2.5D": {"BA": 1790.80, "WL": 33033.66, "peakT": 76.23, "RT": 8127.00},
    },
    {
        "case": "Case 3",
        "FM": {"BA": 774.47, "WL": 83030.00, "peakT": 120.79, "RT": 10.77},
        "TFM": {"BA": 1118.05, "WL": 116463.40, "peakT": 105.95, "RT": 55.90},
        "TAP2.5D": {"BA": 1295.75, "WL": 209893.36, "peakT": 103.16, "RT": 6611.00},
    },
    {
        "case": "Case 4",
        "FM": {"BA": 2642.18, "WL": 146021.99, "peakT": 128.55, "RT": 20.84},
        "TFM": {"BA": 3615.45, "WL": 102988.06, "peakT": 120.25, "RT": 56.71},
        "TAP2.5D": {"BA": 3249.00, "WL": 128161.25, "peakT": 112.07, "RT": 11385.00},
    },
    {
        "case": "Case 5",
        "FM": {"BA": 1362.65, "WL": 74017.52, "peakT": 135.18, "RT": 10.42},
        "TFM": {"BA": 2428.42, "WL": 140437.60, "peakT": 115.43, "RT": 55.14},
        "TAP2.5D": {"BA": 2068.00, "WL": 216750.70, "peakT": 113.37, "RT": 9198.00},
    },
    {
        "case": "Case 6",
        "FM": {"BA": 1772.79, "WL": 113376.66, "peakT": 108.08, "RT": 13.43},
        "TFM": {"BA": 1988.73, "WL": 120103.93, "peakT": 101.95, "RT": 56.54},
        "TAP2.5D": {"BA": 2550.00, "WL": 119125.50, "peakT": 97.41, "RT": 11277.00},
    },
    {
        "case": "Case 7",
        "FM": {"BA": 603.49, "WL": 36289.27, "peakT": 81.84, "RT": 27.60},
        "TFM": {"BA": 2077.77, "WL": 63508.26, "peakT": 70.81, "RT": 63.85},
        "TAP2.5D": {"BA": 1381.98, "WL": 28319.78, "peakT": 74.46, "RT": 13771.00},
    },
    {
        "case": "Case 8",
        "FM": {"BA": 353.47, "WL": 29269.80, "peakT": 81.39, "RT": 37.55},
        "TFM": {"BA": 1138.91, "WL": 60531.09, "peakT": 70.70, "RT": 73.91},
        "TAP2.5D": {"BA": 1239.50, "WL": 28164.88, "peakT": 73.15, "RT": 20614.00},
    },
    {
        "case": "Case 9",
        "FM": {"BA": 2833.52, "WL": 198294.03, "peakT": 133.71, "RT": 43.68},
        "TFM": {"BA": 4843.34, "WL": 279503.32, "peakT": 110.86, "RT": 90.13},
        "TAP2.5D": {"BA": 5094.38, "WL": 219844.09, "peakT": 108.16, "RT": 62034.00},
    },
    {
        "case": "Case 10",
        "FM": {"BA": 1894.14, "WL": 93648.21, "peakT": 115.54, "RT": 100.30},
        "TFM": {"BA": 4546.08, "WL": 150607.63, "peakT": 95.14, "RT": 148.98},
        "TAP2.5D": {"BA": 5054.96, "WL": 231451.30, "peakT": 97.13, "RT": 150536.00},
    },
    {
        "case": "Avg",
        "FM": {"BA": 1548.82, "WL": 86817.23, "peakT": 108.36, "RT": 28.66},
        "TFM": {"BA": 2526.72, "WL": 112625.68, "peakT": 96.45, "RT": 72.29},
        "TAP2.5D": {"BA": 2532.44, "WL": 125954.23, "peakT": 95.45, "RT": 30110.30},
    },
]


def percent_improvement(target, baseline):
    return (baseline - target) / baseline * 100


def temperature_difference(target, baseline):
    return target - baseline


def speedup(target_rt, baseline_rt):
    return baseline_rt / target_rt


def describe_percent(metric_name, target_method, baseline_method, value):
    direction = "小" if value >= 0 else "大"
    return f"{direction} {abs(value):.2f}%"


def describe_temperature(target_method, baseline_method, value):
    direction = "高" if value >= 0 else "低"
    return f" {direction} {abs(value):.2f}°C"


def describe_speedup(target_method, baseline_method, value):
    if value >= 1:
        return f"快 {value:.2f}x"
    return f"{value:.2f}x"


def build_comparison_rows(target_method, baseline_methods):
    rows = []
    for row in DATA:
        target = row[target_method]
        comparison = {"Case": row["case"]}
        for baseline_method in baseline_methods:
            prefix = baseline_method.replace(".", "")
            baseline = row[baseline_method]
            ba_value = percent_improvement(target["BA"], baseline["BA"])
            wl_value = percent_improvement(target["WL"], baseline["WL"])
            peak_value = temperature_difference(target["peakT"], baseline["peakT"])
            speedup_value = speedup(target["RT"], baseline["RT"])
            comparison[f"BA_vs_{prefix}"] = describe_percent("BA", target_method, baseline_method, ba_value)
            comparison[f"WL_vs_{prefix}"] = describe_percent("WL", target_method, baseline_method, wl_value)
            comparison[f"peakT_vs_{prefix}"] = describe_temperature(target_method, baseline_method, peak_value)
            comparison[f"RT_vs_{prefix}"] = describe_speedup(target_method, baseline_method, speedup_value)
        rows.append(comparison)
    return rows


def format_value(value, header):
    return value


def print_table(title, rows):
    headers = list(rows[0].keys())
    formatted_rows = [[format_value(row[header], header) for header in headers] for row in rows]
    widths = [max(len(header), *(len(row[index]) for row in formatted_rows)) for index, header in enumerate(headers)]

    print(title)
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in formatted_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print()


def main():
    fm_rows = build_comparison_rows("FM", ["TFM", "TAP2.5D"])
    tfm_rows = build_comparison_rows("TFM", ["FM", "TAP2.5D"])

    print_table("FM 相较于 TFM 和 TAP2.5D", fm_rows)
    print_table("TFM 相较于 FM 和 TAP2.5D", tfm_rows)


if __name__ == "__main__":
    main()
