"""
Views for the analytics dashboard.
All data is loaded directly from CSV/JSON files in the capstone_2026 directory.
"""
import csv
import json
from pathlib import Path

from django.conf import settings
from django.shortcuts import render


EDA_DIR = settings.REPORTS_DIR / 'eda'
SLICES_DIR = settings.REPORTS_DIR / 'slices'


def _read_csv(filepath):
    """Read a CSV file and return list of dicts."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_json(filepath):
    """Read a JSON file and return parsed data."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────
# 1. Overview
# ─────────────────────────────────────────────────

def overview(request):
    # Dataset overview
    ds = _read_csv(EDA_DIR / 'dataset_overview.csv')
    ds_info = ds[0] if ds else {}

    # Fare components
    fare_comp = _read_csv(EDA_DIR / 'fare_components.csv')
    comp_labels = [r['component'].replace('_', ' ').title() for r in fare_comp]
    comp_values = [round(float(r['average_amount']), 2) for r in fare_comp]

    # Numeric summary
    num_summary = _read_csv(EDA_DIR / 'numeric_summary_sampled.csv')

    # Platform fares
    platform = _read_csv(EDA_DIR / 'platform_fares.csv')
    platform_map = {
        'HV0002': 'Juno',
        'HV0003': 'Uber',
        'HV0004': 'Via',
        'HV0005': 'Lyft',
    }
    for p in platform:
        p['platform_name'] = platform_map.get(p.get('hvfhs_license_num', ''), p.get('hvfhs_license_num', ''))
        p['mean'] = round(float(p.get('mean', 0)), 2)
        p['count'] = int(float(p.get('count', 0)))

    # Trip length
    trip_len = _read_csv(EDA_DIR / 'fare_by_trip_length.csv')
    for t in trip_len:
        t['mean'] = round(float(t.get('mean', 0)), 2)
        t['count'] = int(float(t.get('count', 0)))

    context = {
        'ds_info': ds_info,
        'comp_labels_json': json.dumps(comp_labels),
        'comp_values_json': json.dumps(comp_values),
        'num_summary': num_summary,
        'platform': platform,
        'trip_len': trip_len,
    }
    return render(request, 'analytics/overview.html', context)


# ─────────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────────

def eda(request):
    # Fare by hour
    hourly = _read_csv(EDA_DIR / 'fare_by_hour.csv')
    hour_labels = [r['hour'] for r in hourly]
    hour_means = [round(float(r['mean']), 2) for r in hourly]
    hour_stds = [round(float(r['std']), 2) for r in hourly]

    # Fare by weekday
    weekday = _read_csv(EDA_DIR / 'fare_by_weekday_sampled.csv')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    wd_map = {r['weekday']: round(float(r['base_passenger_fare']), 2) for r in weekday}
    wd_labels = day_order
    wd_values = [wd_map.get(d, 0) for d in day_order]

    # Fare by month
    monthly = _read_csv(EDA_DIR / 'fare_by_month.csv')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
    month_values = [round(float(r['base_passenger_fare']), 2) for r in monthly]

    # Platform fares
    platform = _read_csv(EDA_DIR / 'platform_fares.csv')
    platform_map = {'HV0002': 'Juno', 'HV0003': 'Uber', 'HV0004': 'Via', 'HV0005': 'Lyft'}
    plat_labels = [platform_map.get(r['hvfhs_license_num'], r['hvfhs_license_num']) for r in platform]
    plat_values = [round(float(r['mean']), 2) for r in platform]

    # Trip length
    trip_len = _read_csv(EDA_DIR / 'fare_by_trip_length.csv')
    tl_labels = [r['trip_length_bucket'].title() for r in trip_len]
    tl_means = [round(float(r['mean']), 2) for r in trip_len]
    tl_stds = [round(float(r['std']), 2) for r in trip_len]

    context = {
        'hour_labels_json': json.dumps(hour_labels),
        'hour_means_json': json.dumps(hour_means),
        'hour_stds_json': json.dumps(hour_stds),
        'wd_labels_json': json.dumps(wd_labels),
        'wd_values_json': json.dumps(wd_values),
        'month_labels_json': json.dumps(month_names),
        'month_values_json': json.dumps(month_values),
        'plat_labels_json': json.dumps(plat_labels),
        'plat_values_json': json.dumps(plat_values),
        'tl_labels_json': json.dumps(tl_labels),
        'tl_means_json': json.dumps(tl_means),
        'tl_stds_json': json.dumps(tl_stds),
    }
    return render(request, 'analytics/eda.html', context)


# ─────────────────────────────────────────────────
# 3. Model Comparison
# ─────────────────────────────────────────────────

def model_comparison(request):
    metrics_file = SLICES_DIR / 'overall_metrics_2021-10.csv'
    rows = _read_csv(metrics_file)

    model_names_map = {
        'baseline_hgb': 'HistGradientBoosting',
        'model_xgb': 'XGBoost',
        'model_log': 'Log-Transform HGB',
        'model_gbr': 'GradientBoosting',
        'model_quantile': 'Quantile HGB',
        'model_rf': 'RandomForest',
        'linear_regression': 'Linear Regression',
    }

    models_data = []
    for r in rows:
        if not r.get('model_name'):
            continue
        models_data.append({
            'name': model_names_map.get(r['model_name'], r['model_name']),
            'raw_name': r['model_name'],
            'mae': round(float(r['mae']), 3),
            'rmse': round(float(r['rmse']), 3),
            'r2': round(float(r['r2']), 4),
            'p90': round(float(r['p90_abs_error']), 3),
            'n': int(float(r['n'])),
        })

    # Sort by RMSE ascending (best first)
    models_data.sort(key=lambda x: x['rmse'])

    chart_labels = [m['name'] for m in models_data]
    chart_mae = [m['mae'] for m in models_data]
    chart_rmse = [m['rmse'] for m in models_data]
    chart_r2 = [m['r2'] for m in models_data]

    # Best model summary
    summary_file = SLICES_DIR / 'overall_summary_2021-10.json'
    summary = _read_json(summary_file)

    context = {
        'models_data': models_data,
        'chart_labels_json': json.dumps(chart_labels),
        'chart_mae_json': json.dumps(chart_mae),
        'chart_rmse_json': json.dumps(chart_rmse),
        'chart_r2_json': json.dumps(chart_r2),
        'best_rmse_model': model_names_map.get(
            summary.get('best_rmse', {}).get('model_name', ''), ''
        ),
        'best_rmse_val': round(summary.get('best_rmse', {}).get('rmse', 0), 3),
        'best_mae_model': model_names_map.get(
            summary.get('best_mae', {}).get('model_name', ''), ''
        ),
        'best_mae_val': round(summary.get('best_mae', {}).get('mae', 0), 3),
        'best_r2_model': model_names_map.get(
            summary.get('best_r2', {}).get('model_name', ''), ''
        ),
        'best_r2_val': round(summary.get('best_r2', {}).get('r2', 0), 4),
    }
    return render(request, 'analytics/models.html', context)


# ─────────────────────────────────────────────────
# 4. Volatility
# ─────────────────────────────────────────────────

def volatility(request):
    # Hourly fare volatility (CV time series) — sample every 24 hours for readability
    vol_data = _read_csv(EDA_DIR / 'hourly_fare_volatility.csv')

    # Group by date, compute daily average CV
    daily_cv = {}
    for r in vol_data:
        date_str = r['hour'][:10]  # 'YYYY-MM-DD'
        cv = float(r['cv'])
        if date_str not in daily_cv:
            daily_cv[date_str] = []
        daily_cv[date_str].append(cv)

    sorted_dates = sorted(daily_cv.keys())
    daily_labels = sorted_dates
    daily_values = [round(sum(daily_cv[d]) / len(daily_cv[d]), 4) for d in sorted_dates]

    # Zone average fares — top 20 and bottom 20
    zone_data = _read_csv(EDA_DIR / 'zone_average_fares.csv')
    zone_parsed = [(int(r['PULocationID']), round(float(r['base_passenger_fare']), 2)) for r in zone_data]
    zone_parsed.sort(key=lambda x: x[1], reverse=True)

    top20 = zone_parsed[:20]
    bottom20 = zone_parsed[-20:]

    top_labels = [f'Zone {z[0]}' for z in top20]
    top_values = [z[1] for z in top20]
    bottom_labels = [f'Zone {z[0]}' for z in bottom20]
    bottom_values = [z[1] for z in bottom20]

    # Compute overall volatility stats
    all_cvs = [float(r['cv']) for r in vol_data]
    avg_cv = round(sum(all_cvs) / len(all_cvs), 4) if all_cvs else 0
    max_cv = round(max(all_cvs), 4) if all_cvs else 0
    min_cv = round(min(all_cvs), 4) if all_cvs else 0

    context = {
        'daily_labels_json': json.dumps(daily_labels),
        'daily_values_json': json.dumps(daily_values),
        'top_labels_json': json.dumps(top_labels),
        'top_values_json': json.dumps(top_values),
        'bottom_labels_json': json.dumps(bottom_labels),
        'bottom_values_json': json.dumps(bottom_values),
        'avg_cv': avg_cv,
        'max_cv': max_cv,
        'min_cv': min_cv,
        'num_hours': len(vol_data),
    }
    return render(request, 'analytics/volatility.html', context)


# ─────────────────────────────────────────────────
# 5. Fairness
# ─────────────────────────────────────────────────

def fairness(request):
    model_names_map = {
        'baseline_hgb': 'HistGradientBoosting',
        'model_xgb': 'XGBoost',
        'model_log': 'Log-Transform HGB',
        'model_gbr': 'GradientBoosting',
        'model_quantile': 'Quantile HGB',
        'model_rf': 'RandomForest',
        'linear_regression': 'Linear Regression',
    }

    # Collect heteroscedasticity data for all models
    hetero_files = sorted(SLICES_DIR.glob('hetero_resid_by_decile_*_2021-10.csv'))
    models_hetero = []

    for fp in hetero_files:
        fname = fp.stem  # e.g. hetero_resid_by_decile_model_xgb_2021-10
        # Extract model key
        prefix = 'hetero_resid_by_decile_'
        suffix = '_2021-10'
        model_key = fname[len(prefix):-len(suffix)] if fname.startswith(prefix) and fname.endswith(suffix) else fname

        rows = _read_csv(fp)
        decile_labels = [r['pred_decile'] for r in rows]
        resid_vars = [round(float(r['resid_var']), 2) for r in rows]
        resid_stds = [round(float(r['resid_std']), 2) for r in rows]

        models_hetero.append({
            'name': model_names_map.get(model_key, model_key),
            'key': model_key,
            'decile_labels': decile_labels,
            'resid_vars': resid_vars,
            'resid_stds': resid_stds,
        })

    # Decile labels (same for all models)
    decile_labels = models_hetero[0]['decile_labels'] if models_hetero else []

    # Zone fare disparity
    zone_data = _read_csv(EDA_DIR / 'zone_average_fares.csv')
    fares = [float(r['base_passenger_fare']) for r in zone_data]
    fare_min = round(min(fares), 2) if fares else 0
    fare_max = round(max(fares), 2) if fares else 0
    fare_mean = round(sum(fares) / len(fares), 2) if fares else 0
    fare_range = round(fare_max - fare_min, 2)

    # Gini-like disparity metric
    sorted_fares = sorted(fares)
    n = len(sorted_fares)
    if n > 0 and fare_mean > 0:
        gini = sum((2 * (i + 1) - n - 1) * f for i, f in enumerate(sorted_fares)) / (n * n * fare_mean)
        gini = round(gini, 4)
    else:
        gini = 0

    # Build chart data for all models stacked
    chart_datasets = []
    colors = [
        'rgba(99, 102, 241, 0.8)',
        'rgba(6, 182, 212, 0.8)',
        'rgba(249, 115, 22, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(244, 63, 94, 0.8)',
        'rgba(168, 85, 247, 0.8)',
        'rgba(234, 179, 8, 0.8)',
    ]

    for i, m in enumerate(models_hetero):
        chart_datasets.append({
            'label': m['name'],
            'data': m['resid_vars'],
            'backgroundColor': colors[i % len(colors)],
            'borderColor': colors[i % len(colors)].replace('0.8', '1'),
            'borderWidth': 1,
        })

    context = {
        'decile_labels_json': json.dumps(decile_labels),
        'chart_datasets_json': json.dumps(chart_datasets),
        'models_hetero': models_hetero,
        'fare_min': fare_min,
        'fare_max': fare_max,
        'fare_mean': fare_mean,
        'fare_range': fare_range,
        'gini': gini,
        'num_zones': n,
    }
    return render(request, 'analytics/fairness.html', context)
