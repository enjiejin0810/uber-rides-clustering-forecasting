"""Reproducible analysis for the Uber pickup clustering and forecasting project."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "uber-data.csv"
FIGURES = ROOT / "figures"
RANDOM_STATE = 42


def to_local_km(lat_lon):
    """Convert NYC lat/lon to a local equirectangular coordinate system in km."""
    lat0 = np.deg2rad(40.74)
    lat = np.asarray(lat_lon[:, 0], dtype=float)
    lon = np.asarray(lat_lon[:, 1], dtype=float)
    x = lon * (111.32 * np.cos(lat0))
    y = lat * 111.32
    return np.column_stack([x, y])


def from_local_km(xy):
    lat0 = np.deg2rad(40.74)
    xy = np.asarray(xy)
    lon = xy[:, 0] / (111.32 * np.cos(lat0))
    lat = xy[:, 1] / 111.32
    return np.column_stack([lat, lon])


def run():
    FIGURES.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    data["Date/Time"] = pd.to_datetime(data["Date/Time"], format="%m/%d/%Y %H:%M:%S")
    data["date"] = data["Date/Time"].dt.date
    data["hour"] = data["Date/Time"].dt.hour
    data["weekday"] = data["Date/Time"].dt.day_name()
    coords = data[["Lat", "Lon"]].to_numpy()
    xy = to_local_km(coords)

    # Model selection uses a reproducible spatial sample to keep silhouette computation feasible.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(data), size=min(25000, len(data)), replace=False)
    model_sample = xy[sample_idx]
    quality_idx = rng.choice(len(model_sample), size=min(4000, len(model_sample)), replace=False)
    quality_sample = model_sample[quality_idx]
    quality_rows = []
    for k in range(2, 11):
        candidate = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5,
                                    batch_size=2048, max_iter=100)
        labels = candidate.fit_predict(model_sample)
        qlabels = labels[quality_idx]
        quality_rows.append({
            "k": k,
            "silhouette": silhouette_score(quality_sample, qlabels),
            "calinski_harabasz": calinski_harabasz_score(model_sample, labels),
            "davies_bouldin": davies_bouldin_score(model_sample, labels),
            "inertia": candidate.inertia_,
        })
    quality = pd.DataFrame(quality_rows)
    # Combine independent index rankings so that one metric does not dominate the decision.
    # Silhouette and Calinski-Harabasz are maximized; Davies-Bouldin is minimized.
    quality["silhouette_rank"] = quality["silhouette"].rank(ascending=False, method="min")
    quality["calinski_harabasz_rank"] = quality["calinski_harabasz"].rank(ascending=False, method="min")
    quality["davies_bouldin_rank"] = quality["davies_bouldin"].rank(ascending=True, method="min")
    quality["rank_sum"] = quality[["silhouette_rank", "calinski_harabasz_rank", "davies_bouldin_rank"]].sum(axis=1)
    best_k = int(quality.loc[quality["rank_sum"].idxmin(), "k"])

    final_model = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=5,
                                  batch_size=8192, max_iter=150)
    data["cluster"] = final_model.fit_predict(xy)
    centers_latlon = from_local_km(final_model.cluster_centers_)
    centers = pd.DataFrame(centers_latlon, columns=["center_lat", "center_lon"])
    centers["cluster"] = np.arange(best_k)
    centers["trips"] = data["cluster"].value_counts().sort_index().to_numpy()
    centers["share_pct"] = centers["trips"] / len(data) * 100

    # 1. Model-selection diagnostics.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    axes[0].plot(quality.k, quality.silhouette, marker="o")
    axes[0].set(title="Silhouette (higher is better)", xlabel="Number of clusters (k)", ylabel="Score")
    axes[1].plot(quality.k, quality.calinski_harabasz, marker="o", color="#d95f02")
    axes[1].set(title="Calinski–Harabasz (higher is better)", xlabel="Number of clusters (k)", ylabel="Score")
    axes[2].plot(quality.k, quality.davies_bouldin, marker="o", color="#1b9e77")
    axes[2].set(title="Davies–Bouldin (lower is better)", xlabel="Number of clusters (k)", ylabel="Score")
    axes[3].plot(quality.k, quality.inertia, marker="o", color="#7570b3")
    axes[3].set(title="Inertia (lower; inspect elbow)", xlabel="Number of clusters (k)", ylabel="Within-cluster SSE")
    for ax in axes:
        ax.axvline(best_k, color="black", linestyle="--", alpha=.5)
    fig.tight_layout(); fig.savefig(FIGURES / "cluster_quality.png", dpi=180); plt.close(fig)

    # 2. Geographic distribution. Plot a random sample to keep the image readable.
    plot_idx = rng.choice(len(data), size=min(30000, len(data)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(data.loc[plot_idx, "Lon"], data.loc[plot_idx, "Lat"],
                    c=data.loc[plot_idx, "cluster"], s=3, alpha=.35, cmap="tab10", rasterized=True)
    ax.scatter(centers.center_lon, centers.center_lat, marker="X", s=180,
               c="black", edgecolor="white", linewidth=1.2, label="Cluster centers")
    for row in centers.itertuples():
        ax.annotate(str(row.cluster), (row.center_lon, row.center_lat), xytext=(5, 5),
                    textcoords="offset points", fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=.2", fc="white", alpha=.75, ec="none"))
    ax.set(title=f"Uber pickup locations grouped into {best_k} clusters (sample of 30,000 trips)",
           xlabel="Longitude", ylabel="Latitude")
    ax.legend(loc="best"); ax.grid(alpha=.15); fig.tight_layout()
    fig.savefig(FIGURES / "pickup_clusters.png", dpi=180); plt.close(fig)

    # 3. Hourly profile by cluster.
    hourly_profile = data.groupby(["cluster", "hour"]).size().unstack(0).reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in hourly_profile.columns:
        ax.plot(hourly_profile.index, hourly_profile[c], marker="o", markersize=3, label=f"Cluster {c}")
    ax.set(title="Hourly pickup counts by cluster (September 2014)", xlabel="Hour of day", ylabel="Trips")
    ax.set_xticks(range(0,24,2)); ax.legend(ncol=2); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(FIGURES / "hourly_demand_by_cluster.png", dpi=180); plt.close(fig)

    # 4. Daily trend by cluster.
    daily = data.groupby(["date", "cluster"]).size().unstack(1).sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in daily.columns:
        ax.plot(pd.to_datetime(daily.index), daily[c], marker="o", markersize=3, label=f"Cluster {c}")
    ax.set(title="Daily pickup counts by cluster (September 2014)", xlabel="Date", ylabel="Trips")
    ax.legend(ncol=2); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIGURES / "daily_demand_by_cluster.png", dpi=180); plt.close(fig)

    # Forecast: mean historical demand for the same weekday and hour, fit on Sep 1–23,
    # evaluated on Sep 24–30. This is a simple, interpretable benchmark for a short series.
    data["timestamp_hour"] = data["Date/Time"].dt.floor("h")
    observed = data.groupby(["timestamp_hour", "cluster"]).size().unstack("cluster", fill_value=0)
    full_index = pd.date_range("2014-09-01 00:00:00", "2014-09-30 22:00:00", freq="h")
    observed = observed.reindex(index=full_index, columns=range(best_k), fill_value=0)
    train = observed.loc[:"2014-09-23 23:00:00"]
    test = observed.loc["2014-09-24 00:00:00":"2014-09-30 22:00:00"]
    train_key = pd.MultiIndex.from_arrays([train.index.dayofweek, train.index.hour], names=["dow", "hour"])
    train_long = train.copy(); train_long.index = train_key
    profile = train_long.groupby(level=["dow", "hour"]).mean()
    test_key = pd.MultiIndex.from_arrays([test.index.dayofweek, test.index.hour], names=["dow", "hour"])
    forecast = pd.DataFrame(index=test.index, columns=range(best_k), dtype=float)
    for i, key in enumerate(test_key):
        forecast.iloc[i] = profile.loc[key].to_numpy()
    metrics = []
    for c in range(best_k):
        actual = test[c].to_numpy()
        pred = forecast[c].to_numpy()
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        wape = np.abs(actual - pred).sum() / max(actual.sum(), 1) * 100
        metrics.append({"cluster": c, "MAE": mae, "RMSE": rmse, "WAPE_pct": wape})
    forecast_metrics = pd.DataFrame(metrics)

    # Final 7-day forecast (Oct 1–7), using all observed September data for weekday-hour averages.
    full_long = observed.copy(); full_long.index = pd.MultiIndex.from_arrays(
        [observed.index.dayofweek, observed.index.hour], names=["dow", "hour"])
    final_profile = full_long.groupby(level=["dow", "hour"]).mean()
    future_index = pd.date_range("2014-10-01 00:00:00", periods=168, freq="h")
    future_keys = pd.MultiIndex.from_arrays([future_index.dayofweek, future_index.hour], names=["dow", "hour"])
    future = pd.DataFrame([final_profile.loc[key].to_numpy() for key in future_keys],
                          index=future_index, columns=range(best_k))
    future.to_csv(ROOT / "forecast_october_1_to_7.csv", index_label="timestamp")

    # Forecast plot for overall daily totals in the holdout week.
    fig, ax = plt.subplots(figsize=(12, 5))
    actual_total = test.sum(axis=1).resample("D").sum()
    forecast_total = forecast.sum(axis=1).resample("D").sum()
    ax.plot(actual_total.index, actual_total.values, marker="o", label="Observed")
    ax.plot(forecast_total.index, forecast_total.values, marker="o", label="Forecast")
    ax.set(title="Holdout-week demand forecast (Sep 24–30, 2014)", xlabel="Date", ylabel="Trips per day")
    ax.legend(); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIGURES / "forecast_holdout_week.png", dpi=180); plt.close(fig)

    summary = {
        "rows": int(len(data)),
        "start": str(data["Date/Time"].min()),
        "end": str(data["Date/Time"].max()),
        "duplicates_exact": int(pd.read_csv(DATA_PATH).duplicated().sum()),
        "missing_values": int(pd.read_csv(DATA_PATH).isna().sum().sum()),
        "best_k_by_sample_silhouette": best_k,
        "silhouette_sample": float(quality.loc[quality.k == best_k, "silhouette"].iloc[0]),
        "overall_hourly_peak": int(data["hour"].value_counts().idxmax()),
        "overall_daily_peak": str(data.groupby("date").size().idxmax()),
        "mean_holdout_mae_per_cluster": float(forecast_metrics.MAE.mean()),
        "mean_holdout_rmse_per_cluster": float(forecast_metrics.RMSE.mean()),
        "overall_holdout_wape_pct": float((np.abs(test.to_numpy()-forecast.to_numpy()).sum()/test.to_numpy().sum())*100),
    }
    quality.to_csv(ROOT / "cluster_quality_metrics.csv", index=False)
    centers.to_csv(ROOT / "cluster_centers.csv", index=False)
    forecast_metrics.to_csv(ROOT / "forecast_metrics.csv", index=False)
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nQUALITY TABLE\n", quality.round(4).to_string(index=False))
    print("\nCENTERS\n", centers.round(5).to_string(index=False))
    print("\nFORECAST METRICS\n", forecast_metrics.round(2).to_string(index=False))
    print("\nDAILY MEAN COUNT BY CLUSTER\n", daily.mean().round(0).to_dict())
    print("\nHourly counts by hour, peak:", data.hour.value_counts().sort_index().idxmax(),
          int(data.hour.value_counts().max()))
    print("\nPlots and tables written to", ROOT)


if __name__ == "__main__":
    run()
