# %% 1. Data and study design
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from threadpoolctl import threadpool_limits
threadpool_limits(limits=2)
ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
DATA = ROOT / 'uber-data.csv'
if not DATA.exists():
    raise FileNotFoundError('Place the instructor-provided uber-data.csv beside this script/notebook.')
df = pd.read_csv(DATA)
duplicates = int(df.duplicated().sum())
missing = df.isna().sum().to_dict()
df['time'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M:%S', errors='raise')
valid = df.Lat.between(-90,90) & df.Lon.between(-180,180)
if not valid.all() or df[['Lat','Lon']].isna().any().any():
    raise ValueError('Invalid or missing coordinates require review.')
cutoff = pd.Timestamp('2014-09-24')
training = df.time < cutoff
sx = 111.32 * np.cos(np.deg2rad(40.74))
X = np.c_[(df.Lon.to_numpy()+74)*sx, (df.Lat.to_numpy()-40.74)*111.32]
rng = np.random.default_rng(42)
print(f'{len(df):,} records; {df.time.min()} to {df.time.max()}.')
print(f'Missing values: {missing}; exact repeated rows retained: {duplicates:,}.')
print('Spatial model and demand profile trained on Sep 1–23; test period Sep 24–30 22:00.')

# %% 2. Choosing a spatial partition using training data only
train_ids = np.flatnonzero(training.to_numpy())
sample_ids = rng.choice(train_ids, min(25000,len(train_ids)), replace=False)
S = X[sample_ids]
quality_ids = rng.choice(len(S), min(4000,len(S)), replace=False)
models, rows = {}, []
for k in range(2,11):
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(S)
    labels = m.labels_
    models[k] = m
    rows.append(dict(k=k, silhouette=silhouette_score(S[quality_ids],labels[quality_ids]),
        calinski_harabasz=calinski_harabasz_score(S,labels),
        davies_bouldin=davies_bouldin_score(S,labels), inertia=m.inertia_))
quality = pd.DataFrame(rows)
quality['rank_sum'] = (quality.silhouette.rank(ascending=False)+
    quality.calinski_harabasz.rank(ascending=False)+quality.davies_bouldin.rank())
k = int(quality.loc[quality.rank_sum.idxmin(),'k'])
model = MiniBatchKMeans(n_clusters=k, init=models[k].cluster_centers_, n_init=1,
                       batch_size=8192, max_iter=150, random_state=42).fit(X[training])
# Fixed training-defined zones are used throughout, including the holdout and future forecasts.
df['cluster'] = model.predict(X)
final_labels = model.predict(S)
final_quality = dict(silhouette=silhouette_score(S[quality_ids], final_labels[quality_ids]),
    calinski_harabasz=calinski_harabasz_score(S,final_labels),
    davies_bouldin=davies_bouldin_score(S,final_labels))
quality.to_csv(ROOT/'cluster_quality_metrics.csv',index=False)
print(quality.round(4).to_string(index=False))
print('Selected k:',k,'; final MiniBatch model quality on training sample:',final_quality)
fig,axs=plt.subplots(2,2,figsize=(11,8))
for ax,col in zip(axs.flat,['silhouette','calinski_harabasz','davies_bouldin','inertia']):
    ax.plot(quality.k,quality[col],'-o'); ax.axvline(k,color='gray',ls='--')
    ax.set(title=col.replace('_',' ').title(),xlabel='k'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(FIG/'cluster_quality.png',dpi=150); plt.close(fig)

# %% 3. Geographic interpretation and fixed model centers
centers=pd.DataFrame({'cluster':range(k),'center_lat':model.cluster_centers_[:,1]/111.32+40.74,
                      'center_lon':model.cluster_centers_[:,0]/sx-74})
centers['trips']=df.cluster.value_counts().reindex(range(k),fill_value=0).to_numpy()
centers['share_pct']=100*centers.trips/len(df)
centers.to_csv(ROOT/'cluster_centers.csv',index=False)
print(centers.round(5).to_string(index=False))
colors=plt.get_cmap('tab10')
pidx=rng.choice(len(df),30000,replace=False)
fig,axs=plt.subplots(1,2,figsize=(13,6))
for ax in axs:
    for c in range(k):
        pts=df.iloc[pidx]; pts=pts[pts.cluster==c]
        ax.scatter(pts.Lon,pts.Lat,s=2,alpha=.3,color=colors(c),label=f'C{c}')
    ax.scatter(centers.center_lon,centers.center_lat,c='black',marker='x',s=55)
    for row in centers.itertuples(): ax.annotate(f'C{row.cluster}',(row.center_lon,row.center_lat),fontsize=8)
    ax.set(xlabel='Longitude',ylabel='Latitude'); ax.set_aspect(1/np.cos(np.deg2rad(40.74)))
axs[0].set_title('Full sampled extent (no distant points removed)')
axs[1].set(xlim=(-74.25,-73.7),ylim=(40.55,40.9),title='Central area detail')
axs[0].legend(markerscale=3,fontsize=7);fig.tight_layout();fig.savefig(FIG/'pickup_clusters.png',dpi=150);plt.close(fig)

# %% 4. Demand and empirical cluster centers by hour and date
# Do not refit separate clusters: label switching would make center comparisons ambiguous.
# These are subgroup mean locations within the fixed zones, not moving model centers.
df['hour']=df.time.dt.hour; df['date']=df.time.dt.floor('D')
observed_index=pd.date_range(df.time.min().floor('h'),df.time.max().floor('h'),freq='h')
counts=(df.groupby([df.time.dt.floor('h'),'cluster']).size().unstack(fill_value=0)
        .reindex(index=observed_index,columns=range(k),fill_value=0))
exposure=pd.Series(observed_index.hour).value_counts().reindex(range(24))
hourly=df.groupby(['hour','cluster']).size().unstack(fill_value=0).reindex(range(24)).div(exposure,axis=0)
daily=counts.resample('D').sum()
for name,table,xlabel in [('hourly_demand_by_cluster',hourly,'Hour'),('daily_demand_by_cluster',daily,'Date')]:
    fig,ax=plt.subplots(figsize=(11,5))
    table.plot(ax=ax,color=[colors(c) for c in range(k)])
    ax.set(xlabel=xlabel,ylabel='Mean trips per observed hour' if xlabel=='Hour' else 'Trips',title=name.replace('_',' ').title())
    ax.legend(ncol=4,title='Cluster');fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=150);plt.close(fig)
profiles={}
for field in ['hour','date']:
    tab=df.groupby(['cluster',field]).agg(n=('Lat','size'),lat=('Lat','mean'),lon=('Lon','mean')).reset_index()
    tab=tab.merge(centers[['cluster','center_lat','center_lon']],on='cluster')
    tab['east_km']=(tab.lon-tab.center_lon)*sx
    tab['north_km']=(tab.lat-tab.center_lat)*111.32
    tab['displacement_km']=np.hypot(tab.east_km,tab.north_km)
    tab.to_csv(ROOT/f'cluster_centers_by_{field}.csv',index=False)
    profiles[field]=tab
    fig,axs=plt.subplots((k+1)//2,2,figsize=(12,3*((k+1)//2)),squeeze=False)
    for c,ax in enumerate(axs.flat):
        if c>=k: ax.set_visible(False);continue
        s=tab[tab.cluster==c]
        ax.plot(s[field],s.east_km,label='East–west');ax.plot(s[field],s.north_km,label='North–south')
        ax.axhline(0,color='gray',lw=.7);ax.set(title=f'C{c}: subgroup mean offset',ylabel='km from training center')
        ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(FIG/f'centers_by_{field}.png',dpi=150);plt.close(fig)
zone_rows=[]
for c in range(k):
    h=profiles['hour'].query('cluster == @c'); d=profiles['date'].query('cluster == @c')
    zone_rows.append(dict(cluster=c,peak_hour=int(hourly[c].idxmax()),peak_date=str(daily[c].idxmax().date()),
        max_hour_center_offset_km=h.displacement_km.max(),max_date_center_offset_km=d.displacement_km.max(),
        min_hour_group_n=int(h.n.min()),min_date_group_n=int(d.n.min())))
zone_summary=pd.DataFrame(zone_rows)
zone_summary.to_csv(ROOT/'temporal_cluster_summary.csv',index=False)
print(zone_summary.round(3).to_string(index=False))
print('Sep 30 has 23 observed hourly bins; 23:00 is unobserved, not assumed zero. Hourly profiles use exposure adjustment.')

# %% 5. Chronological forecasts and a weekly-naive comparison
train=counts[counts.index<cutoff]; test=counts[counts.index>=cutoff]
def weekday_hour_prediction(history,index):
    means=history.groupby([history.index.dayofweek,history.index.hour]).mean()
    keys=pd.MultiIndex.from_arrays([index.dayofweek,index.hour])
    return pd.DataFrame(means.reindex(keys).to_numpy(),index=index,columns=history.columns)
pred=weekday_hour_prediction(train,test.index)
weekly=train.reindex(test.index-pd.Timedelta(days=7)).copy();weekly.index=test.index
assert not pred.isna().any().any() and not weekly.isna().any().any()
metric_rows=[]
for name,forecast in [('weekday_hour_mean',pred),('weekly_naive',weekly)]:
    for c in list(range(k))+['ALL']:
        a=test.to_numpy() if c=='ALL' else test[c].to_numpy()
        b=forecast.to_numpy() if c=='ALL' else forecast[c].to_numpy()
        e=a-b
        metric_rows.append(dict(model=name,cluster=c,MAE=np.abs(e).mean(),RMSE=np.sqrt((e*e).mean()),WAPE_pct=100*np.abs(e).sum()/a.sum()))
metrics=pd.DataFrame(metric_rows);metrics.to_csv(ROOT/'forecast_metrics.csv',index=False)
print(metrics.round(3).to_string(index=False))
future_index=pd.date_range('2014-10-01',periods=168,freq='h')
future=weekday_hour_prediction(counts,future_index)
future.to_csv(ROOT/'forecast_october_1_to_7.csv',index_label='timestamp')
fig,axs=plt.subplots((k+1)//2,2,figsize=(13,3*((k+1)//2)),squeeze=False)
for c,ax in enumerate(axs.flat):
    if c>=k: ax.set_visible(False); continue
    ax.plot(test.index,test[c],label='Observed',lw=1)
    ax.plot(pred.index,pred[c],label='Weekday/hour mean',lw=.9)
    ax.set(title=f'C{c}: hourly holdout demand',ylabel='Trips');ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(FIG/'forecast_holdout_week.png',dpi=150);plt.close(fig)
summary=dict(rows=len(df),selected_k_by_combined_rank=k,final_training_sample_quality=final_quality,
    cutoff=str(cutoff),holdout_hours=len(test),duplicates_retained=duplicates,
    forecast_wape_pct=float(metrics.query("model == 'weekday_hour_mean' and cluster == 'ALL'").WAPE_pct.iloc[0]),
    weekly_naive_wape_pct=float(metrics.query("model == 'weekly_naive' and cluster == 'ALL'").WAPE_pct.iloc[0]))
(ROOT/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
