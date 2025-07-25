import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        agg = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'count', 'std']
        })
        agg.columns = ['total_amount', 'avg_amount', 'txn_count', 'std_amount']
        agg.reset_index(inplace=True)
        return agg

class DateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['txn_hour'] = X['TransactionStartTime'].dt.hour
        X['txn_day'] = X['TransactionStartTime'].dt.day
        X['txn_month'] = X['TransactionStartTime'].dt.month
        X['txn_year'] = X['TransactionStartTime'].dt.year
        return X

def build_pipeline():
    num_features = ['Amount', 'Value']  # purely numerical
    cat_features = ['ProductCategory', 'ChannelId', 'CountryCode']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    pipeline = Pipeline([
        ('date_features', DateFeatures()),
        ('preprocessor', preprocessor)
    ])
    return pipeline

def compute_rfm(df, snapshot_date):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']).dt.tz_localize(None)
    snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def assign_high_risk(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm['cluster'] = clusters
    
    # Compute average frequency+monetary per cluster
    cluster_stats = rfm.groupby('cluster')[['Frequency', 'Monetary']].mean()
    cluster_stats['score'] = cluster_stats['Frequency'] + cluster_stats['Monetary']
    high_risk_cluster = cluster_stats['score'].idxmin()  # lowest combined score
    
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    return rfm[['is_high_risk', 'cluster']]  # Keep cluster for analysis
