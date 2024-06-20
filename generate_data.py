import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000
n_features = 2

data = np.random.normal(0, 1, size=(n_samples, n_features))
n_anomalies = int(0.05 * n_samples)
anomalies = np.random.normal(0, 10, size=(n_anomalies, n_features))
data_with_anomalies = np.vstack([data, anomalies])
df = pd.DataFrame(data_with_anomalies, columns=['feature1', 'feature2'])

plt.scatter(df['feature1'], df['feature2'], label='Normal data')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Anomalies')
plt.legend()
plt.show()

df.to_csv('health_data.csv', index=False)
