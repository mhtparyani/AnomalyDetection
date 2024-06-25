import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate synthetic data
data = {
    "heart_rate": np.random.normal(70, 5, 1000),
    "blood_pressure_systolic": np.random.normal(120, 10, 1000),
    "blood_pressure_diastolic": np.random.normal(80, 5, 1000),
    "oxygen_level": np.random.normal(98, 1, 1000)
}

# Introduce anomalies
data["heart_rate"][np.random.randint(0, 1000, 50)] = np.random.uniform(30, 120, 50)
data["blood_pressure_systolic"][np.random.randint(0, 1000, 50)] = np.random.uniform(90, 180, 50)
data["blood_pressure_diastolic"][np.random.randint(0, 1000, 50)] = np.random.uniform(50, 120, 50)
data["oxygen_level"][np.random.randint(0, 1000, 50)] = np.random.uniform(85, 100, 50)

# Convert to DataFrame
df = pd.DataFrame(data)
df.to_csv('health_data.csv', index=False)

# Plot the data
df.plot(subplots=True, figsize=(10, 8))
plt.show()
