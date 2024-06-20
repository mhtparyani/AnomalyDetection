from flask import Flask, jsonify, request
import torch
import pandas as pd
from train_model import Autoencoder  # Ensure this import is correct


app = Flask(__name__)

# Load model
model = Autoencoder(input_dim=2, hidden_dim=64)  # Adjust input_dim and hidden_dim accordingly
model.load_state_dict(torch.load('autoencoder.pth', map_location=torch.device('cpu')))
model.eval()


@app.route('/anomalies', methods=['GET'])
def get_anomalies():
    data = pd.read_csv('health_data.csv')
    data_tensor = torch.tensor(data.values, dtype=torch.float32)

    with torch.no_grad():
        reconstructions = model(data_tensor)
        reconstruction_errors = torch.mean((reconstructions - data_tensor) ** 2, dim=1)

    threshold = torch.mean(reconstruction_errors) + 3 * torch.std(reconstruction_errors)
    anomalies = reconstruction_errors > threshold

    normal_data = data[~anomalies].values.tolist()
    anomaly_data = data[anomalies].values.tolist()

    return jsonify({'normal_data': normal_data, 'anomalies': anomaly_data})


if __name__ == '__main__':
    app.run(debug=True)
