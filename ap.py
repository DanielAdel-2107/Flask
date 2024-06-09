import requests # type: ignore
import json
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Define the URL of the Flask server
url = 'http://127.0.0.1:5000/predict'

# Define the data to be sent in the request
data = {
    'model': 'linear',  # Specify the model type ('linear', 'lasso', 'ridge', etc.)
    'features': [4.0, 4.5, 5420.0, 101930.0, 1, 0, 0, 3, 11, 3890.0, 2001, 98053.0, 47.6561, -122.205, 4760.0, 101930.0]
}

# Convert data to JSON format
json_data = json.dumps(data)


# Set the headers for the request
headers = {'Content-Type': 'application/json'}

# Send the POST request to the Flask server
response = requests.post(url, data=json_data, headers=headers)

# Print the response
print(response.json())
