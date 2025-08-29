from flask import Flask, render_template, request, jsonify
import numpy as np
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import threading
import queue

app = Flask(__name__)

# Global variables to store models and data
models = {}
sample_data = None
scaler = StandardScaler()


def create_sample_data():
    """Create sample data for testing models"""
    global sample_data, scaler
    if sample_data is None:
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42,
        )
        # Scale the features
        X_scaled = scaler.fit_transform(X)
        sample_data = (X_scaled, y)
    return sample_data


def train_models():
    """Train different models for comparison"""
    global models
    X, y = create_sample_data()

    # Define models with different complexities
    model_configs = {
        "logistic_regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "description": "Linear model - fast inference, limited complexity",
            "color": "#FF6B6B",
        },
        "random_forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "description": "Ensemble model - moderate speed, good accuracy",
            "color": "#4ECDC4",
        },
        "svm_linear": {
            "model": SVC(kernel="linear", random_state=42),
            "description": "Linear SVM - moderate speed, good for linear separability",
            "color": "#45B7D1",
        },
        "svm_rbf": {
            "model": SVC(kernel="rbf", random_state=42),
            "description": "RBF SVM - slower, handles non-linear patterns",
            "color": "#96CEB4",
        },
        "neural_net_small": {
            "model": MLPClassifier(
                hidden_layer_sizes=(50,), max_iter=500, random_state=42
            ),
            "description": "Small neural network - moderate speed, some non-linearity",
            "color": "#FFEAA7",
        },
        "neural_net_large": {
            "model": MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            ),
            "description": "Large neural network - slower, high complexity",
            "color": "#DDA0DD",
        },
    }

    # Train all models
    for name, config in model_configs.items():
        print(f"Training {name}...")
        start_time = time.time()
        config["model"].fit(X, y)
        training_time = time.time() - start_time
        config["training_time"] = training_time
        models[name] = config

    print("All models trained successfully!")


def measure_inference_time(model, X, num_runs=100):
    """Measure inference time for a model"""
    times = []

    # Warm up the model
    for _ in range(10):
        model.predict(X[:10])

    # Measure inference times
    for _ in range(num_runs):
        start_time = time.time()
        model.predict(X)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "times": times,
    }


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/train_models", methods=["POST"])
def train_models_endpoint():
    """API endpoint to train models"""
    try:
        # Train models in a separate thread to avoid blocking
        thread = threading.Thread(target=train_models)
        thread.start()
        thread.join()

        return jsonify(
            {
                "status": "success",
                "message": "Models trained successfully!",
                "models": list(models.keys()),
            }
        )
    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Error training models: {str(e)}"}),
            500,
        )


@app.route("/profile_latency", methods=["POST"])
def profile_latency():
    """API endpoint to profile model latency"""
    try:
        data = request.get_json()
        selected_models = data.get("models", [])
        num_runs = data.get("num_runs", 100)

        if not models:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Models not trained yet. Please train models first.",
                    }
                ),
                400,
            )

        if not selected_models:
            return (
                jsonify(
                    {"status": "error", "message": "No models selected for profiling."}
                ),
                400,
            )

        X, _ = create_sample_data()
        results = {}

        for model_name in selected_models:
            if model_name in models:
                model = models[model_name]["model"]
                latency_stats = measure_inference_time(model, X, num_runs)
                results[model_name] = {
                    "stats": latency_stats,
                    "description": models[model_name]["description"],
                    "color": models[model_name]["color"],
                    "training_time": models[model_name]["training_time"],
                }

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Error profiling latency: {str(e)}"}
            ),
            500,
        )


@app.route("/get_models", methods=["GET"])
def get_models():
    """API endpoint to get available models"""
    if not models:
        return jsonify({"status": "error", "message": "Models not trained yet."}), 400

    model_info = {}
    for name, config in models.items():
        model_info[name] = {
            "description": config["description"],
            "color": config["color"],
            "training_time": config["training_time"],
        }

    return jsonify({"status": "success", "models": model_info})


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "models_trained": len(models) > 0,
            "num_models": len(models),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
