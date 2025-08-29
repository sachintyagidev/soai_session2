import pytest
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "models_trained" in data
    assert "num_models" in data


def test_index_page(client):
    """Test the main page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Model Latency Profiler" in response.data


def test_train_models_endpoint(client):
    """Test the model training endpoint."""
    response = client.post("/train_models")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "models" in data


def test_get_models_endpoint(client):
    """Test getting available models."""
    # First train models
    client.post("/train_models")

    # Then get models
    response = client.get("/get_models")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "models" in data


def test_profile_latency_endpoint(client):
    """Test the latency profiling endpoint."""
    # First train models
    client.post("/train_models")

    # Then profile latency
    response = client.post(
        "/profile_latency", json={"models": ["logistic_regression"], "num_runs": 10}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "results" in data


def test_profile_latency_no_models(client):
    """Test profiling with no models selected."""
    response = client.post("/profile_latency", json={"models": [], "num_runs": 10})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_profile_latency_models_not_trained(client):
    """Test profiling when models aren't trained."""
    # Reset the global models variable to simulate untrained state
    import app

    app.models = {}

    response = client.post(
        "/profile_latency", json={"models": ["logistic_regression"], "num_runs": 10}
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
