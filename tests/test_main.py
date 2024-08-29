import pytest
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)  # Create a TestClient instance


# Define a list of dictionaries for different test cases
test_cases = [
    {"path": "/", "expected_status": 200, "expected_text": "Welcome to Air Pollution Data Viewer"},
]


@pytest.mark.parametrize("test_data", test_cases)
def test_read_main(test_data):
    response = client.get(test_data["path"])  # Assuming there's a root endpoint
    assert response.status_code == test_data["expected_status"]
    assert test_data["expected_text"] in response.text
