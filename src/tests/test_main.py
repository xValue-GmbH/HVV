import pytest  # pytest is used for writing and running test cases
from fastapi.testclient import TestClient  # TestClient is used to simulate requests to the FastAPI application
from sqlalchemy import create_engine  # create_engine is used to create a connection to the database
from sqlalchemy.orm import sessionmaker  # sessionmaker is used to create a new session for interacting with the database
from app.main import app, get_db  # Importing the FastAPI app and the get_db dependency
from app.setup_database.models import Base, AirPollutionData  # Importing the database models
import os  # os is used for file operations like deleting the test database file

# Create a test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency override to use the test database
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)  # Create a TestClient instance for making requests to the FastAPI app
@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: Create the database tables
    Base.metadata.create_all(bind=engine)
    yield
    # Teardown: Drop the database tables and delete the test.db file
    Base.metadata.drop_all(bind=engine)
    engine.dispose()  # Ensure all connections are closed
    if os.path.exists("test.db"):
        os.remove("test.db")

def test_main_endpoint():
    response = client.get("/")  # Simulate a GET request to the root endpoint
    assert response.status_code == 200  # Check that the response status code is 200 (OK)
    assert "<h1>Welcome to Air Pollution Data Viewer</h1>" in response.text  # Check that the response contains the expected HTML content
    assert "<form action=\"/get_stats/\" method=\"post\">" in response.text
    assert "<label for=\"entity\">Select an entity:</label>" in response.text
    assert "<input type=\"submit\" value=\"Show Statistics\">" in response.text


def test_get_stats_no_year_range():
    response = client.post("/get_stats/", data={"entity": "Albania"})  # Simulate a POST request without a year range
    assert response.status_code == 200  # Check that the response status code is 200 (OK)
    assert "<p>Statistics for all parameters for Albania for all years:</p>" in response.text  # Check that the response contains the expected HTML content


def test_get_stats_valid_year_range():
    response = client.post("/get_stats/", data={"entity": "Albania", "start_year": 2000, "end_year": 2020})  # Simulate a POST request with a valid year range
    assert response.status_code == 200  # Check that the response status code is 200 (OK)
    assert "Year range must be between 1750 and 2022" not in response.text  # Ensure no error message is present


def test_get_stats_invalid_year_range():
    response = client.post("/get_stats/", data={"entity": "Albania", "start_year": 1700, "end_year": 2025})  # Simulate a POST request with an invalid year range
    assert response.status_code == 200  # Check that the response status code is 200 (OK)
    assert "<p>Year range must be between 1750 and 2022</p>" in response.text  # Check that the response contains the expected error message


def test_get_stats_exception_handling():
    response = client.post("/get_stats/", data={"entity": "Albania", "start_year": "invalid", "end_year": "invalid"})  # Simulate a POST request with invalid year values
    assert response.status_code == 422  # Check that the response status code is 422 (Unprocessable Entity)
    assert response.json() == {  # Check that the response contains the expected JSON error message
        "detail": [
            {
                "loc": ["body", "start_year"],
                "msg": "Input should be a valid integer, unable to parse string as an integer",
                "type": "int_parsing",
                "input": "invalid"
            },
            {
                "loc": ["body", "end_year"],
                "msg": "Input should be a valid integer, unable to parse string as an integer",
                "type": "int_parsing",
                "input": "invalid"
            }
        ]
    }

def test_get_stats_valid_data():
    # Add test data to the database
    db = TestingSessionLocal()
    db.add_all([
        AirPollutionData(entity="MyCountry", year=2000, nitrogen_oxide=10, sulphur_dioxide=20, carbon_monoxide=30,
                         organic_carbon=40, nmvoc=50, black_carbon=60, ammonia=70),
        AirPollutionData(entity="MyCountry", year=2001, nitrogen_oxide=15, sulphur_dioxide=25, carbon_monoxide=35,
                         organic_carbon=45, nmvoc=55, black_carbon=65, ammonia=75),
        AirPollutionData(entity="MyCountry", year=2002, nitrogen_oxide=20, sulphur_dioxide=30, carbon_monoxide=40,
                         organic_carbon=50, nmvoc=60, black_carbon=70, ammonia=80),
    ])
    db.commit()
    db.close()

    response = client.post("/get_stats/", data={"entity": "MyCountry", "start_year": 2000, "end_year": 2002})  # Simulate a POST request with valid data
    assert response.status_code == 200  # Check that the response status code is 200 (OK)
    assert "<p>Statistics for all parameters for MyCountry from 2000 to 2002:</p>" in response.text  # Check that the response contains the expected HTML content
    assert "<h3>Nitrogen Oxide (NOx)</h3>" in response.text
    assert "<li>Mean: 15.0</li>" in response.text
    assert "<li>Median: 15.0</li>" in response.text
    assert "<li>Standard Deviation: 5.0</li>" in response.text