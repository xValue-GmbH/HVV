# OS module is used to handle file paths and directory operations.
import os

# Check if app is running within docker or directly, some directories need to be adressed differently
SECRET_KEY = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", "").lower() in ("yes", "y", "on", "true", "1")

# The atexit module is used to register functions to be called upon normal program termination.
import atexit

# The logging module is used to log messages for tracking events that happen when the software runs.
import logging

# The datetime module supplies classes for manipulating dates and times.
from datetime import datetime

# RotatingFileHandler is used to manage log files, allowing them to rotate when they reach a certain size.
from logging.handlers import RotatingFileHandler

# Import the Generator type from the typing module to specify the return type of generator functions
from typing import Generator

# Pandas is used for data manipulation and analysis.
import pandas as pd

# FastAPI is used to create the web application and handle HTTP requests.
from fastapi import Depends, FastAPI, Form, HTTPException

# HTMLResponse and RedirectResponse are used to return HTML content and handle redirects.
from fastapi.responses import HTMLResponse, RedirectResponse, Response

# StaticFiles enables access to css static file
from fastapi.staticfiles import StaticFiles

# Pydantic is used for data validation and settings management using Python type annotations.
from pydantic import BaseModel

# SQLAlchemy core is used to create the database engine and perform SQL queries.
from sqlalchemy import create_engine, func, select, text

# SQLAlchemy ORM is used to interact with the database in an object-oriented way.
from sqlalchemy.orm import Session, sessionmaker

# Importing the database models and session configuration.
from app.setup_database.models import AirPollutionData, Base

# Create a logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

# Create a file handler that logs messages to a file
if SECRET_KEY:
    log_file = "/app/src/app/logs/app.log"
else:
    log_file = os.path.join(
        os.path.dirname(__file__), "../..", "logs/app.log"
    )  # when executing the file directly, without docker

file_handler = RotatingFileHandler(log_file, maxBytes=2000, backupCount=5)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Initialize the FastAPI application.
app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Log application start
logger.info(f"Application started at {datetime.now()}")


# Register a function to log application end
def log_app_end() -> None:
    logger.info(f"Application ended at {datetime.now()}")


atexit.register(log_app_end)

# Define the path to the database, construct the database URL using the file path.
db_path = os.path.join(os.path.dirname(__file__), "airpollution.db")

# Define the database URL for SQLAlchemy.
DATABASE_URL = f"sqlite:///{db_path}"

# Create the database engine with SQLite.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Configure the session class for database interactions.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the database tables based on the models.
Base.metadata.create_all(bind=engine)


# Define a Pydantic model for input validation.
class AirPollutionDataCreate(BaseModel):
    entity: str
    year: int
    nitrogen_oxide: float
    sulphur_dioxide: float
    carbon_monoxide: float
    organic_carbon: float
    nmvoc: float
    black_carbon: float
    ammonia: float


def get_db() -> Generator[Session, None, None]:
    # Create a new database session.
    db = SessionLocal()

    try:
        yield db  # Yield the session for dependency injection.
    finally:
        db.close()  # Ensure the session is closed after use.


# Endpoint to display the main form.
@app.get("/", response_class=HTMLResponse)
async def main(db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        # Query distinct entities from the database.
        entities = db.execute(select(AirPollutionData.entity).distinct()).fetchall()
        # Generate HTML options for the entities.
        entity_options = "".join([f'<option value="{entity[0]}">{entity[0]}</option>' for entity in entities])
        # Generate the HTML content for the form.
        content = f"""
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="/static/styles.css">
        </head>
        <body>
            <div class="header">Welcome to Air Pollution Data Viewer</div>
            <div class="form-container">
                <h1>Select an entity and optionally a year range to view the summary statistics</h1>
                <form action="/get_stats/" method="post">
                    <label for="entity">Select an entity:</label>
                    <select name="entity" id="entity" required>
                        {entity_options}
                    </select>
                    <label for="start_year">Select start year (optional). Statistics is calculated including provided year. Minimum is 1750:</label>
                    <input type="number" name="start_year" id="start_year" min="1750" max="2022" step="1" pattern="\\d{4}">
                    <label for="end_year">Select end year (optional). Statistics is calculated including provided year. Maximum is 2022:</label>
                    <input type="number" name="end_year" id="end_year" min="1750" max="2022" step="1" pattern="\\d{4}">
                    <input type="submit" value="Show Statistics">
                </form>
            </div>
        </body>
        </html>
        """  # noqa: E501
        # Return the HTML content as a response.
        return HTMLResponse(content=content)

    except Exception as e:
        logger.error(f"Error in main endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Endpoint to handle form submission and redirect the appropriate statistics page
@app.post("/get_stats/", response_class=HTMLResponse)
async def get_stats(
    entity: str = Form(...), start_year: int = Form(None), end_year: int = Form(None), db: Session = Depends(get_db)
) -> Response:
    try:
        if start_year is None and end_year is not None:
            # Fetch the minimum and maximum available years from the database
            min_year_record = (
                db.query(AirPollutionData)
                .filter(AirPollutionData.entity == entity)
                .order_by(AirPollutionData.year)
                .first()
            )
            start_year = min_year_record.year

        elif end_year is None and start_year is not None:
            max_year_record = (
                db.query(AirPollutionData)
                .filter(AirPollutionData.entity == entity)
                .order_by(AirPollutionData.year.desc())
                .first()
            )
            end_year = max_year_record.year

        # Validate the year range
        if start_year and end_year:
            if start_year < 1750 or end_year > 2022:
                return HTMLResponse(content="<p>Year range must be between 1750 and 2022</p>")
            return RedirectResponse(url=f"/data/{entity}/{start_year}/{end_year}/stats", status_code=303)
        else:
            return RedirectResponse(url=f"/data/{entity}/all/stats", status_code=303)

    except Exception as e:
        logger.error(f"Error in get_stats endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Endpoint to get statistics for a specific entity and year range
@app.get("/data/{entity}/{start_year}/{end_year}/stats", response_class=HTMLResponse)
async def get_entity_stats(entity: str, start_year: int, end_year: int, db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        data = (
            db.query(AirPollutionData)
            .filter(
                AirPollutionData.entity == entity,
                AirPollutionData.year >= start_year,
                AirPollutionData.year <= end_year,
            )
            .all()
        )

        if not data:
            return HTMLResponse(content="<p>Data not found</p>")
        # In contrast to the calculation across all years, here,
        # we take the approach of calculating median/sd by pandas directly.
        # This shall demonstrate another option and work well with small datasets,
        # in contrast to the extended selection, assuming a large database in the real use case

        df = pd.DataFrame([d.__dict__ for d in data])
        # List of parameters to calculate statistics for
        parameters = [
            "nitrogen_oxide",
            "sulphur_dioxide",
            "carbon_monoxide",
            "organic_carbon",
            "nmvoc",
            "black_carbon",
            "ammonia",
        ]
        # Sort the parameters alphabetically
        parameters.sort()

        # Initialize a dictionary to store the statistics
        stats = {param: {"mean": None, "median": None, "stddev": None} for param in parameters}

        for parameter in parameters:
            mean = round(df[parameter].mean(), 3)
            median = round(df[parameter].median(), 3)
            stddev = round(df[parameter].std(), 3)
            stats[parameter]["mean"] = mean
            stats[parameter]["median"] = median
            stats[parameter]["stddev"] = stddev

        # Generate the HTML content for the statistics
        stats_html = "".join(
            [
                f"""
            <section class="stat-section">
                <h3>{'Non-methane Volatile Organic Compounds (NMVOC)' if param == 'nmvoc'
            else 'Nitrogen Oxide (NOx)' if param == 'nitrogen_oxide'
            else 'Carbon monoxide (CO)' if param == 'carbon_monoxide'
            else 'Sulphur dioxide (SO₂)' if param == 'sulphur_dioxide'
            else 'Ammonia (NH₃)' if param == 'ammonia'
            else param.replace('_', ' ').title()}</h3>
                <ul class="stat-list">
                    <li><strong>Mean:</strong> {stats[param]['mean']}</li>
                    <li><strong>Median:</strong> {stats[param]['median']}</li>
                    <li><strong>Standard Deviation:</strong> {stats[param]['stddev']}</li>
                </ul>
            </section>
            """
                for param in parameters
            ]
        )
        # Return the statistics as an HTML response.
        return HTMLResponse(
            content=f"""
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="/static/styles.css">
        </head>
        <body>
            <div class="header">Statistics for all parameters for {entity} from {start_year} to {end_year}:</div>
            <div class="container">
                {stats_html}
            </div>
        </body>
        </html>
        """
        )

    except Exception as e:
        logger.error(f"Error in get_stats endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Endpoint to get statistics for a specific entity for all years
@app.get("/data/{entity}/all/stats", response_class=HTMLResponse)
async def get_stats_all(entity: str, db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        # List of parameters to calculate statistics for
        parameters = [
            "nitrogen_oxide",
            "sulphur_dioxide",
            "carbon_monoxide",
            "organic_carbon",
            "nmvoc",
            "black_carbon",
            "ammonia",
        ]

        # Sort the parameters alphabetically
        parameters.sort()

        # Initialize a dictionary to store the statistics
        stats = {param: {"mean": None, "median": None, "stddev": None} for param in parameters}

        for parameter in parameters:
            # Calculate the mean for the parameter directly in the database, function avg available
            mean = (
                db.query(func.avg(getattr(AirPollutionData, parameter)))
                .filter(AirPollutionData.entity == entity)
                .scalar()
            )
            mean = round(mean, 3) if mean is not None else None
            # Custom SQL query to calculate the median
            median_query = text(
                f"""
            WITH ranked_data AS (
                SELECT {parameter},
                       ROW_NUMBER() OVER (ORDER BY {parameter}) AS row_num,
                       COUNT(*) OVER () AS total_rows
                FROM air_pollution_data
                WHERE entity = :entity
            )
            SELECT AVG({parameter}) AS median_value
            FROM ranked_data
            WHERE row_num IN (
                (total_rows + 1) / 2,
                (total_rows + 2) / 2
            )
            """
            )

            median_result = db.execute(median_query, {"entity": entity}).fetchone()
            median = round(median_result[0], 3) if median_result and median_result[0] is not None else None

            # Custom SQL query to calculate the standard deviation
            stddev_query = text(
                f"""
            WITH avg_data AS (
                SELECT avg({parameter}) as avg_{parameter}
                FROM air_pollution_data
                WHERE entity = :entity
            )
            SELECT sqrt(sum(power(t.{parameter} - avg_data.avg_{parameter}, 2)) / nullif(count(*) - 1, 0)) as stddev
            FROM air_pollution_data t
            JOIN avg_data ON 1=1
            WHERE t.entity = :entity
            """
            )

            stddev_result = db.execute(stddev_query, {"entity": entity}).fetchone()
            stddev = round(stddev_result[0], 3) if stddev_result and stddev_result[0] is not None else None

            # Store the calculated statistics in the dictionary
            stats[parameter]["mean"] = mean
            stats[parameter]["median"] = median
            stats[parameter]["stddev"] = stddev

        # Generate the HTML content for the statistics
        stats_html = "".join(
            [
                f"""
            <section class="stat-section">
                <h3>{'Non-methane Volatile Organic Compounds (NMVOC)' if param == 'nmvoc'
            else 'Nitrogen Oxide (NOx)' if param == 'nitrogen_oxide'
            else 'Carbon monoxide (CO)' if param == 'carbon_monoxide'
            else 'Sulphur dioxide (SO₂)' if param == 'sulphur_dioxide'
            else 'Ammonia (NH₃)' if param == 'ammonia' else param.replace('_', ' ').title()}</h3>
                <ul class="stat-list">
                    <li><strong>Mean:</strong>  {stats[param]['mean']}</li>
                    <li><strong>Median:</strong> {stats[param]['median']}</li>
                    <li><strong>Standard Deviation:</strong> {stats[param]['stddev']}</li>
                </ul>
            </section>
            """
                for param in parameters
            ]
        )

        # Return the statistics as an HTML response.
        return HTMLResponse(
            content=f"""
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="/static/styles.css">
        </head>
        <body>
            <div class="header">Statistics for all parameters for {entity} for all years:</div>
            <div class="container">
                {stats_html}
            </div>
        </body>
        </html>
        """
        )

    except Exception as e:
        logger.error(f"Error in get_stats_all endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Endpoint to add new air pollution data.
@app.post("/data")
async def create_data(data: AirPollutionDataCreate, db: Session = Depends(get_db)) -> Response:
    try:
        logger.info(f"Received data: {data}")
        # Create a new AirPollutionData instance from the input data.
        db_data = AirPollutionData(**data.model_dump())
        db.add(db_data)  # Add the new data to the session.
        db.commit()  # Commit the transaction to save the data.
        db.refresh(db_data)  # Refresh the instance to get the updated data.
        logger.info(f"Data added to DB: {data}")
        return db_data  # Return the newly created data.

    except Exception as e:
        logger.error(f"Error in create_data endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Endpoint to update existing air pollution data.
@app.put("/data/{entity}/{year}")
async def update_data(entity: str, year: int, data: AirPollutionDataCreate, db: Session = Depends(get_db)) -> Response:
    try:
        # Query the existing data by entity and year.
        db_data = (
            db.query(AirPollutionData).filter(AirPollutionData.entity == entity, AirPollutionData.year == year).first()
        )
        if not db_data:
            raise HTTPException(status_code=404, detail="Data not found")

        # Update the data with the new values.
        for key, value in data.model_dump().items():
            setattr(db_data, key, value)

        db.commit()  # Commit the transaction to save the changes.
        db.refresh(db_data)  # Refresh the instance to get the updated data.
        return db_data  # Return the updated data.

    except Exception as e:
        logger.error(f"Error in update_data endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Endpoint to delete existing air pollution data.
@app.delete("/data/{entity}/{year}", response_class=HTMLResponse)
async def delete_data(entity: str, year: int, db: Session = Depends(get_db)) -> HTMLResponse:
    try:
        logger.info(f"Attempting to delete data for entity: {entity}, year: {year}")
        # Query the database for the specific data point
        data_point = (
            db.query(AirPollutionData).filter(AirPollutionData.entity == entity, AirPollutionData.year == year).first()
        )

        # If the data point is not found, raise a 404 error
        if not data_point:
            logger.warning(f"Data point not found for entity: {entity}, year: {year}")
            raise HTTPException(status_code=404, detail="Data point not found")

        # Delete the data point from the database
        db.delete(data_point)
        db.commit()
        logger.info(f"Data point for entity: {entity}, year: {year} deleted successfully")

        # Return a success message
        return HTMLResponse(content=f"<p>Data point for {entity} in {year} has been deleted successfully.</p>")

    except HTTPException as http_exc:
        logger.error(f"HTTPException: {http_exc.detail}", exc_info=True)
        raise http_exc

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    # Uvicorn is used to run the FastAPI application.
    import uvicorn

    # Run the FastAPI application on host 0.0.0.0 and port 8000.
    uvicorn.run(app, host="0.0.0.0", port=8000)
