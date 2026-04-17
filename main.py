import sqlite3

from config import settings
from data import SQLRepository
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import GarchModel
from pydantic import BaseModel


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#`FitIn` class
class FitIn(BaseModel):
    ticker: str
    use_new_data: bool
    n_observations: int
    p: int
    q: int


#`FitOut` class
class FitOut(FitIn):
    success: bool
    message: str


#`PredictIn` class
class PredictIn(BaseModel):
    ticker: str
    n_days: int


# `PredictOut` class
class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


def build_model(ticker, use_new_data):
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)
    repo = SQLRepository(connection=connection)
    model = GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)
    return model


# `"/hello" path with 200 status code
@app.get("/hello", status_code=200)
def hello():
    """Return dictionary with greeting message."""
    return {"message":"Hello world"}


# `"/fit" path, 200 status code
@app.post("/fit", response_model=FitOut)
def fit_model(request: FitIn):
    """Fit model, return confirmation message.

    Parameters
    ----------
    request : FitIn

    Returns
    ------
    dict
        Must conform to `FitOut` class
    """
    # Create `response` dictionary from `request`
    response = request.dict()

    # Create try block to handle exceptions
    try:
    # Build model
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
    
        # Wrangle data
        model.wrangle_data(n_observations=request.n_observations)
    
        # Fit model
        model.fit(p=request.p, q=request.q)
    
        # Save model
        filepath = model.dump()
    
        # Success response
        response["success"] = True
    
        # Updated message with AIC & BIC
        response["message"] = (
            f"Trained and saved '{filepath}'. "
            f"Metrics: AIC {model.aic}, BIC {model.bic}."
        )

    except Exception as e:
        response["success"] = False

        response["message"] = str(e)

    # Return response

    return response


# `"/predict" path, 200 status code
@app.post('/predict', status_code = 200, response_model=PredictOut)
def get_prediction(request: PredictIn):
    # Create `response` dictionary from `request`
    response = request.dict()
    # Create try block to handle exceptions
    try:
        # Build model with `build_model` function
        model = build_model(ticker=request.ticker, use_new_data=False)
        # Load stored model
        model.load()
        # Generate prediction
        prediction = model.predict_volatility(horizon=request.n_days)

        response["success"] = True

        response["forecast"] = prediction

        response["message"] = ""

    # Create except block
    except Exception as e:
        response["success"] = False

        response["forecast"] = {}

        response["message"] = str(e)

    return response
