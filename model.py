import os
from glob import glob
import warnings

import joblib
import pandas as pd
from arch import arch_model
from config import settings
from data import AlphaVantageAPI, SQLRepository
from statsmodels.tsa.arima.model import ARIMA


class GarchModel:
    """Class for training GARCH model and generating predictions.

    Atttributes
    -----------
    ticker : str
        Ticker symbol of the equity whose volatility will be predicted.
    repo : SQLRepository
        The repository where the training data will be stored.
    use_new_data : bool
        Whether to download new data from the AlphaVantage API to trsain
        the model or to use the existing data stored in the repository.
    model_directory : str
        Path for directory where trained models will be stored.

    Methods
    -------
    wrangle_data
        Generate equity returns from data in database.
    fit
        Fit model to training data.
    predict
        Generate volatilty forecast from trained model.
    dump
        Save trained model to file.
    load
        Load trained model from file.
    """

    def __init__(self, ticker, repo, use_new_data):
        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory

    def wrangle_data(self, n_observations):
        """Extract data from database (or get from AlphaVantage), transform it
        for training model, and attach it to `self.data`.

        Parameters
        ----------
        n_observations : int
            Number of observations to retrieve from database

        Returns
        -------
        None
        """
        # Add new data to database if required
        if self.use_new_data:
            api = AlphaVantageAPI()
            new_data = api.get_daily(ticker=self.ticker)
            self.repo.insert_table(
                table_name=self.ticker, records=new_data, if_exists="replace"
            )
        # Pull data from SQL database
        df = self.repo.read_table(table_name=self.ticker, limit=n_observations+1)

        # Clean data, attach to class as `data` attribute
        df.sort_index(ascending=True, inplace=True)
        df['return'] = df['close'].pct_change() * 100
        self.data = df['return'].dropna()

    def fit(self, p, q):
        """Create model, fit to `self.data`, and attach to `self.model` attribute.
        For assignment, also assigns adds metrics to `self.aic` and `self.bic`.

        Parameters
        ----------
        p : int
            Lag order of the symmetric innovation

        q : ind
            Lag order of lagged volatility

        Returns
        -------
        None
        """
        # Train Model, attach to `self.model`
        self.model = arch_model(self.data, p=p, q=q, rescale=False).fit(disp='off')

        self.aic = self.model.aic
        self.bic = self.model.bic
        

    def __clean_prediction(self, prediction):
    
        """Reformat model prediction to JSON.

        Parameters
        ----------
        prediction : pd.DataFrame
            Variance from a `ARCHModelForecast`

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Calculate forecast start date
        start = prediction.index[0] + pd.DateOffset(days=1)

        # Create date range
        prediction_dates = pd.bdate_range(start=start, periods=prediction.shape[1])
    
        # Create prediction index labels, ISO 8601 format
        prediction_index = [d.isoformat() for d in prediction_dates]
    
        # Extract predictions from DataFrame, get square root
        data = prediction.values.flatten() ** 0.5
    
        # Combine `data` and `prediction_index` into Series
        prediction_formatted = pd.Series(data, index=prediction_index)
    
        # Return Series as dictionary
        return prediction_formatted.to_dict()

    def predict_volatility(self, horizon):
        """Predict volatility using `self.model`

        Parameters
        ----------
        horizon : int
            Horizon of forecast, by default 5.

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Generate variance forecast from `self.model`
        prediction = self.model.forecast(horizon=horizon, reindex=False).variance

        # Format prediction with `self.__clean_predction`
        prediction_formatted = self.__clean_prediction(prediction)

        # Return `prediction_formatted
        return prediction_formatted

    def dump(self):
        """Save model to `self.model_directory` with timestamp.

        Returns
        -------
        str
            filepath where model was saved.
        """
        # Create timestamp in ISO format
        timestamp = pd.Timestamp.now().isoformat().replace(":", "-")
        filepath = os.path.join(self.model_directory, f"{timestamp}_{self.ticker}_garch.pkl")

        joblib.dump(self.model, filepath)

        arima_path = os.path.join(self.model_directory, f"{timestamp}_{self.ticker}_arima.pkl")
        joblib.dump(self.arima_model, arima_path)
        # Return filepath
        return filepath

    def load(self):
        """Load most recent model in `self.model_directory` for `self.ticker`,
        attach to `self.model` attribute.

        """
        garch_pattern = os.path.join(self.model_directory, f"*{self.ticker}.pkl")
        arima_pattern = os.path.join(self.model_directory, f"*{self.ticker}_arima.pkl")
        
        try:
            garch_files = [f for f in glob(garch_pattern) if "_arima" not in f]
            arima_files = glob(arima_pattern)
            
            if not garch_files:
                raise Exception(f"No GARCH model trained for '{self.ticker}'.")
            if not arima_files:
                raise Exception(f"No ARIMA model trained for '{self.ticker}'.")
                
            self.model_path = sorted(garch_files)[-1]
            self.arima_path = sorted(arima_files)[-1]
        except IndexError:
            raise Exception(f"No model trained for '{self.ticker}'.")

        self.model = joblib.load(self.model_path)
        self.arima_model = joblib.load(self.arima_path)
        
        # Reload data for date indexing in predictions
        if self.repo:
            df = self.repo.read_table(table_name=self.ticker, limit=100)
            df.sort_index(ascending=True, inplace=True)
            df['return'] = df['close'].pct_change() * 100
            self.data = df['return'].dropna()
    
    def fit_arima(self, order=(1, 0, 1)):
        """Fit an ARIMA model to `self.data` (returns series).

        Parameters
        ----------
        order : tuple, optional
            The (p, d, q) order of the ARIMA model. By default (1, 0, 1).
            - p: autoregressive lags
            - d: differencing order (0 since returns are usually stationary)
            - q: moving average lags

        Returns
        -------
        None
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.arima_model = ARIMA(self.data, order=order).fit()

        self.arima_aic = self.arima_model.aic
        self.arima_bic = self.arima_model.bic

    def predict_returns(self, horizon):
        """Predict returns using `self.arima_model`.

        Parameters
        ----------
        horizon : int
            Number of business days to forecast.

        Returns
        -------
        dict
            Forecast of returns. Each key is a date in ISO 8601 format.
            Each value is the predicted return.
        """
        forecast = self.arima_model.forecast(steps=horizon)

        # Build business-day date index starting from the day after last observation
        start = self.data.index[-1] + pd.DateOffset(days=1)
        prediction_dates = pd.bdate_range(start=start, periods=horizon)
        prediction_index = [d.isoformat() for d in prediction_dates]

        return pd.Series(forecast.values, index=prediction_index).to_dict()