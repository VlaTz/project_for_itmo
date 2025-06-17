import logging
import warnings

from forecast_demand import forecast_demand
from multiprocessing_logging import install_mp_handler

warnings.simplefilter(action='ignore', category=FutureWarning)
install_mp_handler()
logging.basicConfig(level=logging.DEBUG, filename="output/py_log.log", filemode="w", force=True,
                    format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    kwargs = {'forecasting_config': 'settings.csv',
              'detailed_preferences': 'detailed_preferences.csv',
              'forecast_fact_out': 'output//forecast_fact.csv',
              'forecast_future_out': 'output//forecast_future.csv',
              'meta_parameters_out': 'output//meta_parameters_out.csv'}
    forecast_demand('fact.csv', **kwargs)
