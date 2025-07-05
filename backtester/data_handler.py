# backtester/data_handler.py
import time
import requests
import pandas as pd
from datetime import datetime
import yaml

class DeltaExchangeDataHandler:
    def __init__(self, config_path='../config/config.yaml'):
        """
        Initializes the DeltaExchangeDataHandler with configuration.
        """
        #with open(config_path, 'r') as f:
            #self.config = yaml.safe_load(f)
        self.base_url = "https://cdn.india.deltaex.org/v2/history"

    def fetch_historical_data(self, ticker, timeframe, start_date, end_date):
        """
        Fetches historical candlestick data for a given ticker and timeframe
        from the Delta Exchange API, handling reverse chronological order and limit.
        """
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        resolution = self._timeframe_to_resolution(timeframe)
        all_candles = []
        current_end_time = end_timestamp  # Start from the end and go backwards

        while current_end_time > start_timestamp:
            request_end_time = current_end_time
            request_start_time = max(start_timestamp, current_end_time - (2000 * self._resolution_to_seconds(resolution)))

            params = {
                'resolution': resolution,
                'symbol': ticker,
                'start': request_start_time,
                'end': request_end_time
            }

            #print(f"Requesting data from: {datetime.fromtimestamp(request_start_time)} to {datetime.fromtimestamp(request_end_time)}")

            try:
                response = requests.get(f"{self.base_url}/candles", params=params)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict) and 'result' in data and isinstance(data['result'], list):
                    candles = sorted([{'timestamp': datetime.fromtimestamp(item['time']),
                                       'open': float(item['open']),
                                       'high': float(item['high']),
                                       'low': float(item['low']),
                                       'close': float(item['close']),
                                       'volume': float(item.get('volume', 0.0))}
                                      for item in data['result']], key=lambda x: x['timestamp'])  # Sort by timestamp

                    if candles:
                        print(f"  Fetched {len(candles)} candles. First timestamp: {candles[0]['timestamp']}, Last timestamp: {candles[-1]['timestamp']}")
                        all_candles.extend(candles)
                        current_end_time = int(candles[0]['timestamp'].timestamp())-1  # Move backwards to the start of the fetched range
                    else:
                        print("  No candles fetched in this range.")
                        break
                else:
                    print(f"Error fetching data for {ticker} at timestamp {request_start_time}: {data}")
                    break

                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Request error for {ticker} at timestamp {request_start_time}: {e}")
                break

        df = pd.DataFrame(all_candles)
        df_sorted_asc = df.sort_values(by='timestamp')
        df_sorted_asc.reset_index(drop=True, inplace=True)
        df_sorted_asc = df_sorted_asc.set_index('timestamp')
        return df_sorted_asc

    def _resolution_to_seconds(self, resolution):
        """Converts the Delta Exchange resolution string to seconds."""
        value = int(resolution[:-1])
        unit = resolution[-1].lower()
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

    def _timeframe_to_resolution(self, timeframe):
        """Maps our timeframe format to Delta Exchange's 'resolution' format."""
        timeframe_lower = timeframe.lower()
        if timeframe_lower == '1m':
            return '1m'
        elif timeframe_lower == '5m':
            return '5m'
        elif timeframe_lower == '1h':
            return '1h'
        elif timeframe_lower == '1d':
            return '1D'
        # Add more mappings as needed
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

if __name__ == '__main__':
    # Example usage (only runs if this script is executed directly)
    config_path = '../config/config.yaml'
    # Ensure config directory exists
    import os
    if not os.path.exists('../config'):
        os.makedirs('../config')
    with open(config_path, 'w') as f:
        yaml.dump({'tickers': ['BTCUSD'], 'timeframe': '1h', 'start_date': '2025-01-01', 'end_date': '2025-06-30'}, f)

    data_handler = DeltaExchangeDataHandler()
    tickers = data_handler.config.get('tickers', [])
    timeframe = data_handler.config.get('timeframe', '1h')
    start_date = data_handler.config.get('start_date', None)
    end_date = data_handler.config.get('end_date', None)

    if tickers and start_date and end_date:
        for ticker in tickers:
            historical_data = data_handler.fetch_historical_data(ticker, timeframe, start_date, end_date)
            if not historical_data.empty:
                print(f"Fetched {len(historical_data)} data points for {ticker} ({timeframe} from {start_date} to {end_date}):")
                print(historical_data.head())
                historical_data.to_csv('D:/Downloaded/file2.csv')
            else:
                print(f"Could not fetch historical data for {ticker}.")
    else:
        print("Please configure tickers, start_date, and end_date in config.yaml")