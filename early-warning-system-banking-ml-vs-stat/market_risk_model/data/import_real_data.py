# List of tickers
tickers = [''] # Enter the ticker for your desired stock
# Create output folder
folder_name = ''
os.makedirs(folder_name, exist_ok=True)
# Download and save adjusted OHLCV for each ticker
for ticker in tickers:
    stock = yf.Ticker(ticker)
    # auto_adjust=True gives adjusted OHLC
    hist = stock.history(period='max', interval='1d', auto_adjust=True)
    # Keep only adjusted OHLCV columns
    hist = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
    hist.to_csv(os.path.join(folder_name, f'{ticker}_daily.csv'))