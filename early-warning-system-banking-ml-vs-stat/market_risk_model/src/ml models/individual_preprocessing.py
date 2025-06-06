def preprocess_ohlcv(df, dataset_type='stock', weights=None, epsilon=1e-6):
    """
    Preprocesses OHLCV data for stock or portfolio, checking columns, handling missing values,
    cleaning outliers with IQR, normalizing, and calculating features.
    
    Parameters:
    - df: Input DataFrame with OHLCV data (single stock or portfolio).
    - dataset_type: 'stock' for single stock, 'portfolio' for portfolio.
    - weights: List of weights for portfolio stocks (sum to 1, required for portfolio).
    - epsilon: Small value to avoid log(0) in log returns (default=1e-6).
    
    Returns:
    - processed_df: Preprocessed DataFrame with calculated features.
    """
    
    # Step 1: Check and Prepare Columns
    if dataset_type == 'stock':
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError("Single stock DataFrame must contain: Date, Open, High, Low, Close, Volume")
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[expected_cols].copy()
    else:
        if weights is None:
            raise ValueError("Weights must be provided for portfolio dataset")
        n_stocks = len(weights)
        if abs(sum(weights) - 1) > 1e-6:
            raise ValueError("Portfolio weights must sum to 1")
        expected_cols = ['Date'] + [f'{col}_{i}' for i in range(n_stocks) for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Portfolio DataFrame must contain: Date, Open_0, High_0, ..., Volume_{n_stocks-1}")
        ohlcv_cols = [col for col in df.columns if col != 'Date']
        df = df[expected_cols].copy()
    
    # Step 2: Index Date Column
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Step 3: Handle Missing Values with Linear Interpolation
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.interpolate(method='time', limit_direction='both')
    else:
        df = df.interpolate(method='linear', limit_direction='both')
    
    # Step 4: Clean Outliers with IQR Method
    def remove_iqr_outliers(data, cols):
        Q1 = data[cols].quantile(0.25)
        Q3 = data[cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = ((data[cols] >= lower_bound) & (data[cols] <= upper_bound)).all(axis=1)
        return data[mask]
    
    if dataset_type == 'stock':
        df = remove_iqr_outliers(df, ohlcv_cols)
    else:
        # Apply IQR per stock
        mask = None
        for i in range(len(weights)):
            stock_cols = [f'{col}_stock_cols{i}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
            stock_mask = remove_iqr_outliers(df, stock_cols).index
            mask = stock_mask & mask if mask is not None else stock_mask
        df = df.loc[mask]
    
    # Step 5: Normalize Data with MinMaxScaler
    scaler = MinMaxScaler()
    df[ohlcv_cols] = scaler.fit_transform(df[ohlcv_cols])
    
    # Step 6: Calculate Features
    processed_df = df.copy()
    
    if dataset_type == 'stock':
        # Log return
        processed_df['log_return'] = np.log((processed_df['Close'] + epsilon) / (processed_df['Close'].shift(1) + epsilon))
        # Volatility (20-day)
        processed_df['volatility_20d'] = processed_df['log_return'].rolling(window=20).std()
        # SMA (50-day)
        processed_df['sma_50'] = processed_df['Close'].rolling(window=50).mean()
    else:
        # Per-stock features
        for i in range(len(weights)):
            processed_df[f'{i}_log_return'] = np.log((processed_df['f'{i}_Close'] + epsilon]) / 
                                           (processed_df['f'{i}_Close'].shift(1) + epsilon]))
            processed_df['f'{i}_volatility_20d'] = processed_df[f'{i}_log_return'].rolling(window=20).std()
            processed_df[f'{i}_sma_50'] = processed_df[f'{i}_Close'].rolling(window=50).mean()
        # Portfolio-level features
        processed_df['portfolio_return'] = sum(weights[i] * processed_df[f'{i}_log_return'] for i in range(len(weights)))
        processed_df['portfolio_volatility_30d'] = processed_df['portfolio_return'].rolling(window=30).std()
        processed_df['portfolio_sma_50'] = sum(weights[i] * processed_df[f'{i}_sma_50'] for i in range(len(weights)))
    
    # Step 7: Drop NaN Values
    processed_df = processed_df.dropna()
    
    return processed_df