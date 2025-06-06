def preprocess_csvs(file_path, weights=None):
    # 1. List and sort files to maintain order
    csv_files = sorted([f for f in os.listdir(file_path) if f.endswith('.csv')])
    dataframes = []
    date_sets = []

    # 2. Read each file and store DataFrame and date set
    for idx, file in enumerate(csv_files, 1):
        df = pd.read_csv(os.path.join(file_path, file), parse_dates=['Date'])
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.set_index('Date', inplace=True)
        # Rename columns to match required format (e.g., 1_Open, 1_High, ...)
        df.columns = [f"{idx}_{col}" for col in df.columns]
        dataframes.append(df)
        date_sets.append(set(df.index))

    # 3. Find intersection of all dates
    common_dates = sorted(set.intersection(*date_sets))

    # 4. Reindex all dataframes to common dates
    dataframes = [df.loc[common_dates] for df in dataframes]

    # 5. Concatenate along columns
    merged_df = pd.concat(dataframes, axis=1)
    merged_df.index.name = 'Date'

    # 6. Handle missing data using linear interpolation (row-wise)
    merged_df.interpolate(method='linear', axis=0, inplace=True, limit_direction='both')

    # 7. Handle outliers using IQR capping for each column
    for col in merged_df.columns:
        Q1 = merged_df[col].quantile(0.25)
        Q3 = merged_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_cap = Q1 - 1.5 * IQR
        upper_cap = Q3 + 1.5 * IQR
        merged_df[col] = np.where(merged_df[col] < lower_cap, lower_cap, merged_df[col])
        merged_df[col] = np.where(merged_df[col] > upper_cap, upper_cap, merged_df[col])

    # 8. Ensure weights are in the order of the files
    if weights is not None:
        assert len(weights) == len(csv_files), "Number of weights must match number of files"
        weights = np.array(weights)
        # Optionally, you can normalize weights if needed:
        weights = weights / 100

    return merged_df, weights
