# GAT Model
class PortfolioGAT(torch.nn.Module):
    def _init_(self, num_features, hidden_dim=32, heads=2):
        super()._init_()
        self.gat1 = GATConv(num_features, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim, heads=1)
        self.fc = Linear(hidden_dim, 1)
        self.relu = ReLU()
        self.dropout = Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)

def preprocess_gat(portfolio_df, window_size, weights):
    close_cols = [col for col in portfolio_df.columns if col.endswith('_Close')]
    close_cols = sorted(close_cols, key=lambda x: int(x.split('_')[0]))
    assert len(close_cols) == len(weights), f"Number of weights ({len(weights)}) does not match number of close columns ({len(close_cols)}): {close_cols}"

    returns = np.log(portfolio_df[close_cols].astype(float) / portfolio_df[close_cols].astype(float).shift(1))
    volatility = returns.rolling(window=20).std()
    features_df = pd.concat([returns, volatility], axis=1).dropna().reset_index(drop=True)
    weighted_returns = returns.values @ np.array(weights)
    target = pd.Series(weighted_returns).rolling(window=window_size).std().shift(-window_size+1)
    target = target.dropna().reset_index(drop=True)
    features_df = features_df.iloc[:len(target)]
    return features_df, target, close_cols

def create_graph(features_df):
    num_nodes = features_df.shape[1] // 2
    node_features = []
    for i in range(num_nodes):
        node_feat = np.stack([
            features_df.iloc[:, i],
            features_df.iloc[:, i+num_nodes]
        ], axis=1)
        node_features.append(node_feat)
    node_features = np.stack(node_features, axis=1)
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return node_features, edge_index

def train_gat_portfolio_risk(
    portfolio_df,
    weights,
    window_size=20,
    embargo=10,
    n_splits=5,
    epochs=50,
    lr=0.001
):
    try:
        features_df, target, close_cols = preprocess_gat(portfolio_df, window_size, weights)
        node_features, edge_index = create_graph(features_df)
        num_samples, num_nodes, num_node_features = node_features.shape
        metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'R2': []}
        tscv = TimeSeriesSplit(n_splits=n_splits)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for fold, (train_idx, test_idx) in enumerate(tscv.split(node_features)):
            if embargo > 0 and test_idx[0] - train_idx[-1] <= embargo:
                test_idx = test_idx[embargo:]
                if len(test_idx) == 0:
                    continue

            X_train, X_test = node_features[train_idx], node_features[test_idx]
            y_train, y_test = target.iloc[train_idx].values, target.iloc[test_idx].values

            scaler = StandardScaler().fit(X_train.reshape(-1, num_node_features))
            X_train_scaled = scaler.transform(X_train.reshape(-1, num_node_features)).reshape(X_train.shape)
            X_test_scaled = scaler.transform(X_test.reshape(-1, num_node_features)).reshape(X_test.shape)

            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

            model = PortfolioGAT(num_node_features).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                preds = torch.stack([model(X_train_tensor[i], edge_index.to(device)).mean() for i in range(X_train_tensor.shape[0])])
                loss = loss_fn(preds, y_train_tensor)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = torch.stack([model(X_test_tensor[i], edge_index.to(device)).mean() for i in range(X_test_tensor.shape[0])]).cpu().numpy()
            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)
            metrics['MSE'].append(mse)
            metrics['MAE'].append(mae)
            metrics['RMSE'].append(rmse)
            metrics['R2'].append(r2)
            print(f"Fold {fold+1}: MSE={mse:.5f}, MAE={mae:.5f}, RMSE={rmse:.5f}, R2={r2:.5f}")

        results = {k: np.mean(v) for k, v in metrics.items()}
        print("\n--- Cross-Validated Performance ---")
        for k, v in results.items():
            print(f"{k}: {v:.5f}")
        return results, model
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

# Your specific weights for 14 stocks
# weights = np.array([0.08, 0.07, 0.06, 0.09, 0.05, 0.04, 0.13, 0.02, 0.12, 0.05, 0.07, 0.06, 0.08, 0.08])

# Train the model with your parameters
try:
    results, trained_model = train_gat_portfolio_risk(
        portfolio_df=df,
        weights=weights,
        window_size=20,
        embargo=10,
        n_splits=5,
        epochs=50,
        lr=0.001
    )

    if trained_model is not None:
        # Save the model to a pickle file
        pickle_file = 'portfolio_gat_model.pkl'
        if os.path.exists(pickle_file):
            print(f"Warning: File '{pickle_file}' already exists. Overwriting it.")
        with open(pickle_file, 'wb') as f:
            pickle.dump(trained_model, f)
        print(f"Model successfully saved as '{pickle_file}'")
    else:
        print("Model training failed, no pickle file created.")
except Exception as e:
    print(f"Failed to train or save model: {str(e)}")