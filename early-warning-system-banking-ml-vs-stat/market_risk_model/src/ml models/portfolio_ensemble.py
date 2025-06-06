series = data['log_return'].values
def create_dataset(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_dataset(series, window_size)

# Scaling
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X.reshape(-1, window_size)).reshape(-1, window_size, 1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1,1))

# ------------------ Embargo Split ------------------
def embargo_splits(n_samples, n_splits=5, embargo_size=5):
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = n_samples if i == n_splits-1 else (i+1) * fold_size
        test_idx = indices[test_start:test_end]
        emb_lower = max(0, test_start - embargo_size)
        emb_upper = min(n_samples, test_end + embargo_size)
        train_idx = np.concatenate([indices[:emb_lower], indices[emb_upper:]])
        yield train_idx, test_idx

# For simplicity, use the first fold
n_samples = X_scaled.shape[0]
train_idx, test_idx = next(embargo_splits(n_samples, n_splits=5, embargo_size=5))

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

# ------------------ Base Models ------------------
input_shape = (window_size, 1)

def lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def transformer_model(input_shape, num_heads=4, ff_dim=32):
    inputs = Input(shape=input_shape)
    x = LayerNormalization()(inputs)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(x, x)
    x = Add()([x, attn_output])
    x_ff = LayerNormalization()(x)
    x_ff = Dense(ff_dim, activation='relu')(x_ff)
    x_ff = Dense(input_shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = GlobalAveragePooling1D()(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def cnn_rnn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = SimpleRNN(64)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Instantiate models
model_lstm = lstm_attention_model(input_shape)
model_trans = transformer_model(input_shape)
model_cnn_rnn = cnn_rnn_model(input_shape)

# ------------------ Train Base Models ------------------
EPOCHS = 20
BATCH_SIZE = 32

print("Training LSTM + Attention...")
model_lstm.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

print("Training Transformer...")
model_trans.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

print("Training CNN + RNN...")
model_cnn_rnn.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# ------------------ Meta-Model (Stacking) ------------------
# Get predictions from base models
pred_lstm = model_lstm.predict(X_test)
pred_trans = model_trans.predict(X_test)
pred_cnn_rnn = model_cnn_rnn.predict(X_test)

# Stack features
stacked_features = np.hstack([pred_lstm, pred_trans, pred_cnn_rnn])

# Train meta-model
meta_model = LinearRegression()
meta_model.fit(stacked_features, y_test)

# Predict final output
final_pred_scaled = meta_model.predict(stacked_features)
final_pred = scaler_y.inverse_transform(final_pred_scaled.reshape(-1, 1))
y_true = scaler_y.inverse_transform(y_test)

# Evaluate
mse = mean_squared_error(y_true, final_pred)
print(f"\nStacked Ensemble MSE (with embargo): {mse:.5f}")




