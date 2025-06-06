import pickle

# Your feature name list
feature_cols = [
    'MLA', 'EWAQ_Capital', 'EWAQ_GrossLoans', '01_CURR_ACC', 'XX_MLA',
    'XX_TOTAL_LIQUID_ASSET', '03_SAVINGS', '02_TIME_DEPOSIT', 'F077_ASSETS_TOTAL',
    'EWAQ_NPL', 'XX_BAL_IN_OTHER_BANKS', '10_FOREIGN_DEPOSITS_AND_BORROWINGS',
    'EWAQ_NPLsNetOfProvisions', 'LR', 'F125_LIAB_TOTAL', '19_BANKS_ABROAD', 'INF',
    'XX_BOT_BALANCE', 'EWAQ_NPLsNetOfProvisions2CoreCapital', 'DR'
]

# Save the list as a pickle file
with open('feature_order_lgd_model.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("Feature names saved to 'feature_order_lgd_model.pkl'")

