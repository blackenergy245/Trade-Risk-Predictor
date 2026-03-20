📋 OverviewTradeGuard AI is a high-precision machine learning pipeline designed to automate the risk assessment of international trade shipments. By analyzing geopolitical factors, economic data, and historical compliance, the model provides a real-time "Risk Score" (0-100) to assist border authorities in identifying high-threat cargo.
🚀 Key Performance MetricsAfter running an extensive GridSearchCV tournament across multiple algorithms, the champion model achieved:R² Score: 0.988 (98.8% variance explained)Mean Absolute Error (MAE): 2.59 pointsMean Squared Error (MSE): 9.86
🧠 The Winning ArchitectureThe system utilizes a sophisticated Polynomial Lasso Regression pipeline:StandardScaler: Ensures all features (GDP, Transit Days, etc.) are on a level playing field.PolynomialFeatures (Degree 3): Captures complex, non-linear interactions between variables (e.g., how the impact of a Sanction changes based on a country's GDP).Lasso Regularization ($\alpha=0.1$): Automatically performs feature selection by shrinking "noisy" interaction terms to zero, preventing overfitting.
🛠️ Tech Stack & FeaturesHyperparameter Tuning: Automated via GridSearchCV with 5-fold Cross-Validation.Model Comparison: Evaluated Ridge, Lasso, ElasticNet, KNN, Decision Trees, and Random Forests.Persistence: Model exported via joblib for instant deployment without re-training
├── trade.csv                # Training dataset
├── trade_predictor.ipynb    # Main analysis & training script
├── Trade predictor.pkl      # Serialized champion model
└── requirements.txt         # Dependency list
