# Quant-Mill

Quant-Mill is a modular quantitative research framework for generating trading signals using both rule-based strategies and machine learning models.

### 🚀 Features
- **Market Data Ingestion** via `yfinance`
- **Feature Engineering** (SMA20/50, returns, volatility)
- **ML Dataset Creation** with forward-return labels
- **Time-Series Cross-Validation** using `TimeSeriesSplit`
- **RandomForest ML Models** for alpha prediction
- **Signal Generators**:
  - Rule-based SMA crossover
  - ML-based probability signals
- **Command-Line Interface** powered by `click`

### 📘 Example Usage
Train an ML model:

quant-signal train-ml --symbol TCS.NS --start 2015-01-01 --horizon 5 --threshold 0.02


Generate rule-based signals:


quant-signal signals -s TCS.NS -s RELIANCE.NS


Generate ML-based signals:

quant-signal signals-ml --symbols TCS.NS --model-path src/quant_signal/models/artifacts/rf_TCS_NS_h5_t0.020.joblib --start 2015-01-01 --horizon 5 --threshold 0.02 --prob-cutoff 0.6 


### 📂 Structure


#### src/quant_signal/ 
#### data/  market data loaders
#### features/  technical indicator features
#### models/  dataset builder + ML trainers
#### signals/  rule-based and ML signal generators
#### cli.py  CLI entrypoint


### 🛠 Dependencies
`pandas`, `numpy`, `scikit-learn`, `yfinance`, `click`, `joblib`

---
