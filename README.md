# DeepFraud 🧠💳

A blazing-fast fraud detection engine written in **Rust**, with a **Python interface** using `PyO3` and `maturin`.

Built to leverage Rust’s performance and Python’s data science ecosystem.

## 📦 Features

- Custom Neural Network implementation in Rust
- Python bindings via [PyO3](https://pyo3.rs/)
- Training & inference from Python using NumPy-like data
- Tested on the Kaggle [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset

## 🛠 Setup

### Requirements

- Python ≥ 3.8
- Rust ≥ 1.70
- pip + virtualenv (recommended)
- maturin (for building Python bindings)

### 1. Clone and enter the project

```bash
git clone https://github.com/ash2228/DeepFraud.git
cd DeepFraud
