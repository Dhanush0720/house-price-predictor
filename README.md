# 🏡 Indian Residential Property Valuation Engine

An end-to-end, production-grade machine learning application designed to predict residential real estate market values across regional Indian tier-1 tech hubs and metropolitan zones. 

🚀 https://house-price-predictor-ffzgaejspv64soyup9uxc4.streamlit.app/

---

## 🎯 The Business Problem
Real estate evaluation in expanding Indian tech corridors faces severe price opacity, driven by fragmented localization data, speculative listing padding, and non-standard properties (e.g., varying built-up area ratios, non-numeric configuration syntax like "2.5 BHK"). 

This project implements a completely automated machine learning pipeline that parses real-world raw housing data (13,300+ entries) to deliver clean valuation metrics within a predictable margin of error, optimizing pricing intelligence for prospective retail buyers.

---

## 🏗️ Decoupled Production Architecture
Unlike monolithic notebooks, this system implements a strict software engineering design split into isolated modules:

* **`src/preprocessing.py`**: Handles numerical transformation vectors, categorical handling, and robust type safety features via custom Scikit-Learn pipelines.
* **`src/train.py`**: Automated training engine executing ingestion, hyperparameter mapping, and binary export operations.
* **`app.py`**: Clean, user-facing presentation layer served dynamically over a Streamlit UI dashboard framework.

```text
house-price-predictor/
├── data/raw/             # Source data registries (13k+ Rows)
├── models/               # Serialized Production Pipeline Binaries (.joblib)
├── src/                  # Isolated data engineering modules
└── app.py                # Live Interactive User Dashboard Interface
