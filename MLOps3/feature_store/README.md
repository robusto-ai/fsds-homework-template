# Homework: Introduction to Feast Feature Store

## Prerequisites

1. **Python Version**: Use Python 3.9+.
2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Directory Structure**:
   - Create a `data` folder in the same directory as `homework.py`.
   - The grader will generate `data/driver_stats.parquet` automatically.

## Requirements and Description

This homework introduces Feast, a feature store for managing ML features. You’ll simulate a driver performance use case with dummy data (e.g., conversion rate and accuracy rate for drivers). The focus is on defining features, materializing them to an online store, and retrieving them for training and inference, all using a simple local setup—no Docker or external services required.

You will implement three tasks:

1. **Exercise 1: Define Driver Features**
   - **Task**: Write a function `define_driver_features` that defines a Feast entity ("driver") and a feature view ("driver_stats") with two features: `conv_rate` (float) and `acc_rate` (float).
   - **Requirements**:
     - Entity: `"driver"` with `driver_id` as the join key (string).
     - Feature View: `"driver_stats"` using a `FileSource` pointing to `"data/driver_stats.parquet"`.
     - Features: `conv_rate` and `acc_rate`, both floats, with a TTL of 1 day (86400 seconds).
   - **Example**:
     ```python
     df = pd.DataFrame({
         "driver_id": ["d1", "d2"],
         "conv_rate": [0.8, 0.6],
         "acc_rate": [0.9, 0.7],
         "event_timestamp": [pd.Timestamp("2025-02-27 10:00"), pd.Timestamp("2025-02-27 10:01")]
     })
     # Defines features for drivers d1 and d2
     ```

2. **Exercise 2: Materialize Features**
   - **Task**: Write a function `materialize_features` that applies feature definitions and materializes data from the parquet file to the online store (SQLite).
   - **Requirements**:
     - Use `FeatureStore` with `repo_path="feast_repo"`.
     - Materialize data from `"2025-02-27 00:00"` to `"2025-02-28 00:00"`.
     - Handle exceptions if materialization fails.
   - **Example**:
     ```bash
     >>> materialize_features()
     # Features for "d1" and "d2" are now in the online store
     ```

3. **Exercise 3: Retrieve Features for Prediction**
   - **Task**: Write a function `predict_driver_performance` that retrieves online features for a `driver_id` and predicts performance: `good` if `conv_rate + acc_rate > 1.5`, else `poor`.
   - **Requirements**:
     - Use `FeatureStore.get_online_features` to fetch `conv_rate` and `acc_rate`.
     - Return a dict with `"driver_id"`, `"conv_rate"`, `"acc_rate"`, and `"performance"`.
     - Raise a `ValueError` if features are missing.
   - **Example**:
     ```bash
     >>> predict_driver_performance("d1")
     {'driver_id': 'd1', 'conv_rate': 0.8, 'acc_rate': 0.9, 'performance': 'good'}
     >>> predict_driver_performance("d3")
     ValueError: No features found for driver_id d3
     ```

### Requirements File (`requirements.txt`)

```
feast==0.31.0
pandas==2.0.3
numpy==1.23.5
```

- **`feast`**: Core library for feature store operations.
- **`pandas`**: For handling dummy data and parquet files.
- **`numpy`**: For compatibility with pandas.

Install with:
```bash
pip install -r requirements.txt
```

## Instructions

Follow these steps in order:

1. **Complete the Homework**:
   - Open `homework.py` and fill in the three functions:
     - `define_driver_features`: Define the entity and feature view.
     - `materialize_features`: Materialize features to the online store.
     - `predict_driver_performance`: Retrieve features and predict performance.

2. **Initialize Feast Repository**:
   - Run this command to create the `feast_repo` directory:
     ```bash
     feast init feast_repo
     ```
   - Replace `feast_repo/feature_repo/example_repo.py` with the output of `define_driver_features` if needed.

## Workflow

1. **Feature Definition**: Define the entity and feature view in `define_driver_features`.
2. **Materialization**: Apply and materialize features in `materialize_features`.
3. **Prediction**: Retrieve and predict in `predict_driver_performance`.

## Notes
- Ensure `feast_repo` is created before running `materialize_features`.
- If you encounter issues, check that `data/driver_stats.parquet` exists and `requirements.txt` is installed.

```

