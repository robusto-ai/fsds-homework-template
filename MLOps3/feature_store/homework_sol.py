
import pandas as pd
from feast import Entity, FeatureView, FileSource, FeatureStore, ValueType, Feature
from datetime import datetime, timedelta

def define_driver_features():
    driver = Entity(name="driver", join_keys=["driver_id"], value_type=ValueType.STRING)

    source = FileSource(
        path="data/driver_stats.parquet",
        timestamp_field="event_timestamp"
    )

    stats_view = FeatureView(
        name="driver_stats",
        entities=[driver],
        ttl=timedelta(seconds=86400),
        features=[
            Feature(name="conv_rate", dtype=ValueType.FLOAT),
            Feature(name="acc_rate", dtype=ValueType.FLOAT)
        ],
        source=source
    )

    return driver, stats_view

def materialize_features():
    store = FeatureStore(repo_path="feast_repo")
    entity, view = define_driver_features()
    store.apply([entity, view])
    store.materialize(
        start_date=datetime(2025, 2, 27),
        end_date=datetime(2025, 2, 28)
    )

def predict_driver_performance(driver_id):
    store = FeatureStore(repo_path="feast_repo")
    features = store.get_online_features(
        features=["driver_stats:conv_rate", "driver_stats:acc_rate"],
        entity_rows=[{"driver_id": driver_id}]
    ).to_dict()

    if not features.get("conv_rate") or not features.get("acc_rate"):
        raise ValueError(f"No features found for driver_id {driver_id}")

    conv = features["conv_rate"][0]
    acc = features["acc_rate"][0]
    performance = "good" if conv + acc > 1.5 else "poor"

    return {"driver_id": driver_id, "conv_rate": conv, "acc_rate": acc, "performance": performance}