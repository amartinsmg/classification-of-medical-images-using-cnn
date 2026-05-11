#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

db.py — Experiment persistence in the database.

Typical usage in the execution loop:

    from db import get_engine, insert_run

    engine = get_engine("sqlite:///base/experiments.db")

    for i in range(3):
        config  = train_pipeline(...)
        metrics = test_pipeline(...)

        insert_run(
            engine,
            experiment = EXPERIMENT_NAME,
            run_name   = f"run-{i+1:02d}",
            config     = config,
            metrics    = metrics
        )

"""

import pathlib
import warnings

import pandas as pd
import sqlalchemy


def get_engine(db_url: str, schema_dir="schema"):
    engine = sqlalchemy.create_engine(db_url)

    schema_path = pathlib.Path(schema_dir) / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")

    if db_url.startswith("sqlite"):

        @sqlalchemy.event.listens_for(engine, "connect")
        def set_sqlite_pragma(conn, _):
            conn.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as conn:
        db_api = conn.connection
        db_api.executescript(schema_sql)

    return engine


def _parse_params(params: dict):
    p = {k.replace("-", "_"): v for k, v in params.items()}

    optimizer = p.pop("optimizer", {})
    p["optimizer_name"] = optimizer.get("name")
    p["learning_rate"] = optimizer.get("learning-rate")

    preprocessing = p.pop("preprocessing", [])
    non_aug = [x for x in preprocessing if x != "data-augmentation"]
    p["normalization"] = non_aug[0] if non_aug else None
    p["data_aug"] = "data-augmentation" in preprocessing

    image_size = p.pop("image-size", [224, 224])
    p["image_size"] = f"{image_size[0]}x{image_size[1]}"

    return p


def _parse_metrics(metrics: dict):
    m = {k.replace("-", "_"): v for k, v in metrics.items()}

    return m


def insert_run(engine, experiment: str, run_name: str, config: dict, metrics: dict):
    with engine.connect() as conn:
        try:
            result = conn.execute(
                sqlalchemy.text(
                    "INSERT INTO runs (experiment, run_name) "
                    + "VALUES (:experiment, :run_name)"
                ),
                {"experiment": experiment, "run_name": run_name},
            )
            conn.commit()
            RUN_ID = result.lastrowid
        except sqlalchemy.exc.IntegrityError:
            warnings.warn(
                f"Run '{run_name}' of experiment '{experiment}' already exists. Skipping insertion.",
                UserWarning,
                stacklevel=2,
            )
            return None

    parsed_params = _parse_params(
        {**config, "decision-threshold": metrics["decision-threshold"]}
    )
    parsed_metrics = _parse_metrics(
        {k: v for k, v in metrics.items() if k != "decision-threshold"}
    )

    param_row = {"run_id": RUN_ID, **parsed_params}
    metric_row = {"run_id": RUN_ID, **parsed_metrics}

    pd.DataFrame([param_row]).to_sql(
        "run_params", engine, if_exists="append", index=False
    )
    pd.DataFrame([metric_row]).to_sql(
        "run_metrics", engine, if_exists="append", index=False
    )

    print(f"Run '{run_name}' of experiment '{experiment}' inserted.")

    return RUN_ID
