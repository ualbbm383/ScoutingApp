from pathlib import Path
import pandas as pd
import streamlit as st
import duckdb

PLAYER_METRICS_PATH = Path("event_data/processed/player_metrics_enriched.parquet")
TEAM_METRICS_PATH = Path("event_data/processed/team_metrics.parquet")
EVENTS_PARQUET_PATH = Path("event_data/processed/top5_events_current.parquet")
DUCKDB_PATH = Path("event_data/processed/events.duckdb")


@st.cache_data
def load_player_metrics():
    if not PLAYER_METRICS_PATH.exists():
        raise FileNotFoundError(f"No se encontró {PLAYER_METRICS_PATH}")
    return pd.read_parquet(PLAYER_METRICS_PATH)


@st.cache_data
def load_team_metrics():
    if not TEAM_METRICS_PATH.exists():
        raise FileNotFoundError(f"No se encontró {TEAM_METRICS_PATH}")
    return pd.read_parquet(TEAM_METRICS_PATH)


def build_duckdb():
    if not EVENTS_PARQUET_PATH.exists():
        raise FileNotFoundError(f"No se encontró {EVENTS_PARQUET_PATH}")

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute("DROP TABLE IF EXISTS top5_events")
        con.execute(f"""
            CREATE TABLE top5_events AS
            SELECT *
            FROM read_parquet('{EVENTS_PARQUET_PATH.as_posix()}')
        """)
    finally:
        con.close()


@st.cache_resource
def ensure_duckdb():
    if not DUCKDB_PATH.exists():
        build_duckdb()


def query_events(sql: str) -> pd.DataFrame:
    ensure_duckdb()

    try:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    except Exception:
        if DUCKDB_PATH.exists():
            DUCKDB_PATH.unlink()

        build_duckdb()

        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            return con.execute(sql).df()
        finally:
            con.close()