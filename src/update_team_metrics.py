from pathlib import Path
import pandas as pd

from src.team_metrics_builder import build_team_metrics

EVENTS_PATH = Path("event_data/processed/top5_events_current.parquet")
TEAM_METRICS_PATH = Path("event_data/processed/team_metrics.parquet")


def update_team_metrics() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el parquet de eventos: {EVENTS_PATH}"
        )

    df_events = pd.read_parquet(EVENTS_PATH)

    team_metrics = build_team_metrics(df_events)

    TEAM_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_metrics.to_parquet(TEAM_METRICS_PATH, index=False)

    return team_metrics


if __name__ == "__main__":
    team_metrics = update_team_metrics()
    print("team_metrics actualizado correctamente")
    print(team_metrics.shape)