from pathlib import Path
import pandas as pd

from src.player_metrics_builder import build_player_metrics

EVENTS_PATH = Path("event_data/processed/top5_events_current.parquet")
PLAYER_METRICS_PATH = Path("event_data/processed/player_metrics.parquet")


def update_player_metrics() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el parquet de eventos: {EVENTS_PATH}"
        )

    df_events = pd.read_parquet(EVENTS_PATH)

    player_metrics = build_player_metrics(df_events)

    PLAYER_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    player_metrics.to_parquet(PLAYER_METRICS_PATH, index=False)

    return player_metrics


if __name__ == "__main__":
    player_metrics = update_player_metrics()
    print("player_metrics actualizado correctamente")
    print(player_metrics.shape)