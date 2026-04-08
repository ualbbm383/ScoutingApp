from pathlib import Path

import pandas as pd

from src.data_loader import load_player_metrics         
from src.player_clustering import recalculate_and_update_position, get_supported_positions

ENRICHED_PARQUET_PATH = Path("event_data/processed/player_metrics_enriched.parquet")

# minutos mínimos por defecto para cada posición
DEFAULT_MIN_MINUTES = {
    "Midfielder": 600,
    "Center Back": 600,
    "Striker": 600,
    "Winger": 600,
    "Full Back": 600,
}


def update_player_clusters(min_minutes_map: dict | None = None) -> pd.DataFrame:
    if min_minutes_map is None:
        min_minutes_map = DEFAULT_MIN_MINUTES

    df_full = load_player_metrics().copy()

    supported_positions = get_supported_positions()

    for position_group in supported_positions:
        min_minutes = min_minutes_map.get(position_group, 600)

        print(f"Recalculando clusters para {position_group} (min_minutes={min_minutes})...")

        _, df_full = recalculate_and_update_position(
            df_full=df_full,
            position_group=position_group,
            min_minutes=min_minutes,
            parquet_path=ENRICHED_PARQUET_PATH,
        )

    # guardar el resultado final completo
    ENRICHED_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(ENRICHED_PARQUET_PATH, index=False)

    print("Clusters de jugadores actualizados correctamente.")
    return df_full


if __name__ == "__main__":
    df_updated = update_player_clusters()
    print(df_updated.shape)