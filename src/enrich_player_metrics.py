from pathlib import Path
import pandas as pd

PLAYER_METRICS_PATH = Path("event_data/processed/player_metrics.parquet")
PLAYER_METADATA_PATH = Path("data/player_metadata_master.parquet")

OUTPUT_PATH = Path("event_data/processed/player_metrics_enriched.parquet")


def enrich_player_metrics():
    print("Cargando player_metrics...")
    player_metrics = pd.read_parquet(PLAYER_METRICS_PATH)

    print("Cargando player_metadata_master...")
    player_metadata = pd.read_parquet(PLAYER_METADATA_PATH)

    merge_cols = ["Player ID", "player_name", "team_name", "league", "season"]

    # seguridad: una sola fila de metadata por clave
    player_metadata = player_metadata.drop_duplicates(subset=merge_cols).copy()

    print("Haciendo merge...")
    enriched = player_metrics.merge(
        player_metadata[
            merge_cols + [
                "position_raw",
                "position_primary",
                "position_group",
                "age",
                "market_value",
                "metadata_source",
            ]
        ],
        on=merge_cols,
        how="left",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(OUTPUT_PATH, index=False)

    print("\nArchivo generado:")
    print(OUTPUT_PATH)
    print("Jugadores:", len(enriched))

    print("\nResumen position_group:")
    print(enriched["position_group"].value_counts(dropna=False))

    print("\nResumen metadata_source:")
    print(enriched["metadata_source"].value_counts(dropna=False))


if __name__ == "__main__":
    enrich_player_metrics()