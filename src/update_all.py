from pathlib import Path
import subprocess
import sys

from src.build_top5_events import build_top5_events
from src.update_player_metrics import update_player_metrics
from src.update_team_metrics import update_team_metrics
from src.enrich_player_metrics import enrich_player_metrics
from src.update_player_clusters import update_player_clusters

LEAGUES = [
    "laliga",
    "premier_league",
    "serie_a",
    "bundesliga",
    "ligue1"
]

SCRAPER_PATH = Path("event_data/scraper/script/whoscored_downloader.py")


def run_scraper_for_league(league: str):
    print(f"\nDescargando datos de {league}...")

    cmd = [
        sys.executable,
        str(SCRAPER_PATH),
        "--league",
        league
    ]

    subprocess.run(cmd, check=True)


def update_all():
    print("===== SCRAPING DE EVENTOS =====")
    for league in LEAGUES:
        run_scraper_for_league(league)

    print("\n===== CONSOLIDANDO TOP5 EVENTS =====")
    build_top5_events()

    print("\n===== ACTUALIZANDO PLAYER METRICS =====")
    update_player_metrics()

    print("\n===== ACTUALIZANDO TEAM METRICS =====")
    update_team_metrics()

    print("\n===== ENRIQUECIENDO PLAYER METRICS =====")
    enrich_player_metrics()

    print("\n===== ACTUALIZANDO CLUSTERS DE JUGADORES =====")
    update_player_clusters()

    print("\nProceso completado.")


if __name__ == "__main__":
    update_all()