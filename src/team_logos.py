from pathlib import Path
import streamlit as st

LOGOS_DIR = Path("logos")

TEAM_LOGO_MAP = {
    # LaLiga
    "Athletic Club": "athletic-club-bilbao-footballlogos-org",
    "Atletico": "atletico-madrid-footballlogos-org",
    "Barcelona": "fc-barcelona-footballlogos-org",
    "Real Madrid": "real-madrid-footballlogos-org",
    "Real Betis": "real-betis-footballlogos-org",
    "Sevilla": "sevilla-fc-footballlogos-org",
    "Real Sociedad": "real-sociedad-footballlogos-org",
    "Osasuna": "osasuna-footballlogos-org",
    "Mallorca": "rcd-mallorca-footballlogos-org",
    "Espanyol": "rcd-espanyol-footballlogos-org",
    "Celta Vigo": "celta-vigo-footballlogos-org",
    "Girona": "girona-fc-footballlogos-org",
    "Getafe": "getafe-cf-footballlogos-org",
    "Deportivo Alaves": "deportivo-alaves-footballlogos-org",
    "Rayo Vallecano": "rayo-vallecano-footballlogos-org",
    "Valencia": "valencia-cf-footballlogos-org",
    "Villarreal": "villarreal-cf-footballlogos-org",

    # Premier League
    "Arsenal": "arsenal-footballlogos-org",
    "Aston Villa": "aston-villa-footballlogos-org",
    "Bournemouth": "bournemouth-footballlogos-org",
    "Brentford": "brentford-footballlogos-org",
    "Brighton": "brighton-hove-footballlogos-org",
    "Burnley": "burnley-footballlogos-org",
    "Chelsea": "chelsea-footballlogos-org",
    "Crystal Palace": "crystal-palace-footballlogos-org",
    "Everton": "everton-footballlogos-org",
    "Fulham": "fulham-footballlogos-org",
    "Liverpool": "liverpool-fc-footballlogos-org",
    "Man City": "manchester-city-footballlogos-org",
    "Man Utd": "manchester-united-footballlogos-org",
    "Newcastle": "england_newcastle_1500x1500",
    "Nottingham Forest": "nottingham-forest-footballlogos-org",
    "Sunderland": "sunderland-footballlogos-org",
    "Tottenham": "tottenham-hotspur-footballlogos-org",
    "West Ham": "west-ham-united-footballlogos-org",
    "Wolves": "wolverhampton-footballlogos-org",

    # Serie A
    "AC Milan": "ac-milan-footballlogos-org",
    "Roma": "roma-footballlogos-org",
    "Atalanta": "atalanta-footballlogos-org",
    "Bologna": "bologna-footballlogos-org",
    "Cagliari": "cagliari-footballlogos-org",
    "Como": "como-1907-footballlogos-org",
    "Cremonese": "cremonese-footballlogos-org",
    "Fiorentina": "fiorentina-footballlogos-org",
    "Genoa": "genoa-footballlogos-org",
    "Hellas Verona": "hellas-verona-footballlogos-org",
    "Verona": "hellas-verona-footballlogos-org",
    "Inter": "inter-milan-footballlogos-org",
    "Juventus": "juventus-footballlogos-org",
    "Lazio": "lazio-footballlogos-org",
    "Lecce": "lecce-footballlogos-org",
    "Napoli": "napoli-footballlogos-org",
    "Parma Calcio 1913": "parma-footballlogos-org",
    "Pisa": "pisa-footballlogos-org",
    "Sassuolo": "sassuolo-footballlogos-org",
    "Torino": "torino-footballlogos-org",
    "Udinese": "udinese-footballlogos-org",

    # Ligue 1
    "Auxerre": "aj-auxerre-footballlogos-org",
    "Monaco": "as-monaco-footballlogos-org",
    "Angers": "angers-sco-footballlogos-org",
    "Lorient": "fc-lorient-footballlogos-org",
    "Metz": "fc-metz-footballlogos-org",
    "Nantes": "fc-nantes-footballlogos-org",
    "Lille": "losc-lille-footballlogos-org",
    "Le Havre": "le-havre-ac-footballlogos-org",
    "Nice": "ofc-nice-footballlogos-org",
    "Lyon": "olympique-lyonnais-footballlogos-org",
    "Marseille": "olympique-de-marseille-footballlogos-org",
    "Paris FC": "paris-fc-footballlogos-org",
    "PSG": "paris-saint-germain-footballlogos-org",
    "Lens": "rc-lens-footballlogos-org",
    "Strasbourg": "rc-strasbourg-alsace-footballlogos-org",
    "Brest": "brest-footballlogos-org",
    "Rennes": "stade-rennais-footballlogos-org",
    "Toulouse": "toulouse-fc-footballlogos-org",

    # Bundesliga
    "FC Koln": "1-fc-koln-footballlogos-org",
    "Leverkusen": "bayer-leverkusen-footballlogos-org",
    "Bayern": "bayern-munich-footballlogos-org",
    "Borussia Dortmund": "borussia-dortmund-footballlogos-org",
    "Borussia M.Gladbach": "borussia-monchengladbach-footballlogos-org",
    "Eintracht Frankfurt": "eintracht-frankfurt-footballlogos-org",
    "Augsburg": "fc-augsburg-footballlogos-org",
    "FC Heidenheim": "fc-heidenheim-footballlogos-org",
    "St. Pauli": "st-pauli-footballlogos-org",
    "Hamburg": "hamburger-sv-footballlogos-org",
    "Mainz": "mainz-05-footballlogos-org",
    "RBL": "rb-leipzig-footballlogos-org",
    "Freiburg": "sc-freiburg-footballlogos-org",
    "Hoffenheim": "tsg-hoffenheim-footballlogos-org",
    "Union Berlin": "union-berlin-footballlogos-org",
    "Stuttgart": "vfb-stuttgart-footballlogos-org",
    "Wolfsburg": "wfl-wolfsburg-footballlogos-org",
    "Werder Bremen": "werder-bremen-footballlogos-org",
}

def get_team_logo_path(team_name: str):
    if not team_name:
        return None

    base_name = TEAM_LOGO_MAP.get(team_name.strip())
    print("TEAM:", team_name)
    print("BASE NAME:", base_name)

    if not base_name:
        return None

    for ext in [".png", ".webp", ".jpg", ".jpeg", ".svg"]:
        candidate = LOGOS_DIR / f"{base_name}{ext}"
        print("CHECKING:", candidate)

        if candidate.exists():
            print("FOUND:", candidate)
            return candidate

    print("NOT FOUND")
    return None


def show_team_logo(team_name: str, width: int = 90):
    logo_path = get_team_logo_path(team_name)
    if logo_path is not None:
        st.image(str(logo_path), width=width)