from pathlib import Path
import re
import unicodedata

import pandas as pd
from rapidfuzz import process, fuzz


PLAYER_METRICS_PATH = Path("event_data/processed/player_metrics.parquet")
STATSBOMB_DIR = Path("data/statsbomb_exports")
OUTPUT_DIR = Path("data/mapping_outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# NORMALIZACIÓN DE TEXTO
# --------------------------------------------------
def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(name: str) -> str:
    return normalize_text(name)


def normalize_competition(comp: str) -> str:
    comp = normalize_text(comp)

    comp_aliases = {
        "la liga": "laliga",
        "laliga": "laliga",
        "premier league": "premier league",
        "bundesliga": "bundesliga",
        "ligue 1": "ligue 1",
        "serie a": "serie a",
    }
    return comp_aliases.get(comp, comp)


def normalize_league(league: str) -> str:
    league = normalize_text(league)

    league_aliases = {
        "laliga": "laliga",
        "premier league": "premier league",
        "bundesliga": "bundesliga",
        "ligue 1": "ligue 1",
        "serie a": "serie a",
    }
    return league_aliases.get(league, league)


# --------------------------------------------------
# CARGAR CSVs DE STATSBOMB
# --------------------------------------------------
def load_statsbomb_exports(statsbomb_dir: Path) -> pd.DataFrame:
    files = sorted(statsbomb_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No se encontraron CSVs en data/statsbomb_exports")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = file.name
        dfs.append(df)

    statsbomb = pd.concat(dfs, ignore_index=True)
    return statsbomb


# --------------------------------------------------
# LIMPIAR STATSBOMB
# --------------------------------------------------
def clean_statsbomb_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={
        "Name": "sb_player_name",
        "Team": "sb_team_name",
        "Competition": "sb_competition",
        "Primary Position": "sb_primary_position",
        "Secondary Position": "sb_secondary_position",
        "Date of Birth": "sb_birth_date",
    })

    expected_cols = [
        "sb_player_name",
        "sb_team_name",
        "sb_competition",
        "sb_primary_position",
        "sb_secondary_position",
        "sb_birth_date",
        "source_file",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["sb_player_name"] = df["sb_player_name"].astype(str).str.strip()
    df["sb_team_name"] = df["sb_team_name"].astype(str).str.strip()

    df["name_key"] = df["sb_player_name"].apply(normalize_name)
    df["competition_key"] = df["sb_competition"].apply(normalize_competition)

    # Quitamos duplicados por nombre + competición + posición
    df = df.drop_duplicates(
        subset=["name_key", "competition_key", "sb_primary_position", "sb_birth_date"]
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# PREPARAR PLAYER_METRICS
# --------------------------------------------------
def prepare_player_metrics(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()

    pm["name_key"] = pm["player_name"].apply(normalize_name)
    pm["league_key"] = pm["league"].apply(normalize_league)

    return pm


# --------------------------------------------------
# MATCH EXACTO
# --------------------------------------------------
def exact_match(pm: pd.DataFrame, sb: pd.DataFrame) -> pd.DataFrame:
    sb_counts = (
        sb.groupby(["name_key"], as_index=False)
        .size()
        .rename(columns={"size": "sb_name_count"})
    )

    sb_exact = sb.merge(sb_counts, on="name_key", how="left")

    merged = pm.merge(
        sb_exact[
            [
                "name_key",
                "sb_player_name",
                "sb_team_name",
                "sb_competition",
                "sb_primary_position",
                "sb_secondary_position",
                "sb_birth_date",
                "source_file",
                "sb_name_count",
            ]
        ],
        on="name_key",
        how="left",
    )

    merged["match_method"] = None
    merged.loc[merged["sb_player_name"].notna(), "match_method"] = "statsbomb_exact_name"

    return merged


# --------------------------------------------------
# FUZZY MATCH
# --------------------------------------------------
def fuzzy_match_unmatched(
    unmatched_pm: pd.DataFrame,
    sb: pd.DataFrame,
    min_score: int = 90,
    restrict_same_league: bool = True,
) -> pd.DataFrame:
    results = []

    for _, row in unmatched_pm.iterrows():
        player_name_key = row["name_key"]
        player_league_key = row["league_key"]

        candidates = sb.copy()

        if restrict_same_league:
            candidates_same_league = candidates[candidates["competition_key"] == player_league_key].copy()
            if not candidates_same_league.empty:
                candidates = candidates_same_league

        if candidates.empty:
            continue

        choices = candidates["name_key"].tolist()

        best = process.extractOne(
            query=player_name_key,
            choices=choices,
            scorer=fuzz.ratio
        )

        if best is None:
            continue

        best_name_key, best_score, best_idx = best

        if best_score < min_score:
            continue

        best_candidate = candidates.iloc[best_idx]

        out = row.to_dict()
        out.update({
            "sb_player_name": best_candidate["sb_player_name"],
            "sb_team_name": best_candidate["sb_team_name"],
            "sb_competition": best_candidate["sb_competition"],
            "sb_primary_position": best_candidate["sb_primary_position"],
            "sb_secondary_position": best_candidate["sb_secondary_position"],
            "sb_birth_date": best_candidate["sb_birth_date"],
            "source_file": best_candidate["source_file"],
            "match_method": f"statsbomb_fuzzy_{best_score}",
            "fuzzy_score": best_score,
        })
        results.append(out)

    if not results:
        return pd.DataFrame(columns=list(unmatched_pm.columns) + [
            "sb_player_name",
            "sb_team_name",
            "sb_competition",
            "sb_primary_position",
            "sb_secondary_position",
            "sb_birth_date",
            "source_file",
            "match_method",
            "fuzzy_score",
        ])

    return pd.DataFrame(results)


# --------------------------------------------------
# PIPELINE PRINCIPAL
# --------------------------------------------------
def merge_statsbomb_into_player_metrics(
    player_metrics: pd.DataFrame,
    statsbomb_df: pd.DataFrame,
    min_fuzzy_score: int = 90,
    restrict_same_league: bool = True,
):
    pm = prepare_player_metrics(player_metrics)
    sb = clean_statsbomb_df(statsbomb_df)

    # Exact
    exact = exact_match(pm, sb)

    exact_matched = exact[exact["sb_player_name"].notna()].copy()
    unmatched_after_exact = exact[exact["sb_player_name"].isna()].copy()

    # Fuzzy
    fuzzy = fuzzy_match_unmatched(
        unmatched_after_exact,
        sb,
        min_score=min_fuzzy_score,
        restrict_same_league=restrict_same_league,
    )

    fuzzy_keys = fuzzy[["Player ID", "player_name", "team_name", "league", "season"]].drop_duplicates()

    final_unmatched = unmatched_after_exact.merge(
        fuzzy_keys,
        on=["Player ID", "player_name", "team_name", "league", "season"],
        how="left",
        indicator=True
    )

    final_unmatched = final_unmatched[final_unmatched["_merge"] == "left_only"].drop(columns="_merge")

    final_matched = pd.concat(
        [exact_matched, fuzzy],
        ignore_index=True,
        sort=False
    )

    # Dataset full: todos los jugadores
    full_output = pm.merge(
        final_matched[
            [
                "Player ID",
                "player_name",
                "team_name",
                "league",
                "season",
                "sb_player_name",
                "sb_team_name",
                "sb_competition",
                "sb_primary_position",
                "sb_secondary_position",
                "sb_birth_date",
                "source_file",
                "match_method",
            ]
        ].drop_duplicates(
            subset=["Player ID", "player_name", "team_name", "league", "season"]
        ),
        on=["Player ID", "player_name", "team_name", "league", "season"],
        how="left",
    )

    full_output["sb_matched"] = full_output["sb_player_name"].notna()

    fuzzy_review = final_matched[
        final_matched["match_method"].astype(str).str.contains("fuzzy", na=False)
    ].copy()

    return full_output, final_matched, final_unmatched, fuzzy_review


# --------------------------------------------------
# RUN
# --------------------------------------------------
def run_merge_statsbomb():
    print("Cargando player_metrics...")
    player_metrics = pd.read_parquet(PLAYER_METRICS_PATH)

    print("Cargando exports de StatsBomb...")
    sb_raw = load_statsbomb_exports(STATSBOMB_DIR)

    print("Haciendo merge exact + fuzzy...")
    full_output, matched_output, unmatched_output, fuzzy_review = merge_statsbomb_into_player_metrics(
        player_metrics,
        sb_raw,
        min_fuzzy_score=90,
        restrict_same_league=True,
    )

    # Guardar outputs
    full_output.to_parquet(OUTPUT_DIR / "player_metadata_statsbomb_full.parquet", index=False)
    matched_output.to_parquet(OUTPUT_DIR / "player_metadata_statsbomb_matched.parquet", index=False)
    unmatched_output.to_excel(OUTPUT_DIR / "player_metadata_statsbomb_unmatched.xlsx", index=False)
    fuzzy_review.to_excel(OUTPUT_DIR / "player_metadata_statsbomb_fuzzy_review.xlsx", index=False)

    total = len(full_output)
    matched = full_output["sb_matched"].sum()

    print("\nRESULTADOS")
    print("Total jugadores:", total)
    print("Matched:", matched)
    print("Unmatched:", total - matched)
    print("Match rate:", round(matched / total * 100, 2), "%")

    print("\nResumen por método:")
    print(matched_output["match_method"].value_counts(dropna=False))


if __name__ == "__main__":
    run_merge_statsbomb()