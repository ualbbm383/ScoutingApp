from pathlib import Path
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process


STATSBOMB_FULL_PATH = Path("data/mapping_outputs/player_metadata_statsbomb_full.parquet")
HUDL_DIR = Path("data/hudl_exports")
OUTPUT_DIR = Path("data/mapping_outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# NORMALIZACIÓN
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


def build_short_name_key(name: str) -> str:
    name = normalize_text(name)
    if not name:
        return ""

    parts = name.split()
    if len(parts) == 1:
        return parts[0]

    first = parts[0]
    last = parts[-1]
    return f"{first[0]}_{last}"


TEAM_ALIASES = {
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "atletico madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "internazionale": "inter",
    "inter milan": "inter",
    "ac milan": "milan",
    "parma calcio 1913": "parma",
    "fc koln": "koln",
    "1 fc koln": "koln",
    "rbl": "rb leipzig",
    "rb leipzig": "rb leipzig",
    "paris saint germain": "psg",
}


def normalize_team(team: str) -> str:
    team = normalize_text(team)
    return TEAM_ALIASES.get(team, team)


# --------------------------------------------------
# CARGAR HUDL
# --------------------------------------------------
def load_hudl_exports(hudl_dir: Path) -> pd.DataFrame:
    files = sorted(hudl_dir.glob("Search results*.xlsx"))

    if not files:
        raise FileNotFoundError("No se encontraron archivos Hudl en data/hudl_exports")

    dfs = []
    for file in files:
        df = pd.read_excel(file)
        df["source_file"] = file.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def clean_hudl_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={
        "Jugador": "hudl_player_name",
        "Equipo": "hudl_team_name",
        "Posición específica": "hudl_specific_position",
        "Edad": "hudl_age",
        "Valor de mercado (Transfermarkt)": "hudl_market_value",
        "Vencimiento contrato": "hudl_contract_end",
        "Pasaporte": "hudl_passport",
        "En prestamo": "hudl_on_loan",
    })

    expected_cols = [
        "hudl_player_name",
        "hudl_team_name",
        "hudl_specific_position",
        "hudl_age",
        "hudl_market_value",
        "hudl_contract_end",
        "hudl_passport",
        "hudl_on_loan",
        "source_file",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["hudl_player_name"] = df["hudl_player_name"].astype(str).str.strip()
    df["hudl_team_name"] = df["hudl_team_name"].astype(str).str.strip()

    df["team_key"] = df["hudl_team_name"].apply(normalize_team)
    df["name_key"] = df["hudl_player_name"].apply(normalize_name)
    df["short_name_key"] = df["hudl_player_name"].apply(build_short_name_key)

    df = df.drop_duplicates(
        subset=["hudl_player_name", "hudl_team_name", "hudl_specific_position", "hudl_age"]
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# CARGAR UNMATCHED DE STATSBOMB
# --------------------------------------------------
def load_statsbomb_unmatched() -> pd.DataFrame:
    df = pd.read_parquet(STATSBOMB_FULL_PATH).copy()
    unmatched = df[df["sb_matched"] == False].copy()

    unmatched["team_key"] = unmatched["team_name"].apply(normalize_team)
    unmatched["name_key"] = unmatched["player_name"].apply(normalize_name)
    unmatched["short_name_key"] = unmatched["player_name"].apply(build_short_name_key)

    return unmatched


# --------------------------------------------------
# STRONG MATCH
# --------------------------------------------------
def strong_match(unmatched: pd.DataFrame, hudl: pd.DataFrame) -> pd.DataFrame:
    hudl_counts = (
        hudl.groupby(["team_key", "short_name_key"], as_index=False)
        .size()
        .rename(columns={"size": "hudl_match_count"})
    )

    hudl2 = hudl.merge(hudl_counts, on=["team_key", "short_name_key"], how="left")

    merged = unmatched.merge(
        hudl2[
            [
                "team_key",
                "short_name_key",
                "hudl_player_name",
                "hudl_team_name",
                "hudl_specific_position",
                "hudl_age",
                "hudl_market_value",
                "hudl_contract_end",
                "hudl_passport",
                "hudl_on_loan",
                "source_file",
                "hudl_match_count",
            ]
        ],
        on=["team_key", "short_name_key"],
        how="left",
    )

    merged["hudl_match_method"] = None
    merged.loc[merged["hudl_player_name"].notna(), "hudl_match_method"] = "hudl_strong_team_short"

    return merged


# --------------------------------------------------
# WEAK MATCH
# --------------------------------------------------
def weak_match(unmatched_after_strong: pd.DataFrame, hudl: pd.DataFrame) -> pd.DataFrame:
    hudl_name_counts = (
        hudl.groupby("short_name_key", as_index=False)
        .size()
        .rename(columns={"size": "short_name_count"})
    )

    hudl2 = hudl.merge(hudl_name_counts, on="short_name_key", how="left")

    weak = unmatched_after_strong.drop(
        columns=[
            "hudl_player_name",
            "hudl_team_name",
            "hudl_specific_position",
            "hudl_age",
            "hudl_market_value",
            "hudl_contract_end",
            "hudl_passport",
            "hudl_on_loan",
            "source_file",
            "hudl_match_count",
            "hudl_match_method",
        ],
        errors="ignore"
    ).merge(
        hudl2[
            [
                "short_name_key",
                "hudl_player_name",
                "hudl_team_name",
                "hudl_specific_position",
                "hudl_age",
                "hudl_market_value",
                "hudl_contract_end",
                "hudl_passport",
                "hudl_on_loan",
                "source_file",
                "short_name_count",
            ]
        ],
        on="short_name_key",
        how="left",
    )

    weak_valid = weak[weak["short_name_count"] == 1].copy()
    weak_valid["hudl_match_method"] = "hudl_weak_short_only"

    return weak_valid


# --------------------------------------------------
# FUZZY MATCH
# --------------------------------------------------
def fuzzy_match(unmatched_after_weak: pd.DataFrame, hudl: pd.DataFrame, min_score: int = 88) -> pd.DataFrame:
    results = []

    for _, row in unmatched_after_weak.iterrows():
        player_name_key = row["name_key"]
        player_short_name_key = row["short_name_key"]

        candidates = hudl[hudl["short_name_key"] == player_short_name_key].copy()
        if candidates.empty:
            candidates = hudl.copy()

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
            "hudl_player_name": best_candidate["hudl_player_name"],
            "hudl_team_name": best_candidate["hudl_team_name"],
            "hudl_specific_position": best_candidate["hudl_specific_position"],
            "hudl_age": best_candidate["hudl_age"],
            "hudl_market_value": best_candidate["hudl_market_value"],
            "hudl_contract_end": best_candidate["hudl_contract_end"],
            "hudl_passport": best_candidate["hudl_passport"],
            "hudl_on_loan": best_candidate["hudl_on_loan"],
            "source_file": best_candidate["source_file"],
            "hudl_match_method": f"hudl_fuzzy_{best_score}",
            "hudl_fuzzy_score": best_score,
        })
        results.append(out)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# --------------------------------------------------
# RUN
# --------------------------------------------------
def run_hudl_fill():
    print("Cargando unmatched de StatsBomb...")
    unmatched = load_statsbomb_unmatched()

    print("Cargando exports de Hudl...")
    hudl_raw = load_hudl_exports(HUDL_DIR)
    hudl = clean_hudl_df(hudl_raw)

    print("Hudl strong match...")
    strong = strong_match(unmatched, hudl)
    strong_matched = strong[strong["hudl_player_name"].notna()].copy()
    unmatched_after_strong = strong[strong["hudl_player_name"].isna()].copy()

    print("Hudl weak match...")
    weak = weak_match(unmatched_after_strong, hudl)

    weak_keys = weak[["Player ID", "player_name", "team_name", "league", "season"]].drop_duplicates()

    unmatched_after_weak = unmatched_after_strong.merge(
        weak_keys,
        on=["Player ID", "player_name", "team_name", "league", "season"],
        how="left",
        indicator=True
    )
    unmatched_after_weak = unmatched_after_weak[unmatched_after_weak["_merge"] == "left_only"].drop(columns="_merge")

    print("Hudl fuzzy match...")
    fuzzy = fuzzy_match(unmatched_after_weak, hudl, min_score=88)

    fuzzy_keys = pd.DataFrame()
    if not fuzzy.empty:
        fuzzy_keys = fuzzy[["Player ID", "player_name", "team_name", "league", "season"]].drop_duplicates()

    if not fuzzy_keys.empty:
        final_unmatched = unmatched_after_weak.merge(
            fuzzy_keys,
            on=["Player ID", "player_name", "team_name", "league", "season"],
            how="left",
            indicator=True
        )
        final_unmatched = final_unmatched[final_unmatched["_merge"] == "left_only"].drop(columns="_merge")
    else:
        final_unmatched = unmatched_after_weak.copy()

    final_matched = pd.concat(
        [strong_matched, weak, fuzzy],
        ignore_index=True,
        sort=False
    )

    # outputs
    final_matched.to_parquet(OUTPUT_DIR / "player_metadata_hudl_fill_matched.parquet", index=False)
    final_unmatched.to_excel(OUTPUT_DIR / "player_metadata_hudl_fill_unmatched.xlsx", index=False)

    if not fuzzy.empty:
        fuzzy.to_excel(OUTPUT_DIR / "player_metadata_hudl_fill_fuzzy_review.xlsx", index=False)

    print("\nRESULTADOS HUDL FILL")
    print("Total recibidos desde StatsBomb unmatched:", len(unmatched))
    print("Matched por Hudl:", len(final_matched))
    print("Siguen unmatched:", len(final_unmatched))

    if "hudl_match_method" in final_matched.columns:
        print("\nResumen por método:")
        print(final_matched["hudl_match_method"].value_counts(dropna=False))


if __name__ == "__main__":
    run_hudl_fill()