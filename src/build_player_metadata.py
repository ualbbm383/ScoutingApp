from pathlib import Path
from datetime import date
import pandas as pd


PLAYER_METRICS_PATH = Path("event_data/processed/player_metrics.parquet")

STATSBOMB_FULL_PATH = Path("data/mapping_outputs/player_metadata_statsbomb_full.parquet")
HUDL_MATCHED_PATH = Path("data/mapping_outputs/player_metadata_hudl_fill_matched.parquet")
MANUAL_POSITIONS_PATH = Path("data/manual_mapping/manual_player_positions.csv")

OUTPUT_METADATA_PATH = Path("data/player_metadata_master.parquet")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def first_hudl_position(pos):
    """
    Ej:
    'LCMF, RCMF, DMF' -> 'LCMF'
    'CF, LW' -> 'CF'
    """
    if pd.isna(pos):
        return None

    pos = str(pos).strip()
    if not pos:
        return None

    first = pos.split(",")[0].strip().upper()
    return first if first else None


def normalize_primary_position(pos):
    """
    Lleva posiciones de distintas fuentes a un lenguaje común
    detallado para comparaciones.
    """
    if pd.isna(pos):
        return None

    pos = str(pos).strip().upper()
    if not pos:
        return None

    # StatsBomb style
    statsbomb_map = {
        "GOALKEEPER": "GK",

        "RIGHT BACK": "RB",
        "LEFT BACK": "LB",
        "RIGHT WING BACK": "RB",
        "LEFT WING BACK": "LB",

        "CENTRE BACK": "CB",
        "CENTER BACK": "CB",
        "RIGHT CENTRE BACK": "CB",
        "LEFT CENTRE BACK": "CB",
        "RIGHT CENTER BACK": "CB",
        "LEFT CENTER BACK": "CB",

        "DEFENSIVE MIDFIELDER": "DMF",
        "LEFT DEFENSIVE MIDFIELDER": "DMF",
        "RIGHT DEFENSIVE MIDFIELDER": "DMF",
        "CENTRE DEFENSIVE MIDFIELDER": "DMF",
        "CENTER DEFENSIVE MIDFIELDER": "DMF",

        "CENTRAL MIDFIELDER": "MF",
        "CENTRE MIDFIELDER": "MF",
        "LEFT CENTRAL MIDFIELDER": "MF",
        "RIGHT CENTRAL MIDFIELDER": "MF",
        "LEFT CENTRE MIDFIELDER": "MF",
        "RIGHT CENTRE MIDFIELDER": "MF",

        "ATTACKING MIDFIELDER": "AMF",
        "CENTRE ATTACKING MIDFIELDER": "AMF",
        "CENTER ATTACKING MIDFIELDER": "AMF",
        "LEFT ATTACKING MIDFIELDER": "AMF",
        "RIGHT ATTACKING MIDFIELDER": "AMF",
        "LEFT CENTRE ATTACKING MIDFIELDER": "AMF",
        "RIGHT CENTRE ATTACKING MIDFIELDER": "AMF",

        "LEFT MIDFIELDER": "LWF",
        "RIGHT MIDFIELDER": "RWF",
        "LEFT WING": "LWF",
        "RIGHT WING": "RWF",

        "CENTRE FORWARD": "CF",
        "CENTER FORWARD": "CF",
        "LEFT CENTRE FORWARD": "CF",
        "RIGHT CENTRE FORWARD": "CF",
        "LEFT CENTER FORWARD": "CF",
        "RIGHT CENTER FORWARD": "CF",
        "STRIKER": "CF",
        "SECOND STRIKER": "CF",
    }

    if pos in statsbomb_map:
        return statsbomb_map[pos]

    # Hudl / manual style
    hudl_manual_map = {
        "GK": "GK",

        "CB": "CB",
        "LCB": "CB",
        "RCB": "CB",

        "LB": "LB",
        "LWB": "LB",

        "RB": "RB",
        "RWB": "RB",

        "DMF": "DMF",
        "LDMF": "DMF",
        "RDMF": "DMF",

        "MF": "MF",
        "CMF": "MF",
        "LCMF": "MF",
        "RCMF": "MF",

        "AMF": "AMF",
        "LAMF": "AMF",
        "RAMF": "AMF",

        "LW": "LWF",
        "LWF": "LWF",

        "RW": "RWF",
        "RWF": "RWF",

        "CF": "CF",
        "ST": "CF",
        "LMF": "LWF",
        "RMF": "RWF",
        "SS": "CF",
    }

    if pos in hudl_manual_map:
        return hudl_manual_map[pos]

    return None


def build_position_group(position_primary):
    """
    Grupo amplio para clustering.
    """
    if pd.isna(position_primary):
        return None

    pos = str(position_primary).strip().upper()

    if pos == "GK":
        return "Goalkeeper"

    if pos in ["LB", "RB"]:
        return "Full Back"

    if pos == "CB":
        return "Center Back"

    if pos in ["DMF", "MF", "AMF"]:
        return "Midfielder"

    if pos in ["LWF", "RWF"]:
        return "Winger"

    if pos == "CF":
        return "Striker"

    return None


def calc_age_from_birthdate(birth_date_value):
    if pd.isna(birth_date_value):
        return None

    try:
        birth_date = pd.to_datetime(birth_date_value).date()
    except Exception:
        return None

    today = date.today()
    age = today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )
    return age


# --------------------------------------------------
# BUILD METADATA
# --------------------------------------------------
def build_player_metadata():
    print("Cargando player_metrics...")
    player_metrics = pd.read_parquet(PLAYER_METRICS_PATH)

    metadata = player_metrics[
        ["Player ID", "player_name", "team_name", "league", "season"]
    ].drop_duplicates().copy()

    metadata["position_raw"] = pd.NA
    metadata["position_primary"] = pd.NA
    metadata["position_group"] = pd.NA
    metadata["age"] = pd.NA
    metadata["market_value"] = pd.NA
    metadata["metadata_source"] = "unknown"

    # --------------------------------------------------
    # 1. STATSBOMB
    # --------------------------------------------------
    if STATSBOMB_FULL_PATH.exists():
        print("Añadiendo StatsBomb...")

        sb = pd.read_parquet(STATSBOMB_FULL_PATH).copy()

        sb = sb[sb["sb_matched"] == True].copy()

        sb = sb[
            [
                "Player ID",
                "sb_primary_position",
                "sb_secondary_position",
                "sb_birth_date",
            ]
        ].drop_duplicates(subset=["Player ID"])

        metadata = metadata.merge(sb, on="Player ID", how="left")

        sb_mask = metadata["sb_primary_position"].notna()

        metadata.loc[sb_mask, "position_raw"] = metadata.loc[sb_mask, "sb_primary_position"]
        metadata.loc[sb_mask, "position_primary"] = metadata.loc[sb_mask, "sb_primary_position"].apply(
            normalize_primary_position
        )
        metadata.loc[sb_mask, "age"] = metadata.loc[sb_mask, "sb_birth_date"].apply(
            calc_age_from_birthdate
        )
        metadata.loc[sb_mask, "metadata_source"] = "statsbomb"

        metadata = metadata.drop(columns=["sb_primary_position", "sb_secondary_position", "sb_birth_date"])

    # --------------------------------------------------
    # 2. HUDL (solo rellena huecos)
    # --------------------------------------------------
    if HUDL_MATCHED_PATH.exists():
        print("Añadiendo Hudl...")

        hudl = pd.read_parquet(HUDL_MATCHED_PATH).copy()

        hudl = hudl[
            [
                "Player ID",
                "hudl_specific_position",
                "hudl_age",
                "hudl_market_value",
            ]
        ].drop_duplicates(subset=["Player ID"])

        metadata = metadata.merge(hudl, on="Player ID", how="left")

        hudl_mask = metadata["position_primary"].isna() & metadata["hudl_specific_position"].notna()

        metadata.loc[hudl_mask, "position_raw"] = metadata.loc[hudl_mask, "hudl_specific_position"]
        metadata.loc[hudl_mask, "position_primary"] = metadata.loc[hudl_mask, "hudl_specific_position"].apply(
            lambda x: normalize_primary_position(first_hudl_position(x))
        )
        metadata.loc[hudl_mask, "metadata_source"] = "hudl"

        # edad: si falta y Hudl la tiene
        age_mask = metadata["age"].isna() & metadata["hudl_age"].notna()
        metadata.loc[age_mask, "age"] = metadata.loc[age_mask, "hudl_age"]

        # market value: si Hudl lo tiene, lo metemos
        mv_mask = metadata["hudl_market_value"].notna()
        metadata.loc[mv_mask, "market_value"] = metadata.loc[mv_mask, "hudl_market_value"]

        metadata = metadata.drop(columns=["hudl_specific_position", "hudl_age", "hudl_market_value"])

    # --------------------------------------------------
    # 3. MANUAL (sobrescribe todo)
    # --------------------------------------------------
    if MANUAL_POSITIONS_PATH.exists():
        print("Añadiendo manual...")

        manual = pd.read_csv(MANUAL_POSITIONS_PATH, encoding="latin1").copy()

        manual = manual[
            ["player_name", "position"]
        ].drop_duplicates(subset=["player_name"])

        metadata = metadata.merge(
            manual,
            on="player_name",
            how="left"
        )

        manual_mask = metadata["position"].notna()

        metadata.loc[manual_mask, "position_raw"] = metadata.loc[manual_mask, "position"]
        metadata.loc[manual_mask, "position_primary"] = metadata.loc[manual_mask, "position"].apply(
            normalize_primary_position
        )
        metadata.loc[manual_mask, "metadata_source"] = "manual"

        metadata = metadata.drop(columns=["position"])

    # --------------------------------------------------
    # 4. POSITION GROUP
    # --------------------------------------------------
    metadata["position_group"] = metadata["position_primary"].apply(build_position_group)

    # --------------------------------------------------
    # 5. LIMPIEZA FINAL
    # --------------------------------------------------
    metadata["age"] = pd.to_numeric(metadata["age"], errors="coerce")
    metadata["market_value"] = pd.to_numeric(metadata["market_value"], errors="coerce")

    metadata = metadata.sort_values(
        ["league", "team_name", "player_name"]
    ).reset_index(drop=True)

    OUTPUT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(OUTPUT_METADATA_PATH, index=False)

    print("\nMetadata guardada en:")
    print(OUTPUT_METADATA_PATH)

    print("\nResumen metadata_source:")
    print(metadata["metadata_source"].value_counts(dropna=False))

    print("\nResumen position_group:")
    print(metadata["position_group"].value_counts(dropna=False))

    print("\nJugadores totales:", len(metadata))


if __name__ == "__main__":
    build_player_metadata()