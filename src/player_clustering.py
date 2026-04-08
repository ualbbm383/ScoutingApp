from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# CONFIGURACIÓN GENERAL
# ==================================================
ENRICHED_PARQUET_PATH = Path("event_data/processed/player_metrics_enriched.parquet")
PLAYER_METADATA_MASTER_PATH = Path("data/player_metadata_master.parquet")

POSITION_CONFIG = {
    "Midfielder": {
        "features": [
            "passes_90",
            "progressive_passes_90",
            "progressive_passes_final_third_90",
            "key_passes_90",
            "big_chances_created_90",
            "takeons_90",
            "recoveries_90",
            "interceptions_90",
            "tackles_90",
            "aerials_90",
            "aerial_win_pct",
            "shots_90",
            "avg_progressive_distance_m",
        ],
        "n_components": 3,
        "profile_names": {
            0: "Físico / Defensivo",
            1: "Ofensivo / Mediapunta",
            2: "Organizador",
        },
    },
    "Center Back": {
        "features": [
            "passes_90",
            "progressive_pass_pct",
            "long_pass_pct",
            "avg_pass_length",
            "interceptions_90",
            "clearances_90",
            "blocked_passes_90",
            "tackles_def_third_90",
            "tackles_mid_third_90",
            "recoveries_def_third_90",
            "recoveries_mid_third_90",
            "aerials_90",
            "aerial_win_pct",
            "fouls_90",
        ],
        "n_components": 3,
        "profile_names": {
            0: "Central directo / Bloque Bajo",
            1: "Central de posesión / Línea Alta",
            2: "Destructor",
        },
    },
    "Striker": {
        "features": [
            "shots_90",
            "inside_box_shot_pct",
            "avg_shot_distance_m",
            "aerials_90",
            "aerial_win_pct",
            "takeons_90",
            "dispossessed_90",
            "key_passes_90",
            "big_chances_created_90",
            "progressive_passes_90",
        ],
        "n_components": 3,
        "profile_names": {
            0: "Delantero referencia",
            1: "Delantero móvil",
            2: "Creativo / Falso 9",
        },
    },
    "Winger": {
        "features": [
            "shots_90",
            "takeons_90",
            "takeon_success_pct",
            "dispossessed_90",
            "key_passes_90",
            "big_chances_created_90",
            "crosses_90",
            "cross_accuracy",
            "progressive_passes_90",
            "recoveries_final_third_90",
        ],
        "n_components": 3,
        "profile_names": {
            0: "Regateador",
            1: "Equilibrado",
            2: "Creativo / Centrador",
        },
    },
    "Full Back": {
        "features": [
            "progressive_pass_pct",
            "aerials_90",
            "fouls_90",
            "key_passes_90",
            "crosses_90",
            "big_chances_created_90",
            "takeons_90",
            "tackles_90",
            "interceptions_90",
            "clearances_90",
            "recoveries_final_third_90",
        ],
        "n_components": 2,
        "profile_names": {
            0: "Defensivo",
            1: "Ofensivo",
        },
    },
}


PLAYER_METRIC_LABELS = {
    "minutes_total": "Minutos",
    "passes_90": "Pases/90",
    "successful_passes_90": "Pases exitosos/90",
    "short_passes_90": "Pases cortos/90",
    "medium_passes_90": "Pases medios/90",
    "long_passes_90": "Pases largos/90",
    "forward_passes_90": "Pases hacia delante/90",
    "backward_passes_90": "Pases hacia atrás/90",
    "lateral_passes_90": "Pases laterales/90",
    "passes_final_third_90": "Pases último tercio/90",
    "passes_final_third_pct": "% pases último tercio",
    "pass_accuracy": "% éxito pase",
    "short_pass_accuracy": "% éxito pase corto",
    "medium_pass_accuracy": "% éxito pase medio",
    "long_pass_accuracy": "% éxito pase largo",
    "forward_pass_accuracy": "% éxito pase hacia delante",
    "backward_pass_accuracy": "% éxito pase hacia atrás",
    "lateral_pass_accuracy": "% éxito pase lateral",
    "progressive_passes_90": "Pases progresivos/90",
    "progressive_passes_final_third_90": "Pases prog. últ. tercio/90",
    "progressive_pass_pct": "% pases progresivos",
    "key_passes_90": "Pases clave/90",
    "crosses_90": "Centros/90",
    "cross_accuracy": "% éxito centros",
    "shots_90": "Tiros/90",
    "goals_90": "Goles/90",
    "shots_on_target_90": "Tiros a puerta/90",
    "big_chances_90": "Ocasiones claras/90",
    "big_chances_created_90": "Ocasiones claras creadas/90",
    "shot_accuracy": "% precisión tiro",
    "goal_conversion": "% conversión gol",
    "inside_box_shot_pct": "% tiros dentro del área",
    "shots_inside_box_90": "Tiros dentro del área/90",
    "shots_outside_box_90": "Tiros fuera del área/90",
    "avg_shot_distance_m": "Distancia media de tiro",
    "recoveries_90": "Recuperaciones/90",
    "recoveries_def_third_90": "Recuperaciones tercio defensivo/90",
    "recoveries_mid_third_90": "Recuperaciones tercio medio/90",
    "recoveries_final_third_90": "Recuperaciones tercio final/90",
    "tackles_90": "Entradas/90",
    "successful_tackles_90": "Entradas exitosas/90",
    "tackle_success_pct": "% éxito entradas",
    "tackles_def_third_90": "Entradas tercio defensivo/90",
    "tackles_mid_third_90": "Entradas tercio medio/90",
    "tackles_final_third_90": "Entradas último tercio/90",
    "interceptions_90": "Intercepciones/90",
    "clearances_90": "Despejes/90",
    "aerials_90": "Duelos aéreos/90",
    "successful_aerials_90": "Duelos aéreos exitosos/90",
    "aerial_win_pct": "% éxito aéreo",
    "blocked_passes_90": "Pases bloqueados/90",
    "fouls_90": "Faltas/90",
    "takeons_90": "Regates/90",
    "successful_takeons_90": "Regates exitosos/90",
    "takeon_success_pct": "% éxito regate",
    "dispossessed_90": "Pérdidas/90",
    "long_pass_pct": "% pases largos",
    "avg_pass_length": "Distancia media de pase",
    "avg_progressive_distance_m": "Distancia progresiva media",
}


# ==================================================
# HELPERS
# ==================================================
def format_metric(metric: str) -> str:
    return PLAYER_METRIC_LABELS.get(metric, metric)


def get_supported_positions():
    return sorted(POSITION_CONFIG.keys())


def get_position_config(position_group: str) -> dict:
    if position_group not in POSITION_CONFIG:
        raise ValueError(f"Position '{position_group}' no está configurada.")
    return POSITION_CONFIG[position_group]


def prepare_position_dataframe(
    df: pd.DataFrame,
    position_group: str,
    min_minutes: int = 600,
) -> pd.DataFrame:
    cfg = get_position_config(position_group)
    features = cfg["features"]

    df_pos = df[df["position_group"] == position_group].copy()
    df_pos = df_pos[df_pos["minutes_total"] >= min_minutes].copy()

    required_cols = [
        "player_name",
        "team_name",
        "league",
        "season",
        "minutes_total",
        "position_group",
    ]
    keep_cols = required_cols + [c for c in features if c in df_pos.columns]

    df_pos = df_pos[keep_cols].copy()

    missing_features = [f for f in features if f not in df_pos.columns]
    if missing_features:
        raise ValueError(f"Faltan features en el dataframe: {missing_features}")

    return df_pos


def scale_features(df_pos: pd.DataFrame, features: list[str]):
    X = df_pos[features].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler


def build_style_mix_from_kmeans_distances(distances: np.ndarray, alpha: float = 2.5) -> np.ndarray:
    """
    Convierte distancias a centroides en una mezcla tipo probabilidad.
    Cuanto mayor es alpha, más se potencia el cluster principal.
    """
    inv_dist = 1 / (distances + 1e-6)
    weighted = inv_dist ** alpha
    style_mix = weighted / weighted.sum(axis=1, keepdims=True)
    return style_mix


def infer_profile_names(position_group: str, cluster_profile: pd.DataFrame, fallback_names: dict) -> dict:
    cluster_ids = cluster_profile.index.tolist()
    assigned = {}
    remaining = set(cluster_ids)

    # --------------------------------------------------
    # MIDFIELDER
    # --------------------------------------------------
    if position_group == "Midfielder":
        defensive_score = (
            cluster_profile["recoveries_90"]
            + cluster_profile["interceptions_90"]
            + cluster_profile["tackles_90"]
            + cluster_profile["aerials_90"]
        )
        defensive_id = defensive_score.idxmax()
        assigned[defensive_id] = "Físico / Defensivo"
        remaining.remove(defensive_id)

        attacking_score = (
            cluster_profile.loc[list(remaining), "shots_90"]
            + cluster_profile.loc[list(remaining), "big_chances_created_90"]
            + cluster_profile.loc[list(remaining), "takeons_90"]
            + cluster_profile.loc[list(remaining), "key_passes_90"]
        )
        attacking_id = attacking_score.idxmax()
        assigned[attacking_id] = "Ofensivo / Mediapunta"
        remaining.remove(attacking_id)

        organizer_id = list(remaining)[0]
        assigned[organizer_id] = "Organizador"
        return assigned

    # --------------------------------------------------
    # CENTER BACK
    # --------------------------------------------------
    if position_group == "Center Back":
        destroyer_score = (
            cluster_profile["clearances_90"]
            + cluster_profile["aerial_win_pct"]
            + cluster_profile["aerials_90"]
            + cluster_profile["avg_pass_length"]
        )
        destroyer_id = destroyer_score.idxmax()
        assigned[destroyer_id] = "Destructor"
        remaining.remove(destroyer_id)

        direct_score = (
            cluster_profile.loc[list(remaining), "long_pass_pct"]
            + cluster_profile.loc[list(remaining), "progressive_pass_pct"]
            + cluster_profile.loc[list(remaining), "tackles_def_third_90"]
        )
        direct_id = direct_score.idxmax()
        assigned[direct_id] = "Central directo / Bloque Bajo"
        remaining.remove(direct_id)

        possession_id = list(remaining)[0]
        assigned[possession_id] = "Central de posesión / Línea Alta"
        return assigned

    # --------------------------------------------------
    # STRIKER
    # --------------------------------------------------
    if position_group == "Striker":
        target_score = (
            cluster_profile["aerials_90"]
            + cluster_profile["aerial_win_pct"]
            + cluster_profile["inside_box_shot_pct"]
        )
        target_id = target_score.idxmax()
        assigned[target_id] = "Delantero referencia"
        remaining.remove(target_id)

        creative_score = (
            cluster_profile.loc[list(remaining), "key_passes_90"]
            + cluster_profile.loc[list(remaining), "big_chances_created_90"]
            + cluster_profile.loc[list(remaining), "progressive_passes_90"]
        )
        creative_id = creative_score.idxmax()
        assigned[creative_id] = "Creativo / Falso 9"
        remaining.remove(creative_id)

        mobile_id = list(remaining)[0]
        assigned[mobile_id] = "Delantero móvil"
        return assigned

    # --------------------------------------------------
    # WINGER
    # --------------------------------------------------
    if position_group == "Winger":
        dribbler_score = (
            cluster_profile["takeons_90"]
            + cluster_profile["takeon_success_pct"]
        )
        dribbler_id = dribbler_score.idxmax()
        assigned[dribbler_id] = "Regateador"
        remaining.remove(dribbler_id)

        creator_score = (
            cluster_profile.loc[list(remaining), "crosses_90"]
            + cluster_profile.loc[list(remaining), "key_passes_90"]
            + cluster_profile.loc[list(remaining), "big_chances_created_90"]
        )
        creator_id = creator_score.idxmax()
        assigned[creator_id] = "Creativo / Centrador"
        remaining.remove(creator_id)

        balanced_id = list(remaining)[0]
        assigned[balanced_id] = "Equilibrado"
        return assigned

    # --------------------------------------------------
    # FULL BACK
    # --------------------------------------------------
    if position_group == "Full Back":
        defensive_score = (
            cluster_profile["tackles_90"]
            + cluster_profile["interceptions_90"]
            + cluster_profile["clearances_90"]
        )
        defensive_id = defensive_score.idxmax()
        offensive_id = [c for c in cluster_ids if c != defensive_id][0]

        return {
            defensive_id: "Defensivo",
            offensive_id: "Ofensivo",
        }

    return {cid: fallback_names.get(cid, f"Cluster {cid}") for cid in cluster_ids}


def fit_kmeans_for_position(
    df: pd.DataFrame,
    position_group: str,
    min_minutes: int = 600,
    random_state: int = 42,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    alpha: float = 2.5,
):
    cfg = get_position_config(position_group)
    features = cfg["features"]
    n_components = cfg["n_components"]
    fallback_profile_names = cfg["profile_names"]

    df_pos = prepare_position_dataframe(df, position_group, min_minutes=min_minutes).reset_index(drop=True)

    if df_pos.empty:
        raise ValueError("No hay jugadores suficientes para esa posición con el filtro actual.")

    X, X_scaled, scaler = scale_features(df_pos, features)

    kmeans = KMeans(
        n_clusters=n_components,
        random_state=random_state,
        n_init=20,
    )
    clusters = kmeans.fit_predict(X_scaled)
    df_pos["cluster"] = clusters

    # --------------------------------------------
    # TARTA = DISTANCIA A CENTROIDES
    # --------------------------------------------
    distances = kmeans.transform(X_scaled)
    style_mix = build_style_mix_from_kmeans_distances(distances, alpha=alpha)

    for i in range(n_components):
        # mantenemos ambos nombres por compatibilidad con lo que ya usa la app
        df_pos[f"gmm_profile_{i+1}_pct"] = style_mix[:, i]
        df_pos[f"profile_{i+1}_pct"] = style_mix[:, i]

    # --------------------------------------------
    # UMAP
    # --------------------------------------------
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_scaled)

    df_pos["umap_x"] = embedding[:, 0]
    df_pos["umap_y"] = embedding[:, 1]

    # --------------------------------------------
    # PERFIL MEDIO DE CLUSTER
    # --------------------------------------------
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    X_scaled_df["cluster"] = df_pos["cluster"].values

    cluster_profile = (
        X_scaled_df.groupby("cluster")[features]
        .mean()
        .round(2)
    )

    # --------------------------------------------
    # INFERIR NOMBRES DE PERFIL AUTOMÁTICAMENTE
    # --------------------------------------------
    profile_names = infer_profile_names(
        position_group=position_group,
        cluster_profile=cluster_profile,
        fallback_names=fallback_profile_names,
    )

    df_pos["cluster_name"] = df_pos["cluster"].map(profile_names)

    return {
        "df_position": df_pos,
        "features": features,
        "kmeans": kmeans,
        "scaler": scaler,
        "cluster_profile": cluster_profile,
        "profile_names": profile_names,
        "position_group": position_group,
        "min_minutes": min_minutes,
    }


def build_cluster_summary(cluster_profile: pd.DataFrame, profile_names: dict, top_n: int = 5) -> dict:
    summary = {}

    for cluster_id in sorted(cluster_profile.index):
        row = cluster_profile.loc[cluster_id].sort_values(ascending=False)

        cluster_name = profile_names.get(cluster_id, f"Cluster {cluster_id}")

        summary[cluster_name] = {
            "cluster_id": int(cluster_id),
            "top": [(metric, float(value)) for metric, value in row.head(top_n).items()],
            "bottom": [(metric, float(value)) for metric, value in row.sort_values(ascending=True).head(top_n).items()],
        }

    return summary


def build_cluster_summary_tables(cluster_summary: dict):
    rows = []

    for cluster_name, info in cluster_summary.items():
        top_metrics = ", ".join([f"{format_metric(m)} ({v:+.2f})" for m, v in info["top"]])
        bottom_metrics = ", ".join([f"{format_metric(m)} ({v:+.2f})" for m, v in info["bottom"]])

        rows.append(
            {
                "Cluster": info["cluster_id"],
                "Style": cluster_name,
                "Strongest metrics": top_metrics,
                "Weakest metrics": bottom_metrics,
            }
        )

    return pd.DataFrame(rows)


def build_single_cluster_metric_tables(cluster_summary: dict, cluster_name: str):
    info = cluster_summary.get(cluster_name)
    if info is None:
        return None, None

    top_df = pd.DataFrame(
        {
            "Metric": [format_metric(m) for m, _ in info["top"]],
        }
    )

    bottom_df = pd.DataFrame(
        {
            "Metric": [format_metric(m) for m, _ in info["bottom"]],
        }
    )

    return top_df, bottom_df


def plot_umap_scatter(df_pos: pd.DataFrame, position_group: str):
    fig = px.scatter(
        df_pos,
        x="umap_x",
        y="umap_y",
        color="cluster_name",
        hover_data=[
            "player_name",
            "team_name",
            "league",
            "minutes_total",
        ],
        title=f"{position_group} clustering map",
    )

    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(height=650)
    return fig


def plot_umap_with_highlight(df_pos: pd.DataFrame, position_group: str, player_label: str | None = None):
    fig = px.scatter(
        df_pos,
        x="umap_x",
        y="umap_y",
        color="cluster_name",
        hover_data=[
            "player_name",
            "team_name",
            "league",
            "minutes_total",
        ],
        title=f"{position_group} clustering map",
    )

    fig.update_traces(marker=dict(size=8, opacity=0.65))

    if player_label:
        df_highlight = df_pos[df_pos["player_label"] == player_label].copy()
        if not df_highlight.empty:
            fig.add_scatter(
                x=df_highlight["umap_x"],
                y=df_highlight["umap_y"],
                mode="markers+text",
                text=df_highlight["player_name"],
                textposition="top center",
                marker=dict(size=18, symbol="star"),
                name="Selected player",
            )

    fig.update_layout(height=650)
    return fig


def plot_profile_pie(player_row: pd.Series, profile_names: dict):
    labels = []
    values = []

    for cluster_id in sorted(profile_names.keys()):
        col = f"profile_{cluster_id+1}_pct"
        if col in player_row.index and pd.notna(player_row[col]):
            labels.append(profile_names[cluster_id])
            values.append(player_row[col])

    if not values:
        return None

    fig = px.pie(
        values=values,
        names=labels,
        title=f"Role distribution - {player_row['player_name']}",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=420)
    return fig


def add_player_label(df_pos: pd.DataFrame) -> pd.DataFrame:
    df_pos = df_pos.copy()
    df_pos["player_label"] = df_pos["player_name"] + " | " + df_pos["team_name"]
    return df_pos


def get_player_row(df_pos: pd.DataFrame, player_label: str):
    res = df_pos[df_pos["player_label"] == player_label]
    if res.empty:
        return None
    return res.iloc[0]


def merge_clustering_results_back(
    df_full: pd.DataFrame,
    df_pos_clustered: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = ["player_name", "team_name", "season", "position_group"]

    cluster_cols = [
        "cluster",
        "cluster_name",
        "umap_x",
        "umap_y",
    ]

    prob_cols = [
        c for c in df_pos_clustered.columns
        if (c.startswith("profile_") or c.startswith("gmm_profile_")) and c.endswith("_pct")
    ]

    cluster_cols += prob_cols

    cluster_export = df_pos_clustered[merge_keys + cluster_cols].copy()

    df_out = df_full.copy()

    for col in cluster_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan

    df_out = df_out.set_index(merge_keys)
    cluster_export = cluster_export.set_index(merge_keys)

    common_idx = df_out.index.intersection(cluster_export.index)

    for col in cluster_cols:
        df_out.loc[common_idx, col] = cluster_export.loc[common_idx, col]

    df_out = df_out.reset_index()

    return df_out


def save_updated_enriched_parquet(df_updated: pd.DataFrame, parquet_path: Path = ENRICHED_PARQUET_PATH):
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_updated.to_parquet(parquet_path, index=False)


def recalculate_and_update_position(
    df_full: pd.DataFrame,
    position_group: str,
    min_minutes: int = 600,
    parquet_path: Path = ENRICHED_PARQUET_PATH,
):
    result = fit_kmeans_for_position(
        df=df_full,
        position_group=position_group,
        min_minutes=min_minutes,
    )

    df_updated = merge_clustering_results_back(df_full, result["df_position"])
    save_updated_enriched_parquet(df_updated, parquet_path=parquet_path)

    return result, df_updated


# Función para la actualización manual de posición para un jugador determinado que no tiene posición o la tiene mal registrada.

def build_player_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["player_label"] = (
        df["player_name"].astype(str)
        + " | "
        + df["team_name"].astype(str)
        + " | "
        + df["season"].astype(str)
    )
    return df


def update_player_position(
    player_label: str,
    new_position_group: str,
    metadata_path: Path = PLAYER_METADATA_MASTER_PATH,
    enriched_path: Path = ENRICHED_PARQUET_PATH,
) -> pd.DataFrame:
    if not enriched_path.exists():
        raise FileNotFoundError(f"No se encontró el parquet enriquecido: {enriched_path}")

    # Cargamos enriched para localizar bien al jugador
    df_enriched = pd.read_parquet(enriched_path).copy()
    df_enriched = build_player_label(df_enriched)

    mask_player = df_enriched["player_label"] == player_label

    if not mask_player.any():
        raise ValueError("No se ha encontrado el jugador seleccionado en el parquet enriquecido.")

    player_row = df_enriched.loc[mask_player].iloc[0]

    # Si no existe metadata master, creamos uno vacío con columnas base
    if metadata_path.exists():
        df_meta = pd.read_parquet(metadata_path).copy()
    else:
        df_meta = pd.DataFrame(columns=[
            "Player ID",
            "player_name",
            "team_name",
            "league",
            "season",
            "position_raw",
            "position_primary",
            "position_group",
            "age",
            "market_value",
            "metadata_source",
        ])

    merge_keys = ["Player ID", "player_name", "team_name", "league", "season"]

    # Si falta alguna columna en metadata, la creamos
    required_cols = [
        "Player ID",
        "player_name",
        "team_name",
        "league",
        "season",
        "position_raw",
        "position_primary",
        "position_group",
        "age",
        "market_value",
        "metadata_source",
    ]
    for col in required_cols:
        if col not in df_meta.columns:
            df_meta[col] = pd.NA

    # Buscar si el jugador ya existe en metadata
    mask_meta = pd.Series(True, index=df_meta.index)
    for key in merge_keys:
        mask_meta &= df_meta[key].astype(str) == str(player_row.get(key))

    if mask_meta.any():
        # Actualizamos fila existente
        df_meta.loc[mask_meta, "position_group"] = new_position_group
        df_meta.loc[mask_meta, "position_primary"] = new_position_group
        df_meta.loc[mask_meta, "metadata_source"] = "manual"
    else:
        # Añadimos fila nueva a metadata master
        new_row = {
            "Player ID": player_row.get("Player ID"),
            "player_name": player_row.get("player_name"),
            "team_name": player_row.get("team_name"),
            "league": player_row.get("league"),
            "season": player_row.get("season"),
            "position_raw": player_row.get("position_raw"),
            "position_primary": new_position_group,
            "position_group": new_position_group,
            "age": player_row.get("age"),
            "market_value": player_row.get("market_value"),
            "metadata_source": "manual",
        }

        df_meta = pd.concat([df_meta, pd.DataFrame([new_row])], ignore_index=True)

    # Limpiar duplicados por si hubiera más de una fila de metadata para el mismo jugador
    df_meta = df_meta.drop_duplicates(subset=merge_keys, keep="last").copy()

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_parquet(metadata_path, index=False)

    return df_meta