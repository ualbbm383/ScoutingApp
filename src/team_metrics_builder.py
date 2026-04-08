from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLS = ["Team ID", "team_name", "league", "season"]

def build_team_ppda(df: pd.DataFrame, press_line_x: float = 40.0) -> pd.DataFrame:
    """
    Calcula una aproximación de PPDA por equipo y temporada.

    PPDA = pases permitidos al rival en zona de presión / acciones defensivas propias en esa zona

    press_line_x:
        Línea de presión en coordenada X (0-100). 
        40.0 significa que contamos acciones en el 60% más ofensivo del campo rival.
    """
    df = df.copy()

    needed_cols = ["matchId", "Team ID", "team_name", "league", "season", "Event Type", "Start X"]
    existing_cols = [c for c in needed_cols if c in df.columns]
    work = df[existing_cols].copy()

    work["Start X"] = pd.to_numeric(work["Start X"], errors="coerce")
    work = work.dropna(subset=["matchId", "Team ID", "team_name", "league", "season", "Event Type", "Start X"])

    # ----------------------------------------------
    # 1. PASSES EN ZONA DE PRESIÓN
    # ----------------------------------------------
    passes = work[work["Event Type"] == "Pass"].copy()

    # asumimos coordenadas normalizadas 0-100 atacando siempre hacia la derecha
    passes_high = passes[passes["Start X"] >= press_line_x].copy()

    opp_passes = (
        passes_high.groupby(["matchId", "Team ID"], as_index=False)
        .size()
        .rename(columns={"size": "opponent_passes_high", "Team ID": "opponent_team_id"})
    )

    # ----------------------------------------------
    # 2. ACCIONES DEFENSIVAS EN ZONA DE PRESIÓN
    # ----------------------------------------------
    def_actions = work[
        work["Event Type"].isin(["Tackle", "Interception", "Foul", "BallRecovery", "BlockedPass"])
    ].copy()

    def_actions_high = def_actions[def_actions["Start X"] >= press_line_x].copy()

    team_def_actions = (
        def_actions_high.groupby(
            ["matchId", "Team ID", "team_name", "league", "season"], as_index=False
        )
        .size()
        .rename(columns={"size": "def_actions_high", "Team ID": "def_team_id"})
    )

    # ----------------------------------------------
    # 3. MAPA DE EQUIPOS POR PARTIDO
    # ----------------------------------------------
    teams_per_match = (
        work[["matchId", "Team ID", "team_name", "league", "season"]]
        .drop_duplicates()
        .copy()
    )

    match_pairs = teams_per_match.merge(
        teams_per_match,
        on="matchId",
        suffixes=("_team", "_opp")
    )

    match_pairs = match_pairs[match_pairs["Team ID_team"] != match_pairs["Team ID_opp"]].copy()

    # nos quedamos con la perspectiva del equipo defensor
    match_pairs = match_pairs.rename(columns={
        "Team ID_team": "def_team_id",
        "team_name_team": "team_name",
        "league_team": "league",
        "season_team": "season",
        "Team ID_opp": "opponent_team_id"
    })

    match_pairs = match_pairs[
        ["matchId", "def_team_id", "team_name", "league", "season", "opponent_team_id"]
    ].drop_duplicates()

    # ----------------------------------------------
    # 4. JUNTAR PASSES RIVALES + ACCIONES DEFENSIVAS
    # ----------------------------------------------
    ppda_match = match_pairs.merge(
        opp_passes,
        on=["matchId", "opponent_team_id"],
        how="left"
    )

    ppda_match = ppda_match.merge(
        team_def_actions,
        on=["matchId", "def_team_id", "team_name", "league", "season"],
        how="left"
    )

    ppda_match["opponent_passes_high"] = ppda_match["opponent_passes_high"].fillna(0)
    ppda_match["def_actions_high"] = ppda_match["def_actions_high"].fillna(0)

    ppda_match["ppda_match"] = np.where(
        ppda_match["def_actions_high"] > 0,
        ppda_match["opponent_passes_high"] / ppda_match["def_actions_high"],
        np.nan
    )

    # ----------------------------------------------
    # 5. AGREGAR A NIVEL EQUIPO-TEMPORADA
    # ----------------------------------------------
    ppda_team = (
        ppda_match.groupby(["def_team_id", "team_name", "league", "season"], as_index=False)
        .agg(
            opponent_passes_high_match=("opponent_passes_high", "mean"),
            def_actions_high_match=("def_actions_high", "mean"),
            ppda=("ppda_match", "mean")
        )
        .rename(columns={"def_team_id": "Team ID"})
    )

    return ppda_team




def build_team_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ==================================================
    # 1. LIMPIEZA DE TIPOS
    # ==================================================
    numeric_cols = [
        "Event Value",
        "Minuto",
        "Secondo",
        "Team ID",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
        "PassEndX",
        "PassEndY",
        "Length",
        "Angle",
        "GoalMouthY",
        "GoalMouthZ",
        "BlockedX",
        "BlockedY",
    ]

    numeric_cols = [col for col in numeric_cols if col in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==================================================
    # 2. PARTIDOS JUGADOS
    # ==================================================
    teams_match = (
        df[["matchId", "match_date", "league", "season", "Team ID", "team_name"]]
        .dropna(subset=["Team ID", "team_name"])
        .drop_duplicates()
        .copy()
    )

    team_matches = (
        teams_match.groupby(GROUP_COLS, as_index=False)["matchId"]
        .nunique()
        .rename(columns={"matchId": "matches_played"})
        .sort_values("matches_played", ascending=False)
        .reset_index(drop=True)
    )

    # ==================================================
    # 3. MÉTRICAS DE PASE
    # ==================================================
    passes = df[df["Event Type"] == "Pass"].copy()

    pass_num_cols = ["Start X", "Start Y", "PassEndX", "PassEndY", "Length", "Angle"]
    for col in pass_num_cols:
        passes[col] = pd.to_numeric(passes[col], errors="coerce")

    passes = passes.dropna(
        subset=["Start X", "PassEndX", "Start Y", "PassEndY", "Length", "Angle"]
    )

    passes["dx"] = passes["PassEndX"] - passes["Start X"]
    passes["dx_m"] = passes["dx"] * 105 / 100
    passes["successful"] = passes["Outcome"] == "Successful"
    passes["angle_deg"] = np.degrees(passes["Angle"]) % 360

    passes["short_pass"] = passes["Length"] < 15
    passes["medium_pass"] = passes["Length"].between(15, 30, inclusive="both")
    passes["long_pass"] = passes["Length"] > 30

    passes["successful_short_pass"] = passes["short_pass"] & passes["successful"]
    passes["successful_medium_pass"] = passes["medium_pass"] & passes["successful"]
    passes["successful_long_pass"] = passes["long_pass"] & passes["successful"]

    passes["forward_pass"] = (
        (passes["angle_deg"] >= 300) | (passes["angle_deg"] <= 60)
    )
    passes["backward_pass"] = (
        (passes["angle_deg"] >= 120) & (passes["angle_deg"] <= 240)
    )
    passes["lateral_pass"] = ~passes["forward_pass"] & ~passes["backward_pass"]

    passes["successful_forward_pass"] = passes["forward_pass"] & passes["successful"]
    passes["successful_backward_pass"] = passes["backward_pass"] & passes["successful"]
    passes["successful_lateral_pass"] = passes["lateral_pass"] & passes["successful"]

    passes["cross"] = passes["Cross"] == "Yes"
    passes["key_pass"] = passes["KeyPass"] == "Yes"
    passes["successful_cross"] = passes["cross"] & passes["successful"]

    passes["progressive_pass"] = np.where(
        passes["Start X"] < 50,
        passes["dx_m"] >= 20,
        passes["dx_m"] >= 10,
    )
    passes["successful_progressive_pass"] = (
        passes["progressive_pass"] & passes["successful"]
    )

    final_third_x = 66.67

    passes["progressive_pass_final_third"] = (
        passes["successful_progressive_pass"] & (passes["PassEndX"] >= final_third_x)
    )

    passes["pass_final_third"] = passes["Start X"] >= final_third_x

    team_pass_tmp = (
        passes.groupby(GROUP_COLS, as_index=False)
        .agg(
            passes=("Team ID", "size"),
            successful_passes=("successful", "sum"),
            passes_final_third=("pass_final_third", "sum"),
            short_passes=("short_pass", "sum"),
            successful_short_passes=("successful_short_pass", "sum"),
            medium_passes=("medium_pass", "sum"),
            successful_medium_passes=("successful_medium_pass", "sum"),
            long_passes=("long_pass", "sum"),
            successful_long_passes=("successful_long_pass", "sum"),
            forward_passes=("forward_pass", "sum"),
            successful_forward_passes=("successful_forward_pass", "sum"),
            backward_passes=("backward_pass", "sum"),
            successful_backward_passes=("successful_backward_pass", "sum"),
            lateral_passes=("lateral_pass", "sum"),
            successful_lateral_passes=("successful_lateral_pass", "sum"),
            crosses=("cross", "sum"),
            successful_crosses=("successful_cross", "sum"),
            key_passes=("key_pass", "sum"),
            progressive_passes=("successful_progressive_pass", "sum"),
            progressive_passes_final_third=("progressive_pass_final_third", "sum"),
            avg_pass_length=("Length", "mean"),
            avg_progressive_distance_m=("dx_m", "mean"),
        )
    )

    team_pass_metrics = team_matches[GROUP_COLS + ["matches_played"]].merge(
        team_pass_tmp,
        on=GROUP_COLS,
        how="left",
    )

    pass_tmp_cols = [
        "passes",
        "passes_final_third",
        "successful_passes",
        "short_passes",
        "successful_short_passes",
        "medium_passes",
        "successful_medium_passes",
        "long_passes",
        "successful_long_passes",
        "forward_passes",
        "successful_forward_passes",
        "backward_passes",
        "successful_backward_passes",
        "lateral_passes",
        "successful_lateral_passes",
        "crosses",
        "successful_crosses",
        "key_passes",
        "progressive_passes",
        "progressive_passes_final_third",
    ]

    team_pass_metrics[pass_tmp_cols] = team_pass_metrics[pass_tmp_cols].fillna(0)

    team_pass_metrics["pass_accuracy"] = np.where(
        team_pass_metrics["passes"] > 0,
        team_pass_metrics["successful_passes"] / team_pass_metrics["passes"] * 100,
        np.nan,
    )
    team_pass_metrics["short_pass_accuracy"] = np.where(
        team_pass_metrics["short_passes"] > 0,
        team_pass_metrics["successful_short_passes"]
        / team_pass_metrics["short_passes"]
        * 100,
        np.nan,
    )
    team_pass_metrics["medium_pass_accuracy"] = np.where(
        team_pass_metrics["medium_passes"] > 0,
        team_pass_metrics["successful_medium_passes"]
        / team_pass_metrics["medium_passes"]
        * 100,
        np.nan,
    )
    team_pass_metrics["long_pass_accuracy"] = np.where(
        team_pass_metrics["long_passes"] > 0,
        team_pass_metrics["successful_long_passes"]
        / team_pass_metrics["long_passes"]
        * 100,
        np.nan,
    )
    team_pass_metrics["forward_pass_accuracy"] = np.where(
        team_pass_metrics["forward_passes"] > 0,
        team_pass_metrics["successful_forward_passes"]
        / team_pass_metrics["forward_passes"]
        * 100,
        np.nan,
    )
    team_pass_metrics["backward_pass_accuracy"] = np.where(
        team_pass_metrics["backward_passes"] > 0,
        team_pass_metrics["successful_backward_passes"]
        / team_pass_metrics["backward_passes"]
        * 100,
        np.nan,
    )
    team_pass_metrics["lateral_pass_accuracy"] = np.where(
        team_pass_metrics["lateral_passes"] > 0,
        team_pass_metrics["successful_lateral_passes"]
        / team_pass_metrics["lateral_passes"]
        * 100,
        np.nan,
    )
    team_pass_metrics["cross_accuracy"] = np.where(
        team_pass_metrics["crosses"] > 0,
        team_pass_metrics["successful_crosses"] / team_pass_metrics["crosses"] * 100,
        np.nan,
    )

    team_pass_metrics["long_pass_pct"] = np.where(
        team_pass_metrics["passes"] > 0,
        team_pass_metrics["long_passes"] / team_pass_metrics["passes"] * 100,
        np.nan,
    )
    team_pass_metrics["progressive_pass_pct"] = np.where(
        team_pass_metrics["passes"] > 0,
        team_pass_metrics["progressive_passes"] / team_pass_metrics["passes"] * 100,
        np.nan,
    )
    team_pass_metrics["passes_final_third_pct"] = np.where(
        team_pass_metrics["passes"] > 0,
        team_pass_metrics["passes_final_third"] / team_pass_metrics["passes"] * 100,
        np.nan,
    )

    pass_per_match_cols = [
        "passes",
        "passes_final_third",
        "successful_passes",
        "short_passes",
        "medium_passes",
        "long_passes",
        "forward_passes",
        "backward_passes",
        "lateral_passes",
        "crosses",
        "key_passes",
        "progressive_passes",
        "progressive_passes_final_third",
    ]

    for col in pass_per_match_cols:
        team_pass_metrics[f"{col}_match"] = np.where(
            team_pass_metrics["matches_played"] > 0,
            team_pass_metrics[col] / team_pass_metrics["matches_played"],
            np.nan,
        )

    pass_keep_cols = (
        GROUP_COLS
        + ["matches_played"]
        + [f"{col}_match" for col in pass_per_match_cols]
        + [
            "pass_accuracy",
            "short_pass_accuracy",
            "medium_pass_accuracy",
            "long_pass_accuracy",
            "forward_pass_accuracy",
            "backward_pass_accuracy",
            "lateral_pass_accuracy",
            "cross_accuracy",
            "long_pass_pct",
            "progressive_pass_pct",
            "passes_final_third_pct",
            "avg_pass_length",
            "avg_progressive_distance_m",
        ]
    )

    team_metrics = team_pass_metrics[pass_keep_cols].copy()

    # ==================================================
    # 4. MÉTRICAS DEFENSIVAS
    # ==================================================
    def_events = df[
        df["Event Type"].isin(
            [
                "BallRecovery",
                "Tackle",
                "Interception",
                "Clearance",
                "Aerial",
                "BlockedPass",
                "Foul",
            ]
        )
    ].copy()

    def_events["Start X"] = pd.to_numeric(def_events["Start X"], errors="coerce")
    def_events = def_events.dropna(subset=["Start X"])

    def_events["recovery"] = def_events["Event Type"] == "BallRecovery"
    def_events["tackle"] = def_events["Event Type"] == "Tackle"
    def_events["interception"] = def_events["Event Type"] == "Interception"
    def_events["clearance"] = def_events["Event Type"] == "Clearance"
    def_events["aerial"] = def_events["Event Type"] == "Aerial"
    def_events["blocked_pass"] = def_events["Event Type"] == "BlockedPass"
    def_events["foul"] = def_events["Event Type"] == "Foul"

    def_events["successful_tackle"] = (
        (def_events["Event Type"] == "Tackle")
        & (def_events["Outcome"] == "Successful")
    )
    def_events["successful_aerial"] = (
        (def_events["Event Type"] == "Aerial")
        & (def_events["Outcome"] == "Successful")
    )

    def_events["def_third"] = def_events["Start X"] < 33
    def_events["mid_third"] = (def_events["Start X"] >= 33) & (
        def_events["Start X"] < 66
    )
    def_events["final_third"] = def_events["Start X"] >= 66

    def_events["recovery_def_third"] = def_events["recovery"] & def_events["def_third"]
    def_events["recovery_mid_third"] = def_events["recovery"] & def_events["mid_third"]
    def_events["recovery_final_third"] = (
        def_events["recovery"] & def_events["final_third"]
    )

    def_events["tackle_def_third"] = def_events["tackle"] & def_events["def_third"]
    def_events["tackle_mid_third"] = def_events["tackle"] & def_events["mid_third"]
    def_events["tackle_final_third"] = def_events["tackle"] & def_events["final_third"]

    team_def_tmp = (
        def_events.groupby(GROUP_COLS, as_index=False)
        .agg(
            recoveries=("recovery", "sum"),
            tackles=("tackle", "sum"),
            successful_tackles=("successful_tackle", "sum"),
            interceptions=("interception", "sum"),
            clearances=("clearance", "sum"),
            aerials=("aerial", "sum"),
            successful_aerials=("successful_aerial", "sum"),
            blocked_passes=("blocked_pass", "sum"),
            fouls=("foul", "sum"),
            recoveries_def_third=("recovery_def_third", "sum"),
            recoveries_mid_third=("recovery_mid_third", "sum"),
            recoveries_final_third=("recovery_final_third", "sum"),
            tackles_def_third=("tackle_def_third", "sum"),
            tackles_mid_third=("tackle_mid_third", "sum"),
            tackles_final_third=("tackle_final_third", "sum"),
        )
    )

    team_def_metrics = team_matches[GROUP_COLS + ["matches_played"]].merge(
        team_def_tmp,
        on=GROUP_COLS,
        how="left",
    )

    def_tmp_cols = [
        "recoveries",
        "tackles",
        "successful_tackles",
        "interceptions",
        "clearances",
        "aerials",
        "successful_aerials",
        "blocked_passes",
        "fouls",
        "recoveries_def_third",
        "recoveries_mid_third",
        "recoveries_final_third",
        "tackles_def_third",
        "tackles_mid_third",
        "tackles_final_third",
    ]

    team_def_metrics[def_tmp_cols] = team_def_metrics[def_tmp_cols].fillna(0)

    for col in def_tmp_cols:
        team_def_metrics[f"{col}_match"] = np.where(
            team_def_metrics["matches_played"] > 0,
            team_def_metrics[col] / team_def_metrics["matches_played"],
            np.nan,
        )

    team_def_metrics["tackle_success_pct"] = np.where(
        team_def_metrics["tackles"] > 0,
        team_def_metrics["successful_tackles"] / team_def_metrics["tackles"] * 100,
        np.nan,
    )

    team_def_metrics["aerial_win_pct"] = np.where(
        team_def_metrics["aerials"] > 0,
        team_def_metrics["successful_aerials"] / team_def_metrics["aerials"] * 100,
        np.nan,
    )

    team_def_metrics["recoveries_def_third_pct"] = np.where(
        team_def_metrics["recoveries"] > 0,
        team_def_metrics["recoveries_def_third"] / team_def_metrics["recoveries"] * 100,
        np.nan,
    )
    team_def_metrics["recoveries_mid_third_pct"] = np.where(
        team_def_metrics["recoveries"] > 0,
        team_def_metrics["recoveries_mid_third"] / team_def_metrics["recoveries"] * 100,
        np.nan,
    )
    team_def_metrics["recoveries_final_third_pct"] = np.where(
        team_def_metrics["recoveries"] > 0,
        team_def_metrics["recoveries_final_third"] / team_def_metrics["recoveries"] * 100,
        np.nan,
    )

    team_def_metrics["tackles_def_third_pct"] = np.where(
        team_def_metrics["tackles"] > 0,
        team_def_metrics["tackles_def_third"] / team_def_metrics["tackles"] * 100,
        np.nan,
    )
    team_def_metrics["tackles_mid_third_pct"] = np.where(
        team_def_metrics["tackles"] > 0,
        team_def_metrics["tackles_mid_third"] / team_def_metrics["tackles"] * 100,
        np.nan,
    )
    team_def_metrics["tackles_final_third_pct"] = np.where(
        team_def_metrics["tackles"] > 0,
        team_def_metrics["tackles_final_third"] / team_def_metrics["tackles"] * 100,
        np.nan,
    )

    def_keep_cols = (
        GROUP_COLS
        + [f"{col}_match" for col in def_tmp_cols]
        + [
            "tackle_success_pct",
            "aerial_win_pct",
            "recoveries_def_third_pct",
            "recoveries_mid_third_pct",
            "recoveries_final_third_pct",
            "tackles_def_third_pct",
            "tackles_mid_third_pct",
            "tackles_final_third_pct",
        ]
    )

    team_def_metrics = team_def_metrics[def_keep_cols].copy()

    team_metrics = team_metrics.merge(
        team_def_metrics,
        on=GROUP_COLS,
        how="left",
    )

    # ==================================================
    # 5. MÉTRICAS DE TIRO
    # ==================================================
    shot_events = ["Goal", "MissedShots", "SavedShot", "ShotOnPost", "ChanceMissed"]

    shots = df[df["Event Type"].isin(shot_events)].copy()

    for col in ["Start X", "Start Y"]:
        shots[col] = pd.to_numeric(shots[col], errors="coerce")

    shots = shots.dropna(subset=["Start X", "Start Y"])

    shots["shot"] = True
    shots["goal"] = shots["Event Type"] == "Goal"
    shots["shot_on_target"] = shots["Event Type"].isin(["Goal", "SavedShot"])
    shots["big_chance"] = shots["BigChance"] == "Yes"

    box_x_min = 84.29
    box_y_min = 20.37
    box_y_max = 79.63

    shots["inside_box"] = (
        (shots["Start X"] >= box_x_min)
        & (shots["Start Y"] >= box_y_min)
        & (shots["Start Y"] <= box_y_max)
    )
    shots["outside_box"] = ~shots["inside_box"]

    shots["dx_goal_m"] = (100 - shots["Start X"]) * 105 / 100
    shots["dy_goal_m"] = (50 - shots["Start Y"]) * 68 / 100
    shots["shot_distance_m"] = np.sqrt(
        shots["dx_goal_m"] ** 2 + shots["dy_goal_m"] ** 2
    )

    team_shot_tmp = (
        shots.groupby(GROUP_COLS, as_index=False)
        .agg(
            shots=("shot", "sum"),
            goals=("goal", "sum"),
            shots_on_target=("shot_on_target", "sum"),
            big_chances=("big_chance", "sum"),
            shots_inside_box=("inside_box", "sum"),
            shots_outside_box=("outside_box", "sum"),
            avg_shot_distance_m=("shot_distance_m", "mean"),
        )
    )

    team_shot_metrics = team_matches[GROUP_COLS + ["matches_played"]].merge(
        team_shot_tmp,
        on=GROUP_COLS,
        how="left",
    )

    shot_tmp_cols = [
        "shots",
        "goals",
        "shots_on_target",
        "big_chances",
        "shots_inside_box",
        "shots_outside_box",
    ]

    team_shot_metrics[shot_tmp_cols] = team_shot_metrics[shot_tmp_cols].fillna(0)

    for col in shot_tmp_cols:
        team_shot_metrics[f"{col}_match"] = np.where(
            team_shot_metrics["matches_played"] > 0,
            team_shot_metrics[col] / team_shot_metrics["matches_played"],
            np.nan,
        )

    team_shot_metrics["shot_accuracy"] = np.where(
        team_shot_metrics["shots"] > 0,
        team_shot_metrics["shots_on_target"] / team_shot_metrics["shots"] * 100,
        np.nan,
    )

    team_shot_metrics["goal_conversion"] = np.where(
        team_shot_metrics["shots"] > 0,
        team_shot_metrics["goals"] / team_shot_metrics["shots"] * 100,
        np.nan,
    )

    team_shot_metrics["inside_box_shot_pct"] = np.where(
        team_shot_metrics["shots"] > 0,
        team_shot_metrics["shots_inside_box"] / team_shot_metrics["shots"] * 100,
        np.nan,
    )

    shot_keep_cols = GROUP_COLS + [
        "shots_match",
        "goals_match",
        "shots_on_target_match",
        "big_chances_match",
        "shots_inside_box_match",
        "shots_outside_box_match",
        "shot_accuracy",
        "goal_conversion",
        "inside_box_shot_pct",
        "avg_shot_distance_m",
    ]

    team_shot_metrics = team_shot_metrics[shot_keep_cols].copy()

    team_metrics = team_metrics.merge(
        team_shot_metrics,
        on=GROUP_COLS,
        how="left",
    )

    # ==================================================
    # 6. REGATE Y CREACIÓN
    # ==================================================
    att_events = df[df["Event Type"].isin(["TakeOn", "Dispossessed"])].copy()

    att_events["takeon"] = att_events["Event Type"] == "TakeOn"
    att_events["dispossessed"] = att_events["Event Type"] == "Dispossessed"
    att_events["successful_takeon"] = (
        (att_events["Event Type"] == "TakeOn")
        & (att_events["Outcome"] == "Successful")
    )

    creation_events = df.copy()

    creation_events["big_chance_created"] = (
        creation_events["BigChanceCreated"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    )

    team_dribble_tmp = (
        att_events.groupby(GROUP_COLS, as_index=False)
        .agg(
            takeons=("takeon", "sum"),
            successful_takeons=("successful_takeon", "sum"),
            dispossessed=("dispossessed", "sum"),
        )
    )

    team_creation_tmp = (
        creation_events.groupby(GROUP_COLS, as_index=False)
        .agg(
            big_chances_created=("big_chance_created", "sum"),
        )
    )

    team_att_tmp = team_dribble_tmp.merge(
        team_creation_tmp,
        on=GROUP_COLS,
        how="outer",
    )

    team_att_metrics = team_matches[GROUP_COLS + ["matches_played"]].merge(
        team_att_tmp,
        on=GROUP_COLS,
        how="left",
    )

    att_tmp_cols = [
        "takeons",
        "successful_takeons",
        "dispossessed",
        "big_chances_created",
    ]

    team_att_metrics[att_tmp_cols] = team_att_metrics[att_tmp_cols].fillna(0)

    for col in att_tmp_cols:
        team_att_metrics[f"{col}_match"] = np.where(
            team_att_metrics["matches_played"] > 0,
            team_att_metrics[col] / team_att_metrics["matches_played"],
            np.nan,
        )

    team_att_metrics["takeon_success_pct"] = np.where(
        team_att_metrics["takeons"] > 0,
        team_att_metrics["successful_takeons"] / team_att_metrics["takeons"] * 100,
        np.nan,
    )

    att_keep_cols = GROUP_COLS + [
        "takeons_match",
        "successful_takeons_match",
        "takeon_success_pct",
        "dispossessed_match",
        "big_chances_created_match",
    ]

    team_att_metrics = team_att_metrics[att_keep_cols].copy()

    team_metrics = team_metrics.merge(
        team_att_metrics,
        on=GROUP_COLS,
        how="left",
    )


        # ==================================================
    # 7. PPDA
    # ==================================================
    team_ppda = build_team_ppda(df, press_line_x=40.0)

    team_metrics = team_metrics.merge(
        team_ppda,
        on=GROUP_COLS,
        how="left",
    )

    # ==================================================
    # 8. REDONDEAR Y ORDENAR
    # ==================================================
    team_metrics = team_metrics.sort_values(
        "matches_played",
        ascending=False,
    ).reset_index(drop=True)

    team_metrics = team_metrics.round(2)

    return team_metrics