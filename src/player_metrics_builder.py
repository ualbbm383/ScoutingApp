from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_COLS = ["Player ID", "player_name", "team_name", "league", "season"]


def build_player_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ==================================================
    # 1. LIMPIEZA DE TIPOS
    # ==================================================
    numeric_cols = [
        "Player ID",
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
        "JerseyNumber",
        "FormationSlot",
        "CaptainPlayerId",
        "GoalMouthY",
        "GoalMouthZ",
        "BlockedX",
        "BlockedY",
    ]

    numeric_cols = [col for col in numeric_cols if col in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ==================================================
    # 2. MINUTOS JUGADOS
    # ==================================================
    players_match = (
        df[
            [
                "matchId",
                "match_date",
                "league",
                "season",
                "Team ID",
                "team_name",
                "Player ID",
                "player_name",
            ]
        ]
        .dropna(subset=["Player ID", "player_name"])
        .drop_duplicates()
        .copy()
    )

    sub_on = (
        df[df["Event Type"] == "SubstitutionOn"][["matchId", "Player ID", "Minuto"]]
        .rename(columns={"Minuto": "minute_on"})
        .copy()
    )

    sub_off = (
        df[df["Event Type"] == "SubstitutionOff"][["matchId", "Player ID", "Minuto"]]
        .rename(columns={"Minuto": "minute_sub_off"})
        .copy()
    )

    card_off = (
        df[
            (df["Event Type"] == "Card")
            & ((df["Red"] == "Yes") | (df["SecondYellow"] == "Yes"))
        ][["matchId", "Player ID", "Minuto"]]
        .rename(columns={"Minuto": "minute_card_off"})
        .copy()
    )

    card_off = (
        card_off.groupby(["matchId", "Player ID"], as_index=False)["minute_card_off"]
        .min()
    )

    minutes_match = players_match.merge(
        sub_on,
        on=["matchId", "Player ID"],
        how="left",
    )

    minutes_match = minutes_match.merge(
        sub_off,
        on=["matchId", "Player ID"],
        how="left",
    )

    minutes_match = minutes_match.merge(
        card_off,
        on=["matchId", "Player ID"],
        how="left",
    )

    minutes_match["starter"] = minutes_match["minute_on"].isna()
    minutes_match["minute_on"] = minutes_match["minute_on"].fillna(0)

    minutes_match["minute_off"] = minutes_match[
        ["minute_sub_off", "minute_card_off"]
    ].min(axis=1, skipna=True)

    minutes_match["minute_off"] = minutes_match["minute_off"].fillna(90)

    minutes_match["minutes_played"] = (
        minutes_match["minute_off"] - minutes_match["minute_on"]
    ).clip(lower=0, upper=90)

    player_minutes = (
        minutes_match.groupby(GROUP_COLS, as_index=False)["minutes_played"]
        .sum()
        .rename(columns={"minutes_played": "minutes_total"})
        .sort_values("minutes_total", ascending=False)
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

    player_pass_tmp = (
        passes.groupby(GROUP_COLS, as_index=False)
        .agg(
            passes=("Player ID", "size"),
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

    player_pass_metrics = player_minutes[GROUP_COLS + ["minutes_total"]].merge(
        player_pass_tmp,
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

    player_pass_metrics[pass_tmp_cols] = player_pass_metrics[pass_tmp_cols].fillna(0)

    player_pass_metrics["pass_accuracy"] = np.where(
        player_pass_metrics["passes"] > 0,
        player_pass_metrics["successful_passes"] / player_pass_metrics["passes"] * 100,
        np.nan,
    )
    player_pass_metrics["short_pass_accuracy"] = np.where(
        player_pass_metrics["short_passes"] > 0,
        player_pass_metrics["successful_short_passes"]
        / player_pass_metrics["short_passes"]
        * 100,
        np.nan,
    )
    player_pass_metrics["medium_pass_accuracy"] = np.where(
        player_pass_metrics["medium_passes"] > 0,
        player_pass_metrics["successful_medium_passes"]
        / player_pass_metrics["medium_passes"]
        * 100,
        np.nan,
    )
    player_pass_metrics["long_pass_accuracy"] = np.where(
        player_pass_metrics["long_passes"] > 0,
        player_pass_metrics["successful_long_passes"]
        / player_pass_metrics["long_passes"]
        * 100,
        np.nan,
    )
    player_pass_metrics["forward_pass_accuracy"] = np.where(
        player_pass_metrics["forward_passes"] > 0,
        player_pass_metrics["successful_forward_passes"]
        / player_pass_metrics["forward_passes"]
        * 100,
        np.nan,
    )
    player_pass_metrics["backward_pass_accuracy"] = np.where(
        player_pass_metrics["backward_passes"] > 0,
        player_pass_metrics["successful_backward_passes"]
        / player_pass_metrics["backward_passes"]
        * 100,
        np.nan,
    )
    player_pass_metrics["lateral_pass_accuracy"] = np.where(
        player_pass_metrics["lateral_passes"] > 0,
        player_pass_metrics["successful_lateral_passes"]
        / player_pass_metrics["lateral_passes"]
        * 100,
        np.nan,
    )
    player_pass_metrics["cross_accuracy"] = np.where(
        player_pass_metrics["crosses"] > 0,
        player_pass_metrics["successful_crosses"] / player_pass_metrics["crosses"] * 100,
        np.nan,
    )

    player_pass_metrics["long_pass_pct"] = np.where(
        player_pass_metrics["passes"] > 0,
        player_pass_metrics["long_passes"] / player_pass_metrics["passes"] * 100,
        np.nan,
    )
    player_pass_metrics["progressive_pass_pct"] = np.where(
        player_pass_metrics["passes"] > 0,
        player_pass_metrics["progressive_passes"] / player_pass_metrics["passes"] * 100,
        np.nan,
    )
    player_pass_metrics["passes_final_third_pct"] = np.where(
        player_pass_metrics["passes"] > 0,
        player_pass_metrics["passes_final_third"] / player_pass_metrics["passes"] * 100,
        np.nan,
    )

    pass_per90_cols = [
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

    for col in pass_per90_cols:
        player_pass_metrics[f"{col}_90"] = np.where(
            player_pass_metrics["minutes_total"] > 0,
            player_pass_metrics[col] / player_pass_metrics["minutes_total"] * 90,
            np.nan,
        )

    pass_keep_cols = (
        GROUP_COLS
        + ["minutes_total"]
        + [f"{col}_90" for col in pass_per90_cols]
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

    player_metrics = player_pass_metrics[pass_keep_cols].copy()

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

    player_def_tmp = (
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

    player_def_metrics = player_minutes[GROUP_COLS + ["minutes_total"]].merge(
        player_def_tmp,
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

    player_def_metrics[def_tmp_cols] = player_def_metrics[def_tmp_cols].fillna(0)

    for col in def_tmp_cols:
        player_def_metrics[f"{col}_90"] = np.where(
            player_def_metrics["minutes_total"] > 0,
            player_def_metrics[col] / player_def_metrics["minutes_total"] * 90,
            np.nan,
        )

    player_def_metrics["tackle_success_pct"] = np.where(
        player_def_metrics["tackles"] > 0,
        player_def_metrics["successful_tackles"] / player_def_metrics["tackles"] * 100,
        np.nan,
    )

    player_def_metrics["aerial_win_pct"] = np.where(
        player_def_metrics["aerials"] > 0,
        player_def_metrics["successful_aerials"] / player_def_metrics["aerials"] * 100,
        np.nan,
    )

    def_keep_cols = (
        GROUP_COLS
        + [f"{col}_90" for col in def_tmp_cols]
        + ["tackle_success_pct", "aerial_win_pct"]
    )

    player_def_metrics = player_def_metrics[def_keep_cols].copy()

    player_metrics = player_metrics.merge(
        player_def_metrics,
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

    player_shot_tmp = (
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

    player_shot_metrics = player_minutes[GROUP_COLS + ["minutes_total"]].merge(
        player_shot_tmp,
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

    player_shot_metrics[shot_tmp_cols] = player_shot_metrics[shot_tmp_cols].fillna(0)

    for col in shot_tmp_cols:
        player_shot_metrics[f"{col}_90"] = np.where(
            player_shot_metrics["minutes_total"] > 0,
            player_shot_metrics[col] / player_shot_metrics["minutes_total"] * 90,
            np.nan,
        )

    player_shot_metrics["shot_accuracy"] = np.where(
        player_shot_metrics["shots"] > 0,
        player_shot_metrics["shots_on_target"] / player_shot_metrics["shots"] * 100,
        np.nan,
    )

    player_shot_metrics["goal_conversion"] = np.where(
        player_shot_metrics["shots"] > 0,
        player_shot_metrics["goals"] / player_shot_metrics["shots"] * 100,
        np.nan,
    )

    player_shot_metrics["inside_box_shot_pct"] = np.where(
        player_shot_metrics["shots"] > 0,
        player_shot_metrics["shots_inside_box"] / player_shot_metrics["shots"] * 100,
        np.nan,
    )

    shot_keep_cols = GROUP_COLS + [
        "shots_90",
        "goals_90",
        "shots_on_target_90",
        "big_chances_90",
        "shots_inside_box_90",
        "shots_outside_box_90",
        "shot_accuracy",
        "goal_conversion",
        "inside_box_shot_pct",
        "avg_shot_distance_m",
    ]

    player_shot_metrics = player_shot_metrics[shot_keep_cols].copy()

    player_metrics = player_metrics.merge(
        player_shot_metrics,
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

    player_dribble_tmp = (
        att_events.groupby(GROUP_COLS, as_index=False)
        .agg(
            takeons=("takeon", "sum"),
            successful_takeons=("successful_takeon", "sum"),
            dispossessed=("dispossessed", "sum"),
        )
    )

    player_creation_tmp = (
        creation_events.groupby(GROUP_COLS, as_index=False)
        .agg(
            big_chances_created=("big_chance_created", "sum"),
        )
    )

    player_att_tmp = player_dribble_tmp.merge(
        player_creation_tmp,
        on=GROUP_COLS,
        how="outer",
    )

    player_att_metrics = player_minutes[GROUP_COLS + ["minutes_total"]].merge(
        player_att_tmp,
        on=GROUP_COLS,
        how="left",
    )

    att_tmp_cols = [
        "takeons",
        "successful_takeons",
        "dispossessed",
        "big_chances_created",
    ]

    player_att_metrics[att_tmp_cols] = player_att_metrics[att_tmp_cols].fillna(0)

    for col in att_tmp_cols:
        player_att_metrics[f"{col}_90"] = np.where(
            player_att_metrics["minutes_total"] > 0,
            player_att_metrics[col] / player_att_metrics["minutes_total"] * 90,
            np.nan,
        )

    player_att_metrics["takeon_success_pct"] = np.where(
        player_att_metrics["takeons"] > 0,
        player_att_metrics["successful_takeons"] / player_att_metrics["takeons"] * 100,
        np.nan,
    )

    att_keep_cols = GROUP_COLS + [
        "takeons_90",
        "successful_takeons_90",
        "takeon_success_pct",
        "dispossessed_90",
        "big_chances_created_90",
    ]

    player_att_metrics = player_att_metrics[att_keep_cols].copy()

    player_metrics = player_metrics.merge(
        player_att_metrics,
        on=GROUP_COLS,
        how="left",
    )

    # ==================================================
    # 7. REDONDEAR Y ORDENAR
    # ==================================================
    player_metrics = player_metrics.sort_values(
        "minutes_total",
        ascending=False,
    ).reset_index(drop=True)

    player_metrics = player_metrics.round(2)

    return player_metrics