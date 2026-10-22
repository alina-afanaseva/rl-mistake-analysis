pub use crate::feature_values::{DataType, FeatureDefinition};

pub const ACTIONS: &[&str] = &["unclassified", "shot", "save", "clear", "pass", "boom"];
pub const RELATIONS: &[&str] = &["self", "teammate", "opponent"];

pub fn definitions() -> Vec<FeatureDefinition> {
    use DataType::{Boolean, F32, U32};
    let mut result = Vec::new();
    let mut add = |name: String, data_type, nullable, source: &str| {
        result.push(FeatureDefinition {
            name,
            data_type,
            nullable,
            source: source.into(),
        });
    };
    for stage in ["candidate", "first_touch", "second_touch"] {
        for axis in ["toward", "side", "up"] {
            add(
                format!("{stage}_ball_position_{axis}"),
                F32,
                true,
                "replay_data.frame_data.ball_data",
            );
        }
        add(
            format!("{stage}_incoming_ball_speed"),
            F32,
            true,
            "replay_data.frame_data.ball_data",
        );
        add(
            format!("{stage}_outgoing_ball_speed"),
            F32,
            true,
            "events.touch.frame,replay_data.frame_data.ball_data",
        );
        for depth in ["level_with_ball", "ahead_of_ball"] {
            add(
                format!("subject_ball_depth_at_{stage}_{depth}"),
                Boolean,
                true,
                "events.ball_depth",
            );
        }
        for action in ACTIONS {
            add(
                format!("{stage}_action_{action}"),
                Boolean,
                stage != "candidate",
                "events.touch.tags.action",
            );
        }
    }
    for vector in [
        "position_relative_ball",
        "heading",
        "velocity_relative_ball",
    ] {
        for axis in ["toward", "side", "up"] {
            add(
                format!("closest_opponent_{vector}_{axis}"),
                F32,
                true,
                "events.replay_meta,events.player_activity,replay_data.frame_data",
            );
        }
    }
    add(
        "closest_opponent_boost_before_pct".into(),
        F32,
        true,
        "events.accumulation_tracks.boost_amount",
    );
    for team in ["teammate", "opponent"] {
        add(
            format!("approaching_{team}_count"),
            U32,
            true,
            "events.replay_meta,events.player_activity,replay_data.frame_data",
        );
    }
    for (name, source) in [
        ("candidate_dodging", "events.touch.tags.dodge_state"),
        ("opponent_beaten_at_candidate", "events.beaten_to_ball"),
        ("ground_carry_before", "events.ball_carry"),
        ("air_dribble_before", "events.air_dribble"),
        ("receiving_pass_before", "events.pass"),
    ] {
        add(name.into(), Boolean, false, source);
    }
    add(
        "subject_boost_before_pct".into(),
        F32,
        true,
        "events.accumulation_tracks.boost_amount",
    );
    add(
        "subject_field_half_at_candidate_offensive".into(),
        Boolean,
        true,
        "events.field_half",
    );
    for role in ["most_back", "most_forward", "mid", "no_teammates"] {
        add(
            format!("subject_depth_role_at_candidate_{role}"),
            Boolean,
            true,
            "events.depth_role",
        );
    }
    for height in ["low_air", "high_air"] {
        add(
            format!("subject_height_band_at_candidate_{height}"),
            Boolean,
            true,
            "events.movement",
        );
    }
    for stage in ["previous_touch", "first_touch", "second_touch"] {
        for relation in RELATIONS {
            add(format!("{stage}_{relation}"), Boolean, true, "events.touch");
        }
        add(format!("{stage}_delay_s"), F32, true, "events.touch");
    }
    for (name, source) in [
        ("opponent_whiff_during_exchange", "events.whiff"),
        ("subject_beaten_during_exchange", "events.beaten_to_ball"),
        ("subject_demo_during_exchange", "events.demolition"),
        (
            "first_touch_opponent_control",
            "events.touch.tags.possession",
        ),
    ] {
        add(name.into(), Boolean, true, source);
    }
    for state in ["own_team", "opponent", "neutral"] {
        add(
            format!("possession_before_{state}"),
            Boolean,
            true,
            "events.possession",
        );
    }
    for state in ["own_team", "opponent", "neutral", "no_change"] {
        add(
            format!("next_possession_{state}"),
            Boolean,
            true,
            "events.possession",
        );
    }
    add(
        "possession_change_delay_s".into(),
        F32,
        true,
        "events.possession",
    );
    for team in ["own_team", "opponent"] {
        add(
            format!("next_goal_{team}"),
            Boolean,
            false,
            "events.goal_context",
        );
    }
    add("next_goal_delay_s".into(), F32, true, "events.goal_context");
    add(
        "touches_before_next_goal".into(),
        U32,
        true,
        "events.touch,events.goal_context",
    );
    add(
        "subject_next_touch_delay_s".into(),
        F32,
        true,
        "events.touch",
    );
    for stage in ["first_touch", "second_touch"] {
        add(
            format!("subject_ball_distance_at_{stage}"),
            F32,
            true,
            "replay_data.frame_data",
        );
        for axis in ["toward", "side", "up"] {
            add(
                format!("subject_displacement_to_{stage}_{axis}"),
                F32,
                true,
                "replay_data.frame_data,events.demolition",
            );
            add(
                format!("subject_heading_at_{stage}_{axis}"),
                F32,
                true,
                "replay_data.frame_data.players.rigid_body.rotation",
            );
        }
    }
    result
}
