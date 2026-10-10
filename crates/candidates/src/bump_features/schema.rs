use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Boolean,
    F32,
    U32,
}

#[derive(Debug, Serialize)]
pub struct FeatureDefinition {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub source: &'static str,
}

pub fn definitions() -> Vec<FeatureDefinition> {
    use DataType::{Boolean, F32, U32};
    let mut definitions = Vec::new();
    let mut add = |name: String, data_type, nullable, source| {
        definitions.push(FeatureDefinition {
            name,
            data_type,
            nullable,
            source,
        });
    };
    for name in [
        "first_touch_p",
        "first_touch_q",
        "first_touch_other_teammate",
        "first_touch_opponent",
    ] {
        add(name.into(), Boolean, false, "events.touch");
    }
    add("first_touch_delay_s".into(), F32, true, "events.touch");
    for name in [
        "possession_before_own_team",
        "possession_before_opponent",
        "next_possession_own_team",
        "next_possession_opponent",
        "next_possession_neutral",
    ] {
        add(name.into(), Boolean, false, "events.possession");
    }
    add(
        "possession_change_delay_s".into(),
        F32,
        true,
        "events.possession",
    );
    for name in ["next_goal_own_team", "next_goal_opponent"] {
        add(name.into(), Boolean, false, "events.goal_context");
    }
    add("next_goal_delay_s".into(), F32, true, "events.goal_context");
    add(
        "touches_before_next_goal".into(),
        U32,
        true,
        "events.touch,events.goal_context",
    );
    add("p_is_initiator".into(), Boolean, false, "events.bump");
    add(
        "contact_gap".into(),
        F32,
        false,
        "events.bump.contact_distance",
    );
    for participant in ["p", "q"] {
        for vector in ["velocity_before", "velocity_delta", "position_delta"] {
            for axis in ["toward", "side", "up"] {
                add(
                    format!("{participant}_{vector}_{axis}"),
                    F32,
                    true,
                    "replay_data.frame_data",
                );
            }
        }
        for name in [
            "ball_distance_before",
            "ball_distance_delta",
            "heading_alignment_before",
            "upright_after",
        ] {
            add(
                format!("{participant}_{name}"),
                F32,
                true,
                "replay_data.frame_data",
            );
        }
        add(
            format!("{participant}_next_touch_delay_s"),
            F32,
            true,
            "events.touch",
        );
        for action in ["unclassified", "shot", "save", "pass", "clear", "boom"] {
            add(
                format!("{participant}_next_touch_action_{action}"),
                Boolean,
                false,
                "events.touch.tags.action",
            );
        }
        add(
            format!("{participant}_whiff_before_next_touch"),
            Boolean,
            false,
            "events.whiff,events.touch,events.goal_context",
        );
        add(
            format!("{participant}_beaten_before_next_touch"),
            Boolean,
            false,
            "events.beaten_to_ball,events.touch,events.goal_context",
        );
        for direction in ["advance", "retreat"] {
            add(
                format!("{participant}_next_touch_ball_{direction}"),
                F32,
                true,
                "events.touch.ball_movement",
            );
        }
        for (name, source) in [
            ("ground_carry_before", "events.ball_carry"),
            ("air_dribble_before", "events.air_dribble"),
            ("receiving_pass_before", "events.pass"),
            ("shadowing_before", "events.shadow_defense"),
        ] {
            add(format!("{participant}_{name}"), Boolean, false, source);
        }
        add(
            format!("{participant}_dodging_before"),
            Boolean,
            true,
            "replay_data.frame_data",
        );
        add(
            format!("{participant}_field_half_before_offensive"),
            Boolean,
            true,
            "events.field_half",
        );
        for depth in ["level_with_ball", "ahead_of_ball"] {
            add(
                format!("{participant}_ball_depth_before_{depth}"),
                Boolean,
                true,
                "events.ball_depth",
            );
        }
        for role in ["most_back", "most_forward", "mid", "no_teammates"] {
            add(
                format!("{participant}_depth_role_before_{role}"),
                Boolean,
                true,
                "events.depth_role",
            );
        }
        for height in ["low_air", "high_air"] {
            add(
                format!("{participant}_height_band_before_{height}"),
                Boolean,
                true,
                "events.movement",
            );
        }
        add(
            format!("{participant}_boost_before_pct"),
            F32,
            true,
            "events.accumulation_tracks.boost_amount",
        );
        for name in ["next_boost_pickup_delay_s", "next_boost_pickup_gain_pct"] {
            add(
                format!("{participant}_{name}"),
                F32,
                true,
                "events.boost_pickups",
            );
        }
        add(
            format!("{participant}_next_pad_distance_delta"),
            F32,
            true,
            "events.boost_pickups,replay_data.boost_pad_events,replay_data.boost_pads",
        );
        add(
            format!("{participant}_previous_bump_age_s"),
            F32,
            true,
            "events.bump",
        );
        add(
            format!("{participant}_previous_demolition_age_s"),
            F32,
            true,
            "events.demolition",
        );
    }
    definitions
}
