use rl_mistake_analysis_candidates::bump_features::{
    FeatureBatch, FeatureSampling, Measurement, ReplayAnalysis, SamplingMethod, SamplingStatus,
    extract,
};
use serde_json::{Value, json};

fn event(id: &str, stream: &str, payload: Value) -> Value {
    json!({"meta": {"id": id, "stream": stream}, "payload": {"kind": stream, "payload": payload}})
}

fn body(x: f64, z: f64, velocity: f64, inverted: bool) -> Value {
    json!({"location": {"x": x, "y": 0, "z": z},
        "rotation": {"x": if inverted {1} else {0}, "y": 0, "z": 0, "w": if inverted {0} else {1}},
        "linear_velocity": {"x": velocity, "y": 0, "z": 0}})
}

fn fixture() -> Value {
    let times: Vec<_> = (0..12)
        .map(|frame| json!({"time": frame as f64 / 8.0}))
        .collect();
    let p_velocities = [0., 0., 0., 10., 20., 50., 80., 85., 87., 90., 90., 90.];
    let q_velocities = [
        100., 100., 100., 90., 60., 50., 50., 45., 45., 45., 45., 45.,
    ];
    let players: Vec<_> =
        [p_velocities, q_velocities]
            .iter()
            .enumerate()
            .map(|(index, velocities)| {
                let frames: Vec<_> = velocities.iter().enumerate().map(|(frame, velocity)| {
            json!({"Data": {"rigid_body": body(index as f64 * 100.0 + frame as f64 * 10.0,
                frame as f64, *velocity, index == 1), "dodge_active": frame == 3}})
        }).collect();
                json!([{"Steam": index + 1}, {"frames": frames}])
            })
            .collect();
    let ball: Vec<_> = (0..12)
        .map(|_| json!({"Data": {"rigid_body": body(1000., 100., 0., false)}}))
        .collect();
    let p = json!({"Steam": 1});
    let q = json!({"Steam": 2});
    let mut events = vec![
        event(
            "bump:4",
            "bump",
            json!({"frame": 4, "time": 0.5, "initiator": p, "victim": q,
            "initiator_is_team_0": true, "victim_is_team_0": true, "is_team_bump": true, "contact_distance": 5.0}),
        ),
        event(
            "possession:0",
            "possession",
            json!({"time": 0.0, "end_time": 1.375, "possession_state": "neutral"}),
        ),
        event(
            "touch:p",
            "touch",
            json!({"frame": 8, "time": 1.0, "player": p, "is_team_0": true, "touch_id": 1,
            "tags": [{"group": "action", "value": "shot"}], "ball_movement": {"advance_distance": 100., "retreat_distance": 20.}}),
        ),
        event(
            "touch:q",
            "touch",
            json!({"frame": 9, "time": 1.125, "player": q, "is_team_0": true, "touch_id": 2, "tags": []}),
        ),
        event(
            "goal:10",
            "goal_context",
            json!({"frame": 10, "time": 1.25, "scoring_team_is_team_0": false}),
        ),
        event(
            "whiff:p",
            "whiff",
            json!({"player": p, "kind": "whiff", "time": 0.8125, "resolved_time": 1.0625}),
        ),
        event(
            "whiff:q",
            "whiff",
            json!({"player": q, "kind": "whiff", "time": 0.75, "resolved_time": 0.875}),
        ),
        event(
            "beaten:p",
            "beaten_to_ball",
            json!({"player": p, "time": 0.625}),
        ),
        event(
            "beaten:q",
            "beaten_to_ball",
            json!({"player": q, "time": 1.1875}),
        ),
    ];
    for (id, player) in [("p", &p), ("q", &q)] {
        events.extend([
            event(
                &format!("half:{id}"),
                "field_half",
                json!({"player": player, "time": 0., "end_time": 1.375, "state": "defensive"}),
            ),
            event(
                &format!("depth:{id}"),
                "ball_depth",
                json!({"player": player, "time": 0., "end_time": 1.375, "state": "behind_ball"}),
            ),
            event(
                &format!("role:{id}"),
                "depth_role",
                json!({"player": player, "time": 0., "end_time": 1.375, "state": "most_back"}),
            ),
            event(
                &format!("movement:{id}"),
                "movement",
                json!({"player": player, "frame": 0, "end_frame": 11, "height_band": "ground"}),
            ),
        ]);
    }
    json!({"name": "subtr-actor", "schema_version": 1, "status": "ok", "revision": "fixture",
    "replay_sha256": "0".repeat(64), "outputs": {
        "events": {"events": {"events": events}, "accumulation_tracks": [
            {"player_id": p, "quantity": "boost_amount", "points": [{"frame": 0, "value": 51.}, {"frame": 4, "value": 0.}]},
            {"player_id": q, "quantity": "boost_amount", "points": [{"frame": 0, "value": 127.5}]}
        ]},
        "replay_data": {"frame_data": {"metadata_frames": times, "players": players, "ball_data": {"frames": ball}},
            "boost_pad_events": [], "boost_pads": []}
    }})
}

fn calculate(document: Value, player: Option<&Value>) -> Result<FeatureBatch, String> {
    let analysis: ReplayAnalysis = serde_json::from_value(document).unwrap();
    extract(
        &analysis,
        FeatureSampling {
            method: SamplingMethod::FirstAccelerationPeakThenMinimum,
        },
        player,
    )
}

fn float(batch: &FeatureBatch, player: u64, name: &str) -> f32 {
    let row = batch
        .rows
        .iter()
        .find(|row| row.player_id == json!({"Steam": player}))
        .unwrap();
    match row.features[name] {
        Some(Measurement::Float(value)) => value,
        ref value => panic!("unexpected {name}: {value:?}"),
    }
}

#[test]
fn independently_samples_and_describes_both_participants() {
    let batch = calculate(fixture(), None).unwrap();
    assert_eq!(batch.rows.len(), 2);
    assert_eq!(batch.feature_definitions.len(), 105);
    let p = &batch.rows[0];
    let q = &batch.rows[1];
    assert_eq!(p.player_id, json!({"Steam": 1}));
    assert_eq!(p.samples.p.before_frame, Some(3));
    assert_eq!(p.samples.p.peak_frame, Some(6));
    assert_eq!(p.samples.p.after_frame, Some(8));
    assert_eq!(p.samples.q.peak_frame, Some(4));
    assert_eq!(p.samples.q.after_frame, Some(6));
    assert_eq!(q.samples.p.after_frame, p.samples.q.after_frame);
    assert_eq!(q.samples.q.after_frame, p.samples.p.after_frame);
    assert_eq!(p.samples.p.status, SamplingStatus::Complete);
    assert_eq!(float(&batch, 1, "p_velocity_before_toward"), 10.);
    assert_eq!(float(&batch, 1, "p_velocity_delta_toward"), 77.);
    assert_eq!(float(&batch, 1, "q_velocity_delta_toward"), 40.);
    assert_eq!(float(&batch, 1, "p_position_delta_toward"), 50.);
    assert_eq!(float(&batch, 1, "q_position_delta_toward"), -30.);
    assert_eq!(float(&batch, 1, "p_position_delta_up"), 5.);
    assert_eq!(float(&batch, 1, "q_position_delta_up"), 3.);
    assert_eq!(float(&batch, 1, "p_upright_after"), 1.);
    assert_eq!(float(&batch, 1, "q_upright_after"), -1.);
    assert_eq!(float(&batch, 1, "p_boost_before_pct"), 20.);
    assert_eq!(float(&batch, 1, "q_boost_before_pct"), 50.);
    let filtered = calculate(fixture(), Some(&json!({"Steam": 2}))).unwrap();
    assert_eq!(filtered.rows.len(), 1);
    assert_eq!(filtered.rows[0].features, q.features);
}

#[test]
fn role_reversal_does_not_duplicate_or_change_subject_measurements() {
    let original = calculate(fixture(), None).unwrap();
    let mut document = fixture();
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    let mut reversed = events[0].clone();
    reversed["meta"]["id"] = json!("bump:5");
    let bump = &mut reversed["payload"]["payload"];
    bump["frame"] = json!(5);
    bump["time"] = json!(0.625);
    bump["initiator"] = json!({"Steam": 2});
    bump["victim"] = json!({"Steam": 1});
    events.insert(0, reversed);
    let batch = calculate(document.clone(), None).unwrap();
    assert_eq!(
        serde_json::to_value(&batch).unwrap(),
        serde_json::to_value(original).unwrap()
    );
    let victim = calculate(document.clone(), Some(&json!({"Steam": 2}))).unwrap();
    assert_eq!(victim.rows.len(), 1);
    assert_eq!(victim.rows[0].source_event_id, "bump:4");
    document["outputs"]["events"]["events"]["events"][0]["payload"]["payload"]["time"] = json!(0.6);
    assert!(calculate(document, None).is_err());
}

#[test]
fn later_contact_history_uses_the_retained_contact_time() {
    let mut document = fixture();
    let frame_data = &mut document["outputs"]["replay_data"]["frame_data"];
    *frame_data["metadata_frames"].as_array_mut().unwrap() = (0..24)
        .map(|frame| json!({"time": frame as f64 / 8.0}))
        .collect();
    for player in frame_data["players"].as_array_mut().unwrap() {
        let frames = player[1]["frames"].as_array_mut().unwrap();
        frames.resize(24, frames.last().unwrap().clone());
    }
    let ball = frame_data["ball_data"]["frames"].as_array_mut().unwrap();
    ball.resize(24, ball.last().unwrap().clone());
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.retain(|event| event["meta"]["stream"] != "goal_context");
    events[1]["payload"]["payload"]["end_time"] = json!(2.875);
    let mut reversed = events[0].clone();
    reversed["meta"]["id"] = json!("bump:5");
    reversed["payload"]["payload"]["frame"] = json!(5);
    reversed["payload"]["payload"]["time"] = json!(0.625);
    reversed["payload"]["payload"]["initiator"] = json!({"Steam": 2});
    reversed["payload"]["payload"]["victim"] = json!({"Steam": 1});
    let mut later = events[0].clone();
    later["meta"]["id"] = json!("bump:16");
    later["payload"]["payload"]["frame"] = json!(16);
    later["payload"]["payload"]["time"] = json!(2.0);
    events.extend([reversed, later]);
    let batch = calculate(document, None).unwrap();
    assert_eq!(batch.rows.len(), 4);
    for row in &batch.rows[2..] {
        assert_eq!(row.source_event_id, "bump:16");
        for participant in ["p", "q"] {
            assert_eq!(
                row.features[&format!("{participant}_previous_bump_age_s")],
                Some(Measurement::Float(1.5))
            );
        }
    }
}

#[test]
fn outcomes_use_each_players_next_touch_and_handle_simultaneous_touches() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(event(
            "touch:opponent",
            "touch",
            json!({"frame": 8, "time": 1.0, "player": {"Steam": 3},
            "is_team_0": false, "touch_id": 1, "tags": []}),
        ));
    let batch = calculate(document, None).unwrap();
    let features = &batch.rows[0].features;
    for (name, value) in [
        ("first_touch_p", true),
        ("first_touch_q", false),
        ("first_touch_opponent", true),
        ("p_next_touch_action_shot", true),
        ("q_next_touch_action_unclassified", true),
        ("p_whiff_before_next_touch", false),
        ("q_whiff_before_next_touch", true),
        ("p_beaten_before_next_touch", true),
        ("q_beaten_before_next_touch", false),
        ("next_goal_opponent", true),
        ("next_goal_own_team", false),
    ] {
        assert_eq!(features[name], Some(Measurement::Boolean(value)), "{name}");
    }
    assert_eq!(
        features["touches_before_next_goal"],
        Some(Measurement::Count(2))
    );
    assert_eq!(features["q_next_touch_ball_advance"], None);
    assert_eq!(float(&batch, 1, "p_next_touch_ball_advance"), 100.);
}

#[test]
fn missing_velocities_stop_sampling_and_are_never_replaced() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][5]["Data"]["rigid_body"]
        ["linear_velocity"] = Value::Null;
    let batch = calculate(document.clone(), None).unwrap();
    assert_eq!(
        batch.rows[0].samples.p.status,
        SamplingStatus::MissingVelocity
    );
    assert_eq!(batch.rows[0].samples.p.after_frame, None);
    assert_eq!(batch.rows[0].features["p_velocity_delta_toward"], None);
    assert_eq!(batch.rows[0].features["p_position_delta_toward"], None);
    assert_eq!(float(&batch, 1, "p_velocity_before_toward"), 10.);
    document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"][5]["Data"]["rigid_body"].as_object_mut().unwrap().remove("linear_velocity");
    assert!(serde_json::from_value::<ReplayAnalysis>(document).is_err());
}

#[test]
fn sampling_does_not_cross_a_goal_or_use_the_match_countdown() {
    let mut document = fixture();
    for metadata in document["outputs"]["replay_data"]["frame_data"]["metadata_frames"]
        .as_array_mut()
        .unwrap()
    {
        metadata["seconds_remaining"] = json!(300);
    }
    for event in document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
    {
        if event["meta"]["stream"] == "goal_context" {
            event["payload"]["payload"]["time"] = json!(0.625);
        }
    }
    let batch = calculate(document, None).unwrap();
    assert_eq!(batch.rows[0].samples.p.status, SamplingStatus::PlayBoundary);
    assert_eq!(batch.rows[0].samples.p.after_frame, None);
    assert_eq!(float(&batch, 1, "next_goal_delay_s"), 0.125);
    assert_eq!(batch.rows[0].features["p_next_touch_delay_s"], None);
}

#[test]
fn kickoff_collisions_use_the_observed_live_action_start() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(event(
            "kickoff:0",
            "kickoff",
            json!({"start_time": 0.0, "live_action_start_time": 0.25, "first_touch_time": 1.0}),
        ));
    let batch = calculate(document.clone(), None).unwrap();
    assert_eq!(batch.rows.len(), 2);
    assert_eq!(batch.rows[0].samples.play_start, 0.25);
    assert_eq!(batch.rows[0].samples.p.before_frame, Some(3));
    assert_eq!(batch.rows[0].samples.p.after_frame, Some(8));
    assert_eq!(batch.rows[0].samples.p.status, SamplingStatus::Complete);
    let kickoff = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap();
    kickoff["payload"]["payload"]["first_touch_time"] = Value::Null;
    assert!(calculate(document, None).is_ok());
}

#[test]
fn kickoff_sampling_excludes_countdown_frames_and_requires_a_start_time() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(event(
            "kickoff:0",
            "kickoff",
            json!({"start_time": 0.0, "live_action_start_time": 0.375, "first_touch_time": 1.0}),
        ));
    let batch = calculate(document.clone(), None).unwrap();
    assert_eq!(batch.rows[0].samples.p.status, SamplingStatus::PlayBoundary);
    assert_eq!(batch.rows[0].samples.p.after_frame, None);
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.last_mut().unwrap()["payload"]["payload"]["live_action_start_time"] = json!(0.625);
    assert!(
        calculate(document.clone(), None)
            .unwrap_err()
            .contains("outside its observed live-play segment")
    );
    let events = document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap();
    events.last_mut().unwrap()["payload"]["payload"]["live_action_start_time"] = Value::Null;
    assert!(
        calculate(document.clone(), None)
            .unwrap_err()
            .contains("invalid live_action_start_time")
    );
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap()["payload"]["payload"]
        .as_object_mut()
        .unwrap()
        .remove("live_action_start_time");
    assert!(
        calculate(document, None)
            .unwrap_err()
            .contains("missing live_action_start_time")
    );
}

#[test]
fn undefined_horizontal_basis_preserves_vertical_measurements() {
    let mut document = fixture();
    document["outputs"]["replay_data"]["frame_data"]["players"][1][1]["frames"][3]["Data"]["rigid_body"]
        ["location"]["x"] = json!(30.);
    let batch = calculate(document, None).unwrap();
    assert_eq!(batch.rows[0].features["p_position_delta_toward"], None);
    assert_eq!(float(&batch, 1, "p_position_delta_up"), 5.);
    assert_eq!(float(&batch, 1, "p_velocity_before_up"), 0.);
}

#[test]
fn pad_distance_requires_an_exact_native_pickup_identity() {
    let mut document = fixture();
    let mut pickup = event(
        "pickup:p",
        "boost_pickups",
        json!({"player_id": {"Steam": 1}, "frame": 9, "time": 1.125, "collected_amount": 30.6}),
    );
    pickup["payload"]["kind"] = json!("boost_pickup");
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .push(pickup);
    document["outputs"]["replay_data"]["boost_pad_events"] = json!([{ "time": 1.125, "frame": 9, "pad_id": "pad-1", "player": {"Steam": 1}, "kind": {"PickedUp": {"sequence": 1}} }]);
    document["outputs"]["replay_data"]["boost_pads"] =
        json!([{ "pad_id": "pad-1", "position": {"x": 1000., "y": 0., "z": 0.} }]);
    let batch = calculate(document.clone(), None).unwrap();
    assert_eq!(float(&batch, 1, "p_next_pad_distance_delta"), -50.);
    assert_eq!(float(&batch, 1, "p_next_boost_pickup_gain_pct"), 12.);
    document["outputs"]["replay_data"]["boost_pad_events"][0]["frame"] = json!(10);
    let unmatched = calculate(document, None).unwrap();
    assert_eq!(
        unmatched.rows[0].features["p_next_pad_distance_delta"],
        None
    );
}

#[test]
fn sampling_configuration_requires_the_named_method() {
    assert!(serde_json::from_value::<FeatureSampling>(json!({})).is_err());
    assert!(serde_json::from_value::<FeatureSampling>(json!({"method": "unknown"})).is_err());
    assert!(
        serde_json::from_value::<FeatureSampling>(
            json!({"before_seconds": 0.2, "after_seconds": 0.2})
        )
        .is_err()
    );
}

#[test]
fn irregular_frame_spacing_changes_the_acceleration_peak() {
    let mut document = fixture();
    let frames = document["outputs"]["replay_data"]["frame_data"]["metadata_frames"]
        .as_array_mut()
        .unwrap();
    for frame in frames.iter_mut().skip(6) {
        frame["time"] = json!(frame["time"].as_f64().unwrap() + 0.125);
    }
    let times: Vec<_> = frames.iter().map(|frame| frame["time"].clone()).collect();
    for event in document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
    {
        if let Some(frame) = event["payload"]["payload"]["frame"].as_u64()
            && event["payload"]["payload"].get("time").is_some()
        {
            event["payload"]["payload"]["time"] = times[frame as usize].clone();
        }
    }
    let batch = calculate(document, None).unwrap();
    assert_eq!(batch.rows[0].samples.p.peak_frame, Some(5));
    assert_eq!(batch.rows[0].samples.p.after_frame, Some(8));
    assert_eq!(batch.rows[0].samples.p.after_time, Some(1.125));
    assert_eq!(batch.rows[0].samples.q.after_time, Some(0.875));
}

#[test]
fn a_replay_without_observed_extrema_leaves_the_after_sample_unavailable() {
    let mut document = fixture();
    document["outputs"]["events"]["events"]["events"]
        .as_array_mut()
        .unwrap()
        .retain(|event| event["meta"]["stream"] != "goal_context");
    for (frame, sample) in
        document["outputs"]["replay_data"]["frame_data"]["players"][0][1]["frames"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
    {
        sample["Data"]["rigid_body"]["linear_velocity"]["x"] = json!(frame);
    }
    let batch = calculate(document, None).unwrap();
    assert_eq!(batch.rows[0].samples.p.status, SamplingStatus::ReplayEnd);
    assert_eq!(batch.rows[0].samples.p.peak_frame, None);
    assert_eq!(batch.rows[0].samples.p.after_frame, None);
    assert_eq!(batch.rows[0].features["p_velocity_delta_toward"], None);
}
