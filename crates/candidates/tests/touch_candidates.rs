use rl_mistake_analysis_candidates::{MistakeKind, ReplayFacts, annotations::Annotation};
use serde_json::{Value, json};

fn event(stream: &str, id: &str, payload: Value) -> Value {
    json!({"meta":{"stream":stream,"id":id},"payload":{"kind":stream,"payload":payload}})
}
fn touch(id: u64, frame: u64, player: &str) -> Value {
    event(
        "touch",
        &format!("touch:{id}"),
        json!({"touch_id":id,"frame":frame,
        "time":frame as f64/30.0,"player":{"Epic":player},"is_team_0":true,"tags":[]}),
    )
}
fn episode(start: u64, end: u64) -> Value {
    event(
        "player_possession",
        &format!("episode:{start}"),
        json!({
        "player_id":{"Epic":"a"},"start_frame":start,"end_frame":end}),
    )
}
fn ownership(id: &str, start: u64, end: u64, state: &str, active: bool) -> Value {
    event(
        "possession",
        id,
        json!({"frame":start,"end_frame":end,"possession_state":state,"active":active}),
    )
}
fn fifty(start: u64, end: u64, first: u64, second: u64) -> Value {
    event(
        "fifty_fifty",
        "fifty",
        json!({"start_frame":start,"resolve_frame":end,
        "team_zero_touch_frame":first,"team_one_touch_frame":second}),
    )
}
fn facts(events: Vec<Value>) -> ReplayFacts {
    serde_json::from_value(
        json!({"name":"subtr-actor","schema_version":1,"status":"ok",
        "revision":"supplier","replay_sha256":"a".repeat(64),"events":events}),
    )
    .unwrap()
}
fn kept(events: Vec<Value>) -> Vec<u64> {
    MistakeKind::BangWithTime
        .candidates(&facts(events))
        .unwrap()
        .candidates
        .iter()
        .map(|c| c.frame)
        .collect()
}

#[test]
fn keeps_last_touch_within_each_recorded_player_possession() {
    let selection = MistakeKind::BangWithTime
        .candidates(&facts(vec![
            touch(2, 30, "a"),
            touch(0, 10, "a"),
            touch(1, 20, "a"),
            episode(9, 20),
            ownership("owned", 0, 40, "team_zero", true),
        ]))
        .unwrap();
    assert_eq!(
        selection
            .candidates
            .iter()
            .map(|c| c.frame)
            .collect::<Vec<_>>(),
        vec![20, 30]
    );
}

#[test]
fn neutral_inactive_and_other_team_intervals_keep_the_launch_before_recovery() {
    for (state, active) in [("neutral", true), ("team_zero", false), ("team_one", true)] {
        assert_eq!(
            kept(vec![
                touch(0, 10, "a"),
                touch(1, 20, "a"),
                episode(0, 30),
                ownership("before", 0, 12, "team_zero", true),
                ownership("break", 12, 18, state, active),
                ownership("after", 18, 30, "team_zero", true)
            ]),
            vec![10, 20]
        );
    }
}

#[test]
fn keeps_touches_separate_across_gaps_in_recorded_possession() {
    let selection = MistakeKind::BangWithTime
        .candidates(&facts(vec![
            touch(0, 10, "a"),
            touch(1, 20, "a"),
            episode(0, 30),
            ownership("before", 0, 12, "team_zero", true),
            ownership("after", 18, 30, "team_zero", true),
        ]))
        .unwrap();
    assert_eq!(selection.candidates.len(), 2);

    assert_eq!(
        kept(vec![
            touch(0, 10, "a"),
            touch(1, 20, "a"),
            episode(0, 15),
            episode(18, 30),
            ownership("owned", 0, 30, "team_zero", true)
        ]),
        vec![10, 20]
    );
}

#[test]
fn another_player_and_ambiguous_contact_frames_break_the_original_sequence() {
    assert_eq!(
        kept(vec![
            touch(0, 10, "a"),
            touch(1, 15, "b"),
            touch(2, 20, "a"),
            touch(3, 30, "a"),
            touch(4, 30, "b"),
            episode(0, 40),
            ownership("owned", 0, 40, "team_zero", true)
        ]),
        vec![10, 15, 20, 30, 30]
    );
}

#[test]
fn repeated_contacts_on_one_frame_only_break_groups_when_players_differ() {
    let events = vec![
        touch(0, 10, "a"),
        touch(1, 20, "a"),
        touch(2, 20, "a"),
        touch(3, 30, "a"),
        episode(0, 40),
        ownership("owned", 0, 40, "team_zero", true),
    ];
    assert_eq!(kept(events.clone()), vec![30]);

    let mut ambiguous = events;
    ambiguous.push(touch(4, 20, "b"));
    assert_eq!(kept(ambiguous), vec![10, 20, 20, 20, 30]);
}

#[test]
fn fifties_remain_boundaries_before_final_filtering() {
    assert_eq!(
        kept(vec![
            touch(0, 10, "a"),
            touch(1, 20, "a"),
            touch(2, 30, "a"),
            episode(0, 40),
            ownership("owned", 0, 40, "team_zero", true),
            fifty(20, 22, 20, 21)
        ]),
        vec![10, 30]
    );
    let selection = MistakeKind::BangWithTime
        .candidates(&facts(vec![
            touch(0, 10, "a"),
            touch(1, 30, "a"),
            episode(0, 40),
            ownership("owned", 0, 40, "team_zero", true),
            fifty(15, 25, 15, 16),
        ]))
        .unwrap();
    assert_eq!(selection.candidates.len(), 2);
}

#[test]
fn contested_tags_and_exact_initial_fifty_frames_are_filtered_without_expansion() {
    let mut tagged = touch(3, 40, "a");
    tagged["payload"]["payload"]["tags"] = json!([{"group":"contested","value":"contested"}]);
    assert_eq!(
        kept(vec![
            touch(0, 10, "a"),
            touch(1, 10, "c"),
            touch(2, 11, "b"),
            tagged,
            fifty(10, 15, 10, 12)
        ]),
        vec![11]
    );
}

#[test]
fn keeps_touches_separate_when_player_possession_is_unrecorded() {
    assert_eq!(
        kept(vec![
            touch(0, 10, "a"),
            touch(1, 20, "a"),
            ownership("owned", 0, 30, "team_zero", true)
        ]),
        vec![10, 20]
    );
}

#[test]
fn malformed_native_events_fail_explicitly() {
    assert!(
        MistakeKind::BangWithTime
            .candidates(&facts(vec![touch(0, 10, "a"), touch(0, 20, "a")]))
            .is_err()
    );
    assert!(
        MistakeKind::BangWithTime
            .candidates(&facts(vec![episode(20, 10)]))
            .is_err()
    );
    assert!(
        MistakeKind::BangWithTime
            .candidates(&facts(vec![
                ownership("a", 10, 20, "team_zero", true),
                ownership("b", 15, 25, "team_zero", true)
            ]))
            .is_err()
    );
    assert!(
        MistakeKind::BangWithTime
            .candidates(&facts(vec![ownership("a", 10, 20, "unknown", true)]))
            .is_err()
    );
    assert!(
        MistakeKind::BangWithTime
            .candidates(&facts(vec![fifty(20, 10, 10, 11)]))
            .is_err()
    );
}

#[test]
fn terminal_annotation_and_batch_preserve_exact_native_anchor() {
    let mut terminal = touch(7, 42, "a");
    let time = 13.065633773803711;
    terminal["payload"]["payload"]["time"] = json!(time);
    let replay = facts(vec![
        touch(6, 41, "a"),
        terminal.clone(),
        episode(40, 43),
        ownership("owned", 40, 43, "team_zero", true),
    ]);
    let batch = MistakeKind::BangWithTime.candidates(&replay).unwrap();
    batch.validate().unwrap();
    assert!(batch.feature_names.is_empty());
    let c = &batch.candidates[0];
    assert_eq!(c.raw_time, time);
    assert_eq!(c.source_event_ids, vec!["touch:7"]);
    assert_eq!(c.evidence, terminal["payload"]["payload"]);
    let mut annotation:Annotation=serde_json::from_value(json!({"kind":"bang_with_time",
        "replay_sha256":batch.replay_sha256,"frame":42,"touch_id":7,"raw_time":c.raw_time,
        "player_id":c.player_id,"evidence":c.evidence,"label":1,"engine_revision":"reviewed-supplier",
        "source_event_ids":c.source_event_ids})).unwrap();
    annotation.validate().unwrap();
    annotation.incident =
        serde_json::from_value(json!({"kind":"bang_with_time","frame":43,"touch_id":7})).unwrap();
    assert!(annotation.validate().is_err());
}
