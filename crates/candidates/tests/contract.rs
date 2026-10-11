use rl_mistake_analysis_candidates::{CandidateBatch, MistakeKind, ReplayFacts};
use serde_json::json;

fn facts() -> ReplayFacts {
    serde_json::from_str(include_str!("../../../examples/bump.json")).unwrap()
}

#[test]
fn context_models_select_unique_known_features_in_manifest_order() {
    use rl_mistake_analysis_candidates::bump_features;

    let kind = MistakeKind::BumpingTeammate;
    let names = vec!["p_boost_before_pct".into(), "p_is_initiator".into()];
    for schema in [
        bump_features::SCHEMA,
        "bumping_teammate.context.v1.expanded90.v1",
        "bumping_teammate.context.v1.custom.v1",
    ] {
        kind.validate_contract(schema, &names).unwrap();
    }
    let selected = bump_features::select_definitions(&names).unwrap();
    assert_eq!(selected[0].name, names[0]);
    assert_eq!(selected[1].name, names[1]);
    for invalid in [
        vec![],
        vec!["unknown_feature".into()],
        vec!["p_is_initiator".into(), "p_is_initiator".into()],
    ] {
        assert!(
            kind.validate_contract(bump_features::SCHEMA, &invalid)
                .is_err()
        );
    }
    for schema in [
        "bumping_teammate.context.v2",
        "bumping_teammate.context.v1.custom.v2",
        "bumping_teammate.context.v1..v1",
        "bumping_teammate.context.v1.custom.extra.v1",
    ] {
        assert!(kind.validate_contract(schema, &names).is_err());
    }
}

#[test]
fn events_only_candidates_include_both_subjects_with_eight_incident_features() {
    let kind = MistakeKind::BumpingTeammate;
    let batch = kind.candidates(&facts()).unwrap();
    assert_eq!(batch.candidates.len(), 2);
    assert_eq!(batch.feature_names.len(), 8);
    assert!(batch.sampling.is_none());
    assert!(batch.feature_definitions.is_empty());
    let initiator = &batch.candidates[0];
    let victim = &batch.candidates[1];
    assert_eq!(initiator.player_id, initiator.evidence["initiator"]);
    assert_eq!(victim.player_id, victim.evidence["victim"]);
    assert_eq!(initiator.features, victim.features);
    assert_eq!(initiator.evidence, victim.evidence);
    assert_eq!(initiator.anchor().unwrap(), victim.anchor().unwrap());
    assert_eq!(initiator.source_event_ids, victim.source_event_ids);
    assert_eq!(
        kind.initiator_candidates(&facts())
            .unwrap()
            .candidates
            .len(),
        1
    );
}

fn repeat(replay: &mut ReplayFacts, frame: u64, reversed: bool) {
    let mut event = replay.events[0].clone();
    event.meta.id = format!("bump:{frame}");
    let bump = &mut event.payload["payload"];
    bump["frame"] = json!(frame);
    bump["time"] = json!(10.0 + (frame - 42) as f64 / 30.0);
    if reversed {
        for (left, right) in [
            ("initiator", "victim"),
            ("initiator_position", "victim_position"),
        ] {
            let value = bump[left].clone();
            bump[left] = bump[right].clone();
            bump[right] = value;
        }
    }
    replay.events.push(event);
}

#[test]
fn role_reversal_keeps_one_contact_with_both_subjects_and_original_identity() {
    let kind = MistakeKind::BumpingTeammate;
    let original = kind.candidates(&facts()).unwrap();
    let mut replay = facts();
    repeat(&mut replay, 43, true);
    replay.events.reverse();
    let batch = kind.candidates(&replay).unwrap();
    assert_eq!(
        serde_json::to_value(batch).unwrap(),
        serde_json::to_value(original).unwrap()
    );
    assert_eq!(
        kind.initiator_candidates(&replay).unwrap().candidates.len(),
        1
    );
}

#[test]
fn repeat_window_is_inclusive_and_preserves_later_collisions_and_other_pairs() {
    let kind = MistakeKind::BumpingTeammate;
    for reversed in [false, true] {
        let mut replay = facts();
        repeat(&mut replay, 52, reversed);
        assert_eq!(kind.candidates(&replay).unwrap().candidates.len(), 2);
        repeat(&mut replay, 63, reversed);
        let batch = kind.candidates(&replay).unwrap();
        assert_eq!(batch.candidates.len(), 4);
        assert_eq!(batch.candidates[2].frame, 63);
    }
    let mut replay = facts();
    repeat(&mut replay, 43, false);
    replay.events[1].payload["payload"]["victim"] = json!({"Steam": 3});
    assert_eq!(kind.candidates(&replay).unwrap().candidates.len(), 4);
}

#[test]
fn suppressed_reversals_extend_the_contact_repeat_window() {
    let mut replay = facts();
    repeat(&mut replay, 43, true);
    repeat(&mut replay, 53, false);
    repeat(&mut replay, 64, true);
    let batch = MistakeKind::BumpingTeammate.candidates(&replay).unwrap();
    assert_eq!(batch.candidates.len(), 4);
    assert_eq!(batch.candidates[0].frame, 42);
    assert_eq!(batch.candidates[2].frame, 64);
}

#[test]
fn suppressed_contacts_still_require_valid_measurements() {
    let mut replay = facts();
    repeat(&mut replay, 43, true);
    replay.events[1].payload["payload"]["confidence"] = json!("invalid");
    assert!(MistakeKind::BumpingTeammate.candidates(&replay).is_err());
}

#[test]
fn candidate_entry_point_rejects_failed_suppliers_and_duplicate_events() {
    let kind = MistakeKind::BumpingTeammate;
    let mut replay = facts();
    replay.status = "error".into();
    assert!(kind.candidates(&replay).is_err());
    replay.status = "ok".into();
    replay.events.push(replay.events[0].clone());
    assert!(kind.candidates(&replay).is_err());
    replay.events[1].meta.id = "different-event-id".into();
    assert!(kind.candidates(&replay).is_err());
}

#[test]
fn serialized_batches_reject_changed_feature_contracts_and_duplicate_contacts() {
    let batch = MistakeKind::BumpingTeammate.candidates(&facts()).unwrap();
    let mut restored: CandidateBatch =
        serde_json::from_str(&serde_json::to_string(&batch).unwrap()).unwrap();
    restored.validate().unwrap();
    restored.feature_names.reverse();
    assert!(restored.validate().is_err());
    restored = batch.clone();
    restored.input_schema = "bumping_teammate.native.v2".into();
    assert!(restored.validate().is_err());
    restored = batch.clone();
    restored.candidates.push(restored.candidates[0].clone());
    assert!(restored.validate().is_err());
    restored = batch;
    restored.candidates[0].features[0] = Some(f64::INFINITY);
    assert!(restored.validate().is_err());
}
