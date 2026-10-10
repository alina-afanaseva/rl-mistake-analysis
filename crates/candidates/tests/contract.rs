use rl_mistake_analysis_candidates::{CandidateBatch, MistakeKind, ReplayFacts};

fn facts() -> ReplayFacts {
    serde_json::from_str(include_str!("../../../examples/bump.json")).unwrap()
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

#[test]
fn candidate_entry_point_rejects_failed_suppliers_and_duplicate_events() {
    let kind = MistakeKind::BumpingTeammate;
    let mut replay = facts();
    replay.status = "error".into();
    assert!(kind.candidates(&replay).is_err());
    replay.status = "ok".into();
    replay.events.push(replay.events[0].clone());
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
