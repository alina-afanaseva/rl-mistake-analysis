use rl_mistake_analysis_candidates::{
    MistakeKind, ReplayFacts,
    annotations::{Annotation, AnnotationDataset, ReplayReference},
};
use serde_json::json;

fn annotations() -> AnnotationDataset {
    let facts: ReplayFacts =
        serde_json::from_str(include_str!("../../../examples/bump.json")).unwrap();
    let batch = MistakeKind::BumpingTeammate.candidates(&facts).unwrap();
    AnnotationDataset {
        schema_version: 1,
        replays: vec![ReplayReference {
            sha256: batch.replay_sha256.clone(),
            rocket_league_id: None,
            download: None,
        }],
        annotations: batch
            .candidates
            .iter()
            .map(|candidate| Annotation {
                replay_sha256: batch.replay_sha256.clone(),
                incident: candidate.anchor().unwrap(),
                player_id: candidate.player_id.clone(),
                raw_time: candidate.raw_time,
                evidence: candidate.evidence.clone(),
                label: u8::from(candidate.player_id == candidate.evidence["victim"]),
                reject_reason: None,
                engine_revision: batch.engine_revision.clone(),
                source_event_ids: candidate.source_event_ids.clone(),
            })
            .collect(),
    }
}

#[test]
fn annotation_roundtrip_preserves_independent_participant_labels() {
    let original = annotations();
    let restored: AnnotationDataset =
        serde_json::from_value(serde_json::to_value(&original).unwrap()).unwrap();
    restored.validate().unwrap();
    assert_eq!(restored.annotations[0].player_id, json!({"Steam": 1}));
    assert_eq!(restored.annotations[0].label, 0);
    assert_eq!(restored.annotations[1].player_id, json!({"Steam": 2}));
    assert_eq!(restored.annotations[1].label, 1);
    let mut duplicate = restored;
    duplicate.annotations.push(duplicate.annotations[1].clone());
    assert!(duplicate.validate().is_err());
}

#[test]
fn annotations_reject_missing_null_and_unrelated_judged_players() {
    let mut unrelated = annotations();
    unrelated.annotations[0].player_id = json!({"Steam": 3});
    assert!(unrelated.validate().is_err());
    for remove in [false, true] {
        let mut document = serde_json::to_value(annotations()).unwrap();
        let review = document["annotations"][0].as_object_mut().unwrap();
        if remove {
            review.remove("player_id");
        } else {
            review.insert("player_id".into(), serde_json::Value::Null);
        }
        if let Ok(restored) = serde_json::from_value::<AnnotationDataset>(document) {
            assert!(restored.validate().is_err());
        }
    }
}
