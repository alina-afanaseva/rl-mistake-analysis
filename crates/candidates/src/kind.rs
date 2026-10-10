use crate::bump_features::{self, FeatureDefinition, FeatureSampling};
use crate::input::ReplayFacts;
use crate::native_bump;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistakeKind {
    BumpingTeammate,
}

impl MistakeKind {
    pub const ALL: [Self; 1] = [Self::BumpingTeammate];

    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "bumping_teammate" => Ok(Self::BumpingTeammate),
            _ => Err(format!("unknown mistake kind: {name}")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::BumpingTeammate => "bumping_teammate",
        }
    }

    pub fn schema(self) -> &'static str {
        match self {
            Self::BumpingTeammate => native_bump::SCHEMA,
        }
    }

    pub fn feature_names(self) -> &'static [&'static str] {
        match self {
            Self::BumpingTeammate => native_bump::FEATURES,
        }
    }

    pub fn validate_contract(self, schema: &str, feature_names: &[String]) -> Result<(), String> {
        let compatible = match schema {
            native_bump::SCHEMA => feature_names == self.feature_names(),
            bump_features::SCHEMA => bump_features::definitions()
                .iter()
                .map(|definition| &definition.name)
                .eq(feature_names),
            _ => false,
        };
        if !compatible {
            return Err(format!("incompatible feature contract for {}", self.name()));
        }
        Ok(())
    }

    pub fn candidates(self, replay: &ReplayFacts) -> Result<CandidateBatch, String> {
        replay.validate()?;
        let candidates = match self {
            Self::BumpingTeammate => native_bump::candidates(replay)?,
        };
        let batch = CandidateBatch {
            kind: self.name().into(),
            input_schema: self.schema().into(),
            feature_names: self
                .feature_names()
                .iter()
                .map(|name| (*name).into())
                .collect(),
            engine_revision: replay.revision.clone(),
            replay_sha256: replay.replay_sha256.clone(),
            sampling: None,
            feature_definitions: Vec::new(),
            candidates,
        };
        batch.validate()?;
        Ok(batch)
    }

    /// Preserve the subject convention of the eight-feature training contract.
    pub fn initiator_candidates(self, replay: &ReplayFacts) -> Result<CandidateBatch, String> {
        let mut batch = self.candidates(replay)?;
        batch
            .candidates
            .retain(|candidate| candidate.evidence.get("initiator") == Some(&candidate.player_id));
        Ok(batch)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub kind: String,
    pub source_event_ids: Vec<String>,
    pub frame: u64,
    pub raw_time: f64,
    pub player_id: Value,
    pub features: Vec<Option<f64>>,
    pub evidence: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateBatch {
    pub kind: String,
    pub input_schema: String,
    pub feature_names: Vec<String>,
    pub engine_revision: String,
    pub replay_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<FeatureSampling>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_definitions: Vec<FeatureDefinition>,
    pub candidates: Vec<Candidate>,
}

impl CandidateBatch {
    pub fn validate(&self) -> Result<(), String> {
        let kind = MistakeKind::parse(&self.kind)?;
        if self.input_schema != kind.schema()
            || self.feature_names != kind.feature_names()
            || self.sampling.is_some()
            || !self.feature_definitions.is_empty()
        {
            return Err("candidate feature definitions disagree with the input schema".into());
        }
        if self.engine_revision.trim().is_empty() || !crate::input::is_sha256(&self.replay_sha256) {
            return Err("missing engine revision or invalid replay fingerprint".into());
        }
        let mut identities = HashSet::new();
        let mut incidents = HashSet::new();
        for candidate in &self.candidates {
            if candidate.kind != self.kind
                || candidate.player_id.is_null()
                || !candidate.raw_time.is_finite()
                || candidate.features.len() != self.feature_names.len()
                || candidate
                    .features
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err("invalid candidate measurements or identity".into());
            }
            native_bump::validate_evidence(&candidate.evidence)?;
            for (index, value) in candidate.features.iter().enumerate() {
                if value.is_none() {
                    return Err(format!(
                        "invalid measurement or subject for {}",
                        self.feature_names[index]
                    ));
                }
            }
            let subject =
                serde_json::to_string(&candidate.player_id).map_err(|error| error.to_string())?;
            if !incidents.insert((candidate.anchor()?.key()?, subject.clone())) {
                return Err("duplicate incident anchor".into());
            }
            let mut ids: Vec<_> = candidate.source_event_ids.iter().collect();
            ids.sort();
            if ids.is_empty()
                || ids.iter().any(|id| id.trim().is_empty())
                || ids.windows(2).any(|pair| pair[0] == pair[1])
                || !identities.insert((ids, subject))
            {
                return Err("candidate event identities must be nonempty and unique".into());
            }
        }
        Ok(())
    }
}

impl Candidate {
    pub fn anchor(&self) -> Result<crate::annotations::IncidentAnchor, String> {
        match MistakeKind::parse(&self.kind)? {
            MistakeKind::BumpingTeammate => native_bump::anchor(self),
        }
    }
}
