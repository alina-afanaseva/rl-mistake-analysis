//! Rocket League mistake candidates and features from subtr-actor replay facts.

pub mod annotations;
pub mod bang_features;
mod bump_contacts;
pub mod bump_features;
mod feature_context;
pub mod feature_contract;
mod feature_input;
mod feature_values;
pub mod input;
pub mod kind;
mod native_bump;
mod native_touch;

pub use input::ReplayFacts;
pub use kind::{Candidate, CandidateBatch, KindDefinition, MistakeKind};
