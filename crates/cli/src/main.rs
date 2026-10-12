mod output;
mod prepare;
mod replay;

use rl_mistake_analysis_candidates::bump_features::{self, FeatureSampling, ReplayAnalysis};
use rl_mistake_analysis_candidates::{MistakeKind, ReplayFacts};
use rl_mistake_analysis_inference::{Model, manifest::Manifest};
use std::{env, fs, io, path::Path};

const HELP: &str = "Usage:
  rl-mistake-analysis outputs
  rl-mistake-analysis extract FILE.replay [--outputs TYPE,TYPE,...] > replay.json
  rl-mistake-analysis replay-data FILE.replay > replay.json
  rl-mistake-analysis events FILE.replay > events.json
  rl-mistake-analysis candidates < events.json
  rl-mistake-analysis bump-features SAMPLING.json [--player PLAYER_ID_JSON] < replay.json
  rl-mistake-analysis bump-feature-contract
  rl-mistake-analysis contract KIND
  rl-mistake-analysis contract KIND --schema bumping_teammate.context.v1
  rl-mistake-analysis validate-annotations ANNOTATIONS.json
  rl-mistake-analysis prepare ANNOTATIONS.json REPLAY_DIRECTORY OUTPUT.parquet KIND
  rl-mistake-analysis validate-manifest DIRECTORY
  rl-mistake-analysis score MODEL_DIRECTORY KIND VERSION < replay.json

outputs lists all five subtr-actor output forms and their implementation status.
extract exports selected outputs for one replay in a single JSON document.
Without --outputs, extract selects events and replay_data.
Available output types: events, replay_data, stats, stats_timeline, features.
replay-data exports the full structured subtr-actor replay, including frames and gameplay events.
events extracts subtr-actor events from a Rocket League replay.
candidates emits eight incident measurements for each participant in a teammate bump.
It requires only events and accepts both events-only and combined replay exports.
bump-features computes subject-specific measurements from combined events,replay_data exports.
SAMPLING.json must specify method: first_acceleration_peak_then_minimum; no defaults are supplied.
Without --player, bump-features emits both participant perspectives for each teammate bump.
bump-feature-contract lists the ordered feature definitions for those measurements.
contract emits a kind's feature schema and ordered feature names.
validate-annotations checks replay references and incident annotations.
prepare writes the native training table; context training uses training.bump_dataset.
validate-manifest checks the model catalog and its declared artifact hashes.
score evaluates one kind with its supplied model.";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let result = match args.as_slice() {
        [] => return Err(HELP.into()),
        [help] if help == "--help" || help == "-h" => {
            println!("{HELP}");
            return Ok(());
        }
        [command] if command == "outputs" => output::catalog(),
        [command] if command == "bump-feature-contract" => serde_json::json!({
            "input_schema": bump_features::SCHEMA,
            "feature_definitions": bump_features::definitions(),
        }),
        [command, sampling, options @ ..] if command == "bump-features" => {
            let sampling: FeatureSampling = serde_json::from_slice(&fs::read(sampling)?)?;
            let subject = match options {
                [] => None,
                [flag, id] if flag == "--player" => Some(serde_json::from_str(id)?),
                _ => return Err("expected --player followed by a JSON player ID".into()),
            };
            let analysis: ReplayAnalysis = serde_json::from_reader(io::stdin().lock())?;
            serde_json::to_value(bump_features::extract(
                &analysis,
                sampling,
                subject.as_ref(),
            )?)?
        }
        [command, path, options @ ..] if command == "extract" => {
            let selected = output::selection(options)?;
            replay::extract(&fs::read(path)?, &selected)?
        }
        [command, path] if command == "replay-data" => replay::replay_data(&fs::read(path)?)?,
        [command, path] if command == "events" => replay::events(&fs::read(path)?)?,
        [command] if command == "candidates" => {
            let replay = read_events()?;
            let batches = MistakeKind::ALL
                .into_iter()
                .map(|kind| kind.candidates(&replay))
                .collect::<Result<Vec<_>, _>>()?;
            serde_json::to_value(batches)?
        }
        [command, name, flag, schema]
            if command == "contract" && flag == "--schema" && schema == bump_features::SCHEMA =>
        {
            let kind = MistakeKind::parse(name)?;
            serde_json::json!({
                "kind": kind.name(),
                "input_schema": bump_features::SCHEMA,
                "feature_names": bump_features::definitions().into_iter().map(|d| d.name).collect::<Vec<_>>(),
                "feature_definitions": bump_features::definitions(),
            })
        }
        [command, name] if command == "contract" => {
            let kind = MistakeKind::parse(name)?;
            serde_json::json!({
                "kind": kind.name(),
                "input_schema": kind.schema(),
                "feature_names": kind.feature_names(),
            })
        }
        [command, path] if command == "validate-annotations" => {
            let dataset = prepare::load_annotations(Path::new(path))?;
            serde_json::json!({"replays": dataset.replays.len(), "annotations": dataset.annotations.len()})
        }
        [command, annotations, recordings, output, name] if command == "prepare" => {
            let count = prepare::prepare(
                Path::new(annotations),
                Path::new(recordings),
                Path::new(output),
                MistakeKind::parse(name)?,
            )?;
            serde_json::json!({"kind": name, "examples": count})
        }
        [command, directory] if command == "validate-manifest" => {
            let directory = Path::new(directory);
            let manifest = Manifest::load(directory)?;
            manifest.verify_files(directory)?;
            serde_json::json!({"version": manifest.version, "models": manifest.models.len()})
        }
        [command, directory, name, version] if command == "score" => {
            let mut model = Model::load(Path::new(directory), name, version)?;
            let kind = MistakeKind::parse(name)?;
            let predictions = match model.metadata().input_schema.as_str() {
                schema if bump_features::supports_schema(schema) => {
                    let sampling = model
                        .metadata()
                        .sampling
                        .ok_or("context model is missing its sampling contract")?;
                    let document: serde_json::Value = serde_json::from_reader(io::stdin().lock())?;
                    let analysis: ReplayAnalysis = serde_json::from_value(document.clone())?;
                    let candidates = kind.candidates(&event_facts(document)?)?;
                    let features = bump_features::extract(&analysis, sampling, None)?;
                    model.score_context(candidates, features)?
                }
                "bumping_teammate.native.v1" => {
                    model.score(kind.initiator_candidates(&read_events()?)?)?
                }
                _ => return Err("unsupported model input schema".into()),
            };
            serde_json::to_value(predictions)?
        }
        _ => return Err(HELP.into()),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn read_events() -> Result<ReplayFacts, Box<dyn std::error::Error>> {
    event_facts(serde_json::from_reader(io::stdin().lock())?)
}

fn event_facts(mut document: serde_json::Value) -> Result<ReplayFacts, Box<dyn std::error::Error>> {
    if document.get("outputs").is_some() {
        let events = document
            .pointer_mut("/outputs/events/events/events")
            .ok_or("extracted replay must include a valid events output")?
            .take();
        document["events"] = events;
    }
    Ok(serde_json::from_value(document)?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rl-mistake-analysis: {error}");
        std::process::exit(1);
    }
}
