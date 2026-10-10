mod output;
mod prepare;
mod replay;

use rl_mistake_analysis_candidates::{MistakeKind, ReplayFacts};
use rl_mistake_analysis_inference::{Model, manifest::Manifest};
use std::{
    env, fs,
    io::{self, Read},
    path::Path,
};

const HELP: &str = "Usage:
  rl-mistake-analysis outputs
  rl-mistake-analysis extract FILE.replay [--outputs TYPE,TYPE,...] > replay.json
  rl-mistake-analysis replay-data FILE.replay > replay.json
  rl-mistake-analysis events FILE.replay > events.json
  rl-mistake-analysis candidates < events.json
  rl-mistake-analysis contract KIND
  rl-mistake-analysis validate-annotations ANNOTATIONS.json
  rl-mistake-analysis prepare ANNOTATIONS.json REPLAY_DIRECTORY OUTPUT.parquet KIND
  rl-mistake-analysis validate-manifest DIRECTORY
  rl-mistake-analysis score MODEL_DIRECTORY KIND VERSION < events.json

outputs lists all five subtr-actor output forms and their implementation status.
extract exports selected outputs for one replay in a single JSON document.
Without --outputs, extract selects events and replay_data.
Available output types: events, replay_data, stats, stats_timeline, features.
replay-data exports the full structured subtr-actor replay, including frames and gameplay events.
events extracts subtr-actor events from a Rocket League replay.
candidates emits candidates for each implemented mistake kind; it also accepts extract JSON containing events.
contract emits a kind's feature schema and ordered feature names.
validate-annotations checks replay references and incident annotations.
prepare extracts features for annotated incidents and writes a Parquet table.
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
            let replay = read_events()?;
            let kind = MistakeKind::parse(name)?;
            serde_json::to_value(model.score(kind.candidates(&replay)?)?)?
        }
        _ => return Err(HELP.into()),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn read_events() -> Result<ReplayFacts, Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let mut document: serde_json::Value = serde_json::from_str(&text)?;
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
