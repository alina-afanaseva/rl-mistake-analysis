use crate::output::OutputKind;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use subtr_actor::{
    Collector, NDArrayCollector, ReplayDataCollector, ReplayProcessor, StatsCollector,
    StatsFrameResolution, StatsTimelineCollector, StatsTimelineEventCollector,
};

type Error = Box<dyn std::error::Error>;

struct Extraction {
    identity: Value,
    outputs: Map<String, Value>,
}

pub fn extract(bytes: &[u8], selected: &[OutputKind]) -> Result<Value, Error> {
    let extracted = collect(bytes, selected)?;
    let mut document = extracted.identity;
    document["outputs"] = Value::Object(extracted.outputs);
    Ok(document)
}

pub fn replay_data(bytes: &[u8]) -> Result<Value, Error> {
    let mut extracted = collect(bytes, &[OutputKind::ReplayData])?;
    Ok(extracted
        .outputs
        .remove(OutputKind::ReplayData.name())
        .ok_or("structured replay output missing")?)
}

pub fn events(bytes: &[u8]) -> Result<Value, Error> {
    let mut extracted = collect(bytes, &[OutputKind::Events])?;
    let mut document = extracted.identity;
    document["events"] = extracted
        .outputs
        .get_mut(OutputKind::Events.name())
        .and_then(|timeline| timeline.pointer_mut("/events/events"))
        .ok_or("analysis event output missing")?
        .take();
    Ok(document)
}

fn collect(bytes: &[u8], selected: &[OutputKind]) -> Result<Extraction, Error> {
    let replay = boxcars::ParserBuilder::new(bytes)
        .must_parse_network_data()
        .on_error_check_crc()
        .parse()?;
    let rocket_league_id = replay
        .properties
        .iter()
        .find(|(name, _)| name == "Id")
        .map(|(_, property)| property)
        .and_then(|property| match property {
            boxcars::HeaderProp::Str(value) => Some(value.clone()),
            _ => None,
        });
    let mut events = selected.contains(&OutputKind::Events).then(|| {
        StatsTimelineEventCollector::new()
            .with_frame_resolution(StatsFrameResolution::TimeStep { seconds: 1.0 })
    });
    let mut replay_data = selected
        .contains(&OutputKind::ReplayData)
        .then(ReplayDataCollector::new);
    let mut stats = selected
        .contains(&OutputKind::Stats)
        .then(StatsCollector::new);
    let mut stats_timeline = selected
        .contains(&OutputKind::StatsTimeline)
        .then(StatsTimelineCollector::new);
    let mut features = selected
        .contains(&OutputKind::Features)
        .then(NDArrayCollector::<f32>::default);
    let mut processor = ReplayProcessor::new(&replay).map_err(extraction_error)?;
    {
        let mut collectors: Vec<&mut dyn Collector> = [
            events.as_mut().map(|c| c as &mut dyn Collector),
            replay_data.as_mut().map(|c| c as &mut dyn Collector),
            stats.as_mut().map(|c| c as &mut dyn Collector),
            stats_timeline.as_mut().map(|c| c as &mut dyn Collector),
            features.as_mut().map(|c| c as &mut dyn Collector),
        ]
        .into_iter()
        .flatten()
        .collect();
        processor
            .process_all(&mut collectors)
            .map_err(extraction_error)?;
    }
    let mut outputs = Map::new();
    if let Some(collector) = events {
        outputs.insert(
            OutputKind::Events.name().into(),
            serialize_output(collector.into_replay_stats_timeline_scaffold())?,
        );
    }
    if let Some(collector) = replay_data {
        outputs.insert(
            OutputKind::ReplayData.name().into(),
            serialize_output(collector.into_replay_data(processor))?,
        );
    }
    if let Some(collector) = stats {
        outputs.insert(
            OutputKind::Stats.name().into(),
            serialize_output(collector.into_stats())?,
        );
    }
    if let Some(collector) = stats_timeline {
        outputs.insert(
            OutputKind::StatsTimeline.name().into(),
            serialize_output(collector.into_legacy_replay_stats_timeline())?,
        );
    }
    if let Some(collector) = features {
        let (metadata, matrix) = collector.get_meta_and_ndarray().map_err(extraction_error)?;
        let rows: Vec<_> = matrix.rows().into_iter().map(|row| row.to_vec()).collect();
        outputs.insert(
            OutputKind::Features.name().into(),
            json!({"metadata": metadata, "matrix": rows}),
        );
    }
    let identity = json!({
        "name": "subtr-actor",
        "schema_version": 1,
        "status": "ok",
        "revision": env!("SUBTR_ACTOR_REVISION"),
        "replay_sha256": format!("{:x}", Sha256::digest(bytes)),
        "rocket_league_id": rocket_league_id,
    });
    Ok(Extraction { identity, outputs })
}

fn serialize_output<T: serde::Serialize>(
    result: subtr_actor::SubtrActorResult<T>,
) -> Result<Value, Error> {
    Ok(serde_json::to_value(result.map_err(extraction_error)?)?)
}

fn extraction_error(error: subtr_actor::SubtrActorError) -> String {
    format!("subtr-actor extraction failed: {error:?}")
}
