//! Subject-specific teammate-bump measurements from combined replay analysis.

mod input;
mod sampling;
mod schema;

pub use input::ReplayAnalysis;
pub use sampling::{FeatureSampling, ParticipantSamples, SamplingMethod, SamplingStatus};
pub use schema::{DataType, FeatureDefinition, definitions};

use crate::bump_contacts::{Contact, unique_contact_ids};
use input::{Body, Event, PadEventKind, PlayerData, PlayerFrame, Vector};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};

pub const SCHEMA: &str = "bumping_teammate.context.v1";
const BOOST_RAW_MAX: f64 = u8::MAX as f64;

/// Named v1 subsets use the same measurements as the base context contract.
pub fn supports_schema(schema: &str) -> bool {
    schema == SCHEMA
        || schema
            .strip_prefix(SCHEMA)
            .and_then(|suffix| suffix.strip_prefix('.'))
            .and_then(|suffix| suffix.strip_suffix(".v1"))
            .is_some_and(|name| {
                !name.is_empty()
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            })
}

/// Resolve unique model inputs in their declared order.
pub fn select_definitions(names: &[String]) -> Result<Vec<FeatureDefinition>, String> {
    if names.is_empty() {
        return Err("context models require at least one feature".into());
    }
    let mut available: HashMap<_, _> = definitions()
        .into_iter()
        .map(|definition| (definition.name.clone(), definition))
        .collect();
    names
        .iter()
        .map(|name| {
            available
                .remove(name)
                .ok_or_else(|| format!("unknown or duplicate context feature: {name}"))
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Measurement {
    Boolean(bool),
    Count(u32),
    Float(f32),
}

#[derive(Debug, Serialize)]
pub struct FeatureBatch {
    pub input_schema: &'static str,
    pub engine_revision: String,
    pub replay_sha256: String,
    pub sampling: FeatureSampling,
    pub feature_definitions: Vec<FeatureDefinition>,
    pub rows: Vec<FeatureRow>,
}

#[derive(Debug, Serialize)]
pub struct FeatureRow {
    pub source_event_id: String,
    pub frame: usize,
    pub raw_time: f64,
    pub player_id: Value,
    pub teammate_id: Value,
    pub initiator: Value,
    pub victim: Value,
    pub samples: SampleFrames,
    pub features: BTreeMap<String, Option<Measurement>>,
}

#[derive(Debug, Serialize)]
pub struct SampleFrames {
    pub play_start: f64,
    pub play_end: f64,
    pub p: ParticipantSamples,
    pub q: ParticipantSamples,
}

struct Context<'a> {
    analysis: &'a ReplayAnalysis,
    streams: HashMap<&'a str, Vec<&'a Event>>,
}

impl<'a> Context<'a> {
    fn events(&self, stream: &str) -> &[&'a Event] {
        match self.streams.get(stream) {
            Some(events) => events,
            None => &[],
        }
    }

    fn player(&self, id: &Value) -> Result<&'a PlayerData, String> {
        self.analysis
            .outputs
            .replay_data
            .frame_data
            .players
            .iter()
            .find(|(player, _)| player == id)
            .map(|(_, data)| data)
            .ok_or_else(|| format!("player frames are missing for {id}"))
    }

    fn next(
        &self,
        stream: &str,
        time_field: &str,
        start: f64,
        end: f64,
        participant: Option<(&str, &Value)>,
    ) -> Result<Option<&'a Event>, String> {
        let mut next: Option<&Event> = None;
        for event in self.events(stream) {
            if let Some((field, player)) = participant
                && event.participant(field)? != player
            {
                continue;
            }
            let time = event.number(time_field)?;
            if time >= start
                && time <= end
                && (next.is_none() || time < next.unwrap().number(time_field)?)
            {
                next = Some(event);
            }
        }
        Ok(next)
    }

    fn state(
        &self,
        stream: &str,
        player: Option<&Value>,
        time: f64,
    ) -> Result<Option<&'a str>, String> {
        let mut state = None;
        for event in self.events(stream) {
            if let Some(player) = player
                && event.participant("player")? != player
            {
                continue;
            }
            if event.number("time")? < time && event.number("end_time")? >= time {
                if state.is_some() {
                    return Err(format!("overlapping {stream} state spans at {time}"));
                }
                state = Some(event.text(if stream == "possession" {
                    "possession_state"
                } else {
                    "state"
                })?);
            }
        }
        Ok(state)
    }

    fn bounds(&self, time: f64) -> Result<(f64, f64), String> {
        let metadata = &self.analysis.outputs.replay_data.frame_data.metadata_frames;
        let mut start = metadata[0].time;
        let mut end = metadata.last().unwrap().time;
        for goal in self.events("goal_context") {
            let goal_time = goal.number("time")?;
            if goal_time < time {
                start = start.max(goal_time);
            } else {
                end = end.min(goal_time);
            }
        }
        for kickoff in self.events("kickoff") {
            let kickoff_start = kickoff.number("start_time")?;
            if kickoff_start > time {
                end = end.min(kickoff_start);
            } else if kickoff_start >= start {
                start = start.max(kickoff.number("live_action_start_time")?);
            }
        }
        if start > time || end < time {
            return Err("collision is outside its observed live-play segment".into());
        }
        Ok((start, end))
    }

    fn samples(
        &self,
        bump: &Event,
        player: &Value,
        teammate: &Value,
        sampling: FeatureSampling,
    ) -> Result<SampleFrames, String> {
        let time = bump.number("time")?;
        let frame = bump.frame("frame")?;
        let (play_start, play_end) = self.bounds(time)?;
        let frames = &self.analysis.outputs.replay_data.frame_data.metadata_frames;
        Ok(SampleFrames {
            play_start,
            play_end,
            p: sampling::measure(
                self.player(player)?,
                frames,
                frame,
                play_start,
                play_end,
                sampling,
            ),
            q: sampling::measure(
                self.player(teammate)?,
                frames,
                frame,
                play_start,
                play_end,
                sampling,
            ),
        })
    }
}

/// Compute both participant perspectives, or only the explicitly selected player.
pub fn extract(
    analysis: &ReplayAnalysis,
    sampling: FeatureSampling,
    subject: Option<&Value>,
) -> Result<FeatureBatch, String> {
    analysis.validate()?;
    let mut streams: HashMap<&str, Vec<&Event>> = HashMap::new();
    for event in &analysis.outputs.events.events.events {
        let expected = match event.meta.stream.as_str() {
            "air_dribble" => "ball_carry",
            "boost_pickups" => "boost_pickup",
            stream => stream,
        };
        if matches!(
            event.meta.stream.as_str(),
            "bump"
                | "touch"
                | "whiff"
                | "beaten_to_ball"
                | "possession"
                | "goal_context"
                | "kickoff"
                | "movement"
                | "ball_carry"
                | "air_dribble"
                | "pass"
                | "shadow_defense"
                | "field_half"
                | "ball_depth"
                | "depth_role"
                | "boost_pickups"
                | "demolition"
        ) && event.payload.kind != expected
        {
            return Err(format!(
                "event {} has inconsistent stream and payload kind",
                event.meta.id
            ));
        }
        streams.entry(&event.meta.stream).or_default().push(event);
    }
    let mut contacts = Vec::new();
    for bump in streams.get("bump").into_iter().flatten() {
        let initiator = bump.participant("initiator")?;
        let victim = bump.participant("victim")?;
        let team = bump.boolean("initiator_is_team_0")?;
        let team_bump = bump.boolean("is_team_bump")?;
        if team_bump != (team == bump.boolean("victim_is_team_0")?) || initiator == victim {
            return Err(format!(
                "event {} has inconsistent bump participants or teams",
                bump.meta.id
            ));
        }
        if !team_bump {
            continue;
        }
        let frame = bump.frame("frame")?;
        let time = bump.number("time")?;
        if analysis
            .outputs
            .replay_data
            .frame_data
            .metadata_frames
            .get(frame)
            .is_none_or(|metadata| metadata.time != time)
        {
            return Err(format!(
                "event {} disagrees with native frame time",
                bump.meta.id
            ));
        }
        contacts.push(Contact {
            id: &bump.meta.id,
            frame: frame as u64,
            initiator,
            victim,
        });
    }
    let selected = unique_contact_ids(contacts.into_iter())?;
    if let Some(bumps) = streams.get_mut("bump") {
        bumps.retain(|bump| {
            bump.boolean("is_team_bump") == Ok(false) || selected.contains(&bump.meta.id)
        });
    }
    let context = Context { analysis, streams };
    if let Some(subject) = subject {
        context.player(subject)?;
    }
    let feature_definitions = definitions();
    let mut rows = Vec::new();
    for bump in context.events("bump") {
        if !bump.boolean("is_team_bump")? {
            continue;
        }
        let initiator = bump.participant("initiator")?;
        let victim = bump.participant("victim")?;
        let team = bump.boolean("initiator_is_team_0")?;
        let frame = bump.frame("frame")?;
        let time = bump.number("time")?;
        for (player, teammate) in [(initiator, victim), (victim, initiator)] {
            if subject.is_some_and(|subject| subject != player) {
                continue;
            }
            let samples = context.samples(bump, player, teammate, sampling)?;
            let mut values = Values::default();
            global_features(
                &context,
                bump,
                player,
                teammate,
                team,
                &samples,
                &mut values,
            )?;
            participant_features(&context, "p", player, teammate, bump, &samples, &mut values)?;
            participant_features(&context, "q", teammate, player, bump, &samples, &mut values)?;
            values.validate(&feature_definitions)?;
            rows.push(FeatureRow {
                source_event_id: bump.meta.id.clone(),
                frame,
                raw_time: time,
                player_id: player.clone(),
                teammate_id: teammate.clone(),
                initiator: initiator.clone(),
                victim: victim.clone(),
                samples,
                features: values.0,
            });
        }
    }
    rows.sort_by(|left, right| {
        left.frame
            .cmp(&right.frame)
            .then_with(|| left.source_event_id.cmp(&right.source_event_id))
            .then_with(|| left.player_id.to_string().cmp(&right.player_id.to_string()))
    });
    Ok(FeatureBatch {
        input_schema: SCHEMA,
        engine_revision: analysis.revision.clone(),
        replay_sha256: analysis.replay_sha256.clone(),
        sampling,
        feature_definitions,
        rows,
    })
}

#[derive(Default)]
struct Values(BTreeMap<String, Option<Measurement>>);

impl Values {
    fn boolean(&mut self, name: &str, value: Option<bool>) {
        self.0.insert(name.into(), value.map(Measurement::Boolean));
    }

    fn float(&mut self, name: &str, value: Option<f64>) -> Result<(), String> {
        if value.is_some_and(|value| !value.is_finite() || !(value as f32).is_finite()) {
            return Err(format!("nonfinite or unrepresentable measurement: {name}"));
        }
        self.0.insert(
            name.into(),
            value.map(|value| Measurement::Float(value as f32)),
        );
        Ok(())
    }

    fn category(
        &mut self,
        prefix: &str,
        state: Option<&str>,
        indicators: &[&str],
        reference: &str,
    ) -> Result<(), String> {
        if let Some(state) = state
            && state != reference
            && !indicators.contains(&state)
        {
            return Err(format!("unsupported category {prefix}: {state}"));
        }
        for indicator in indicators {
            self.boolean(
                &format!("{prefix}_{indicator}"),
                state.map(|state| state == *indicator),
            );
        }
        Ok(())
    }

    fn vector(
        &mut self,
        prefix: &str,
        value: Option<Vector>,
        toward: Option<Vector>,
    ) -> Result<(), String> {
        let horizontal = value.zip(toward).map(|(value, basis)| value.project(basis));
        for (index, axis) in ["toward", "side", "up"].iter().enumerate() {
            self.float(
                &format!("{prefix}_{axis}"),
                if *axis == "up" {
                    value.map(|vector| vector.z)
                } else {
                    horizontal.map(|vector| vector[index])
                },
            )?;
        }
        Ok(())
    }

    fn validate(&self, definitions: &[FeatureDefinition]) -> Result<(), String> {
        if self.0.len() != definitions.len() {
            return Err("feature row does not match its declared contract".into());
        }
        for definition in definitions {
            let value = self
                .0
                .get(&definition.name)
                .ok_or_else(|| format!("missing feature {}", definition.name))?;
            let valid = match value {
                None => definition.nullable,
                Some(Measurement::Boolean(_)) => definition.data_type == DataType::Boolean,
                Some(Measurement::Count(_)) => definition.data_type == DataType::U32,
                Some(Measurement::Float(_)) => definition.data_type == DataType::F32,
            };
            if !valid {
                return Err(format!(
                    "invalid feature type or nullability: {}",
                    definition.name
                ));
            }
        }
        Ok(())
    }
}

fn actor<'a>(
    event: &Event,
    player: &Value,
    teammate: &Value,
    team: bool,
) -> Result<&'a str, String> {
    let toucher = event.participant("player")?;
    Ok(if toucher == player {
        "p"
    } else if toucher == teammate {
        "q"
    } else if event.boolean("is_team_0")? == team {
        "other_teammate"
    } else {
        "opponent"
    })
}

fn possession_state(state: &str, team: bool) -> Result<&'static str, String> {
    match state {
        "neutral" => Ok("neutral"),
        "team_zero" => Ok(if team { "own_team" } else { "opponent" }),
        "team_one" => Ok(if team { "opponent" } else { "own_team" }),
        _ => Err(format!("unsupported possession state: {state}")),
    }
}

fn global_features(
    context: &Context<'_>,
    bump: &Event,
    player: &Value,
    teammate: &Value,
    team: bool,
    samples: &SampleFrames,
    values: &mut Values,
) -> Result<(), String> {
    let time = bump.number("time")?;
    values.boolean(
        "p_is_initiator",
        Some(bump.participant("initiator")? == player),
    );
    values.float("contact_gap", Some(bump.number("contact_distance")?))?;
    let next_touch = context.next("touch", "time", time, samples.play_end, None)?;
    for group in ["p", "q", "other_teammate", "opponent"] {
        let mut touched = false;
        if let Some(next_touch) = next_touch {
            let frame = next_touch.frame("frame")?;
            for touch in context.events("touch") {
                if touch.frame("frame")? == frame && actor(touch, player, teammate, team)? == group
                {
                    touched = true;
                }
            }
        }
        values.boolean(&format!("first_touch_{group}"), Some(touched));
    }
    values.float(
        "first_touch_delay_s",
        next_touch
            .map(|event| event.number("time").map(|next| next - time))
            .transpose()?,
    )?;
    let before = context
        .state("possession", None, time)?
        .ok_or("missing pre-bump possession state")?;
    let before = possession_state(before, team)?;
    values.category(
        "possession_before",
        Some(before),
        &["own_team", "opponent"],
        "neutral",
    )?;
    let mut change: Option<(f64, &str)> = None;
    for event in context.events("possession") {
        let start = event.number("time")?;
        let state = possession_state(event.text("possession_state")?, team)?;
        if start >= time
            && start < samples.play_end
            && state != before
            && change.is_none_or(|(previous, _)| start < previous)
        {
            change = Some((start, state));
        }
    }
    values.category(
        "next_possession",
        Some(match change {
            Some((_, state)) => state,
            None => "no_change",
        }),
        &["own_team", "opponent", "neutral"],
        "no_change",
    )?;
    values.float(
        "possession_change_delay_s",
        change.map(|(start, _)| start - time),
    )?;
    let goal = context.next("goal_context", "time", time, samples.play_end, None)?;
    for (name, same_team) in [("own_team", true), ("opponent", false)] {
        values.boolean(
            &format!("next_goal_{name}"),
            Some(match goal {
                Some(event) => (event.boolean("scoring_team_is_team_0")? == team) == same_team,
                None => false,
            }),
        );
    }
    values.float(
        "next_goal_delay_s",
        goal.map(|event| event.number("time").map(|goal| goal - time))
            .transpose()?,
    )?;
    let count = if let Some(goal) = goal {
        let mut touches = HashSet::new();
        for touch in context.events("touch") {
            let touch_time = touch.number("time")?;
            if touch_time > time && touch_time < goal.number("time")? {
                let id = touch
                    .field("touch_id")?
                    .as_u64()
                    .ok_or("scoring sequence has an unidentified touch")?;
                touches.insert(id);
            }
        }
        Some(Measurement::Count(
            u32::try_from(touches.len()).map_err(|error| error.to_string())?,
        ))
    } else {
        None
    };
    values.0.insert("touches_before_next_goal".into(), count);
    Ok(())
}

fn body(player: &PlayerData, frame: Option<usize>) -> Option<&Body> {
    frame.and_then(|frame| player.frames[frame].body())
}

fn physical_features(
    context: &Context<'_>,
    prefix: &str,
    player: &Value,
    other: &Value,
    samples: &ParticipantSamples,
    values: &mut Values,
) -> Result<(), String> {
    let player_data = context.player(player)?;
    let before = body(player_data, samples.before_frame);
    let after = body(player_data, samples.after_frame);
    let other_before = body(context.player(other)?, samples.before_frame);
    let toward = before
        .zip(other_before)
        .and_then(|(left, right)| right.location.subtract(left.location).horizontal_unit());
    let velocity_before = before.and_then(|body| body.linear_velocity);
    let velocity_after = after.and_then(|body| body.linear_velocity);
    values.vector(
        &format!("{prefix}_velocity_before"),
        velocity_before,
        toward,
    )?;
    values.vector(
        &format!("{prefix}_velocity_delta"),
        velocity_after
            .zip(velocity_before)
            .map(|(after, before)| after.subtract(before)),
        toward,
    )?;
    values.vector(
        &format!("{prefix}_position_delta"),
        after
            .zip(before)
            .map(|(after, before)| after.location.subtract(before.location)),
        toward,
    )?;
    let balls = &context
        .analysis
        .outputs
        .replay_data
        .frame_data
        .ball_data
        .frames;
    let ball_before = samples.before_frame.and_then(|frame| balls[frame].body());
    let ball_after = samples.after_frame.and_then(|frame| balls[frame].body());
    let distance_before = before
        .zip(ball_before)
        .map(|(car, ball)| car.location.horizontal_distance(ball.location));
    let distance_after = after
        .zip(ball_after)
        .map(|(car, ball)| car.location.horizontal_distance(ball.location));
    values.float(&format!("{prefix}_ball_distance_before"), distance_before)?;
    values.float(
        &format!("{prefix}_ball_distance_delta"),
        distance_after
            .zip(distance_before)
            .map(|(after, before)| after - before),
    )?;
    let heading = before
        .map(|body| body.rotation.forward())
        .transpose()?
        .and_then(Vector::horizontal_unit);
    values.float(
        &format!("{prefix}_heading_alignment_before"),
        heading
            .zip(toward)
            .map(|(heading, basis)| heading.project(basis)[0]),
    )?;
    values.float(
        &format!("{prefix}_upright_after"),
        after.map(|body| body.rotation.upright()).transpose()?,
    )?;
    Ok(())
}

fn touch_action(event: &Event) -> Result<Option<&str>, String> {
    let tags = event
        .field("tags")?
        .as_array()
        .ok_or("touch tags must be an array")?;
    let mut action = None;
    for tag in tags {
        let group = tag
            .get("group")
            .and_then(Value::as_str)
            .ok_or("invalid touch tag group")?;
        let value = tag
            .get("value")
            .and_then(Value::as_str)
            .ok_or("invalid touch tag value")?;
        if group == "action" && action.replace(value).is_some() {
            return Err("touch has multiple action tags".into());
        }
    }
    Ok(action)
}

fn participant_features(
    context: &Context<'_>,
    prefix: &str,
    player: &Value,
    other: &Value,
    bump: &Event,
    samples: &SampleFrames,
    values: &mut Values,
) -> Result<(), String> {
    let time = bump.number("time")?;
    let frame = bump.frame("frame")?;
    let physical_samples = match prefix {
        "p" => &samples.p,
        "q" => &samples.q,
        _ => return Err("unsupported participant feature prefix".into()),
    };
    physical_features(context, prefix, player, other, physical_samples, values)?;
    let next_touch = context.next(
        "touch",
        "time",
        time,
        samples.play_end,
        Some(("player", player)),
    )?;
    values.float(
        &format!("{prefix}_next_touch_delay_s"),
        next_touch
            .map(|event| event.number("time").map(|next| next - time))
            .transpose()?,
    )?;
    let action = match next_touch.map(touch_action).transpose()? {
        None => "no_touch",
        Some(None) => "unclassified",
        Some(Some(action)) => action,
    };
    values.category(
        &format!("{prefix}_next_touch_action"),
        Some(action),
        &["unclassified", "shot", "save", "pass", "clear", "boom"],
        "no_touch",
    )?;
    let cutoff = match next_touch {
        Some(event) => event.number("time")?,
        None => samples.play_end,
    };
    let mut whiff = false;
    for event in context.events("whiff") {
        if event.participant("player")? == player
            && event.text("kind")? == "whiff"
            && event.number("time")? > time
            && event.number("resolved_time")? < cutoff
        {
            whiff = true;
        }
    }
    let mut beaten = false;
    for event in context.events("beaten_to_ball") {
        if event.participant("player")? == player
            && event.number("time")? > time
            && event.number("time")? < cutoff
        {
            beaten = true;
        }
    }
    values.boolean(&format!("{prefix}_whiff_before_next_touch"), Some(whiff));
    values.boolean(&format!("{prefix}_beaten_before_next_touch"), Some(beaten));
    for (name, field) in [
        ("advance", "advance_distance"),
        ("retreat", "retreat_distance"),
    ] {
        let distance = match next_touch {
            Some(event) => match event.payload.payload.get("ball_movement") {
                None | Some(Value::Null) => None,
                Some(movement) => Some(
                    movement
                        .get(field)
                        .and_then(Value::as_f64)
                        .ok_or("invalid touch ball-movement measurement")?,
                ),
            },
            None => None,
        };
        values.float(&format!("{prefix}_next_touch_ball_{name}"), distance)?;
    }
    for (stream, feature, participant) in [
        ("ball_carry", "ground_carry_before", "player_id"),
        ("air_dribble", "air_dribble_before", "player_id"),
        ("pass", "receiving_pass_before", "receiver"),
        ("shadow_defense", "shadowing_before", "player"),
    ] {
        let mut active = false;
        for event in context.events(stream) {
            if event.participant(participant)? != player {
                continue;
            }
            let (start, end) = match stream {
                "pass" => (event.number("start_time")?, event.number("time")?),
                "shadow_defense" => (event.number("time")?, event.number("end_time")?),
                _ => (event.number("start_time")?, event.number("end_time")?),
            };
            if start < time && end >= time {
                active = true;
            }
        }
        values.boolean(&format!("{prefix}_{feature}"), Some(active));
    }
    let player_data = context.player(player)?;
    let previous_frame = frame.checked_sub(1);
    let dodging = previous_frame.and_then(|frame| match &player_data.frames[frame] {
        PlayerFrame::Data { dodge_active, .. } => Some(*dodge_active),
        PlayerFrame::Empty => None,
    });
    values.boolean(&format!("{prefix}_dodging_before"), dodging);
    values.category(
        &format!("{prefix}_field_half_before"),
        context.state("field_half", Some(player), time)?,
        &["offensive"],
        "defensive",
    )?;
    values.category(
        &format!("{prefix}_ball_depth_before"),
        context.state("ball_depth", Some(player), time)?,
        &["level_with_ball", "ahead_of_ball"],
        "behind_ball",
    )?;
    values.category(
        &format!("{prefix}_depth_role_before"),
        context.state("depth_role", Some(player), time)?,
        &["most_back", "most_forward", "mid", "no_teammates"],
        "other",
    )?;
    let mut height = None;
    if let Some(previous_frame) = previous_frame {
        for event in context.events("movement") {
            if event.participant("player")? == player
                && event.frame("frame")? <= previous_frame
                && event.frame("end_frame")? >= previous_frame
            {
                if height.is_some() {
                    return Err("overlapping player movement spans".into());
                }
                height = Some(event.text("height_band")?);
            }
        }
    }
    values.category(
        &format!("{prefix}_height_band_before"),
        height,
        &["low_air", "high_air"],
        "ground",
    )?;
    let track = context
        .analysis
        .outputs
        .events
        .accumulation_tracks
        .iter()
        .find(|track| &track.player_id == player && track.quantity == "boost_amount")
        .ok_or_else(|| format!("missing boost-amount track for {player}"))?;
    let boost = previous_frame
        .and_then(|frame| track.points.iter().rev().find(|point| point.frame <= frame))
        .map(|point| point.value);
    if boost.is_some_and(|boost| !(0.0..=BOOST_RAW_MAX).contains(&boost)) {
        return Err("boost level is outside the raw replay scale".into());
    }
    values.float(
        &format!("{prefix}_boost_before_pct"),
        boost.map(|boost| boost * 100.0 / BOOST_RAW_MAX),
    )?;
    let pickup = context.next(
        "boost_pickups",
        "time",
        time,
        samples.play_end,
        Some(("player_id", player)),
    )?;
    values.float(
        &format!("{prefix}_next_boost_pickup_delay_s"),
        pickup
            .map(|event| event.number("time").map(|next| next - time))
            .transpose()?,
    )?;
    values.float(
        &format!("{prefix}_next_boost_pickup_gain_pct"),
        pickup
            .map(|event| {
                event
                    .number("collected_amount")
                    .map(|gain| gain * 100.0 / BOOST_RAW_MAX)
            })
            .transpose()?,
    )?;
    pad_distance(context, prefix, player, pickup, physical_samples, values)?;
    for (stream, feature) in [
        ("bump", "previous_bump_age_s"),
        ("demolition", "previous_demolition_age_s"),
    ] {
        let mut previous: Option<f64> = None;
        for event in context.events(stream) {
            let involved = event.participant("victim")? == player
                || (stream == "bump" && event.participant("initiator")? == player);
            let event_time = event.number("time")?;
            if involved
                && event_time >= samples.play_start
                && event_time < time
                && previous.is_none_or(|previous| event_time > previous)
            {
                previous = Some(event_time);
            }
        }
        values.float(
            &format!("{prefix}_{feature}"),
            previous.map(|previous| time - previous),
        )?;
    }
    Ok(())
}

fn pad_distance(
    context: &Context<'_>,
    prefix: &str,
    player: &Value,
    pickup: Option<&Event>,
    samples: &ParticipantSamples,
    values: &mut Values,
) -> Result<(), String> {
    let replay = &context.analysis.outputs.replay_data;
    let mut matches = Vec::new();
    if let Some(pickup) = pickup {
        for event in &replay.boost_pad_events {
            if event.player.as_ref() == Some(player)
                && event.frame == pickup.frame("frame")?
                && event.time == pickup.number("time")?
                && matches!(event.kind, PadEventKind::PickedUp { .. })
            {
                matches.push(event);
            }
        }
    }
    let distance = if matches.len() == 1 {
        let id = &matches[0].pad_id;
        let pads: Vec<_> = replay
            .boost_pads
            .iter()
            .filter(|pad| pad.pad_id.as_ref() == Some(id))
            .collect();
        if pads.len() > 1 {
            return Err(format!("duplicate resolved boost pad {id}"));
        }
        let player = context.player(player)?;
        pads.first()
            .zip(body(player, samples.before_frame).zip(body(player, samples.after_frame)))
            .map(|(pad, (before, after))| {
                after.location.horizontal_distance(pad.position)
                    - before.location.horizontal_distance(pad.position)
            })
    } else {
        None
    };
    values.float(&format!("{prefix}_next_pad_distance_delta"), distance)
}
