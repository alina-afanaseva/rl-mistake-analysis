//! Terminal touch selection from native possession events.

use crate::{Candidate, ReplayFacts, annotations::IncidentAnchor};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;

pub const SCHEMA: &str = "bang_with_time.candidates.v1";

#[derive(Deserialize)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
enum NativeEvent {
    Touch(Touch),
    FiftyFifty(Fifty),
    PlayerPossession(Episode),
    Possession(Possession),
}

#[derive(Deserialize)]
struct Tag {
    group: String,
    value: String,
}

#[derive(Deserialize)]
struct Touch {
    touch_id: u64,
    frame: u64,
    time: f64,
    player: Value,
    is_team_0: bool,
    tags: Vec<Tag>,
}

#[derive(Deserialize)]
struct Fifty {
    start_frame: u64,
    resolve_frame: u64,
    team_zero_touch_frame: Option<u64>,
    team_one_touch_frame: Option<u64>,
}

#[derive(Deserialize)]
struct Episode {
    player_id: Value,
    start_frame: u64,
    end_frame: u64,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum PossessionState {
    TeamZero,
    TeamOne,
    Neutral,
}

#[derive(Deserialize)]
struct Possession {
    frame: u64,
    end_frame: u64,
    active: bool,
    possession_state: PossessionState,
}

fn validate_player(player: &Value) -> Result<(), String> {
    if player.as_object().is_none_or(|id| id.is_empty()) {
        return Err("touches and possession episodes require player identities".into());
    }
    Ok(())
}

pub(crate) fn anchor(candidate: &Candidate) -> Result<IncidentAnchor, String> {
    let touch: Touch = serde_json::from_value(candidate.evidence.clone())
        .map_err(|error| format!("invalid native touch evidence: {error}"))?;
    validate_player(&touch.player)?;
    if !touch.time.is_finite()
        || candidate.frame != touch.frame
        || candidate.raw_time != touch.time
        || candidate.player_id != touch.player
    {
        return Err("touch anchor does not agree with its event evidence".into());
    }
    Ok(IncidentAnchor::BangWithTime {
        frame: touch.frame,
        touch_id: touch.touch_id,
    })
}

pub(crate) fn select(replay: &ReplayFacts) -> Result<Vec<Candidate>, String> {
    let mut touches = Vec::new();
    let mut touch_ids = HashSet::new();
    let mut fifty_frames = HashSet::new();
    let mut episodes = Vec::<Episode>::new();
    let mut possession = Vec::<Possession>::new();
    let mut fifties = Vec::<Fifty>::new();
    for event in &replay.events {
        let stream = event.meta.stream.as_str();
        if !matches!(
            stream,
            "touch" | "fifty_fifty" | "player_possession" | "possession"
        ) {
            continue;
        }
        if event.payload["kind"] != stream {
            return Err(format!("event payload disagrees with stream: {stream}"));
        }
        let native = serde_json::from_value(event.payload.clone())
            .map_err(|error| format!("invalid native {stream}: {error}"))?;
        match native {
            NativeEvent::Touch(touch) => {
                validate_player(&touch.player)?;
                if !touch.time.is_finite() || !touch_ids.insert(touch.touch_id) {
                    return Err("invalid touch time or duplicate touch ID".into());
                }
                touches.push((touch, event));
            }
            NativeEvent::FiftyFifty(fifty) => {
                if fifty.resolve_frame < fifty.start_frame {
                    return Err("fifty has reversed boundaries".into());
                }
                fifty_frames.extend(
                    [fifty.team_zero_touch_frame, fifty.team_one_touch_frame]
                        .into_iter()
                        .flatten(),
                );
                fifties.push(fifty);
            }
            NativeEvent::PlayerPossession(episode) => {
                validate_player(&episode.player_id)?;
                if episode.end_frame < episode.start_frame {
                    return Err("player possession has reversed boundaries".into());
                }
                episodes.push(episode);
            }
            NativeEvent::Possession(span) => {
                if span.end_frame < span.frame {
                    return Err("resolved possession has reversed boundaries".into());
                }
                if span.frame < span.end_frame {
                    possession.push(span);
                }
            }
        }
    }
    possession.sort_by_key(|span| span.frame);
    if possession
        .windows(2)
        .any(|pair| pair[0].end_frame > pair[1].frame)
    {
        return Err("resolved possession intervals overlap".into());
    }
    touches.sort_by_key(|(touch, _)| (touch.frame, touch.touch_id));
    let ambiguous_frames: HashSet<_> = touches
        .windows(2)
        .filter(|pair| pair[0].0.frame == pair[1].0.frame && pair[0].0.player != pair[1].0.player)
        .map(|pair| pair[0].0.frame)
        .collect();
    let eligible = |touch: &Touch| {
        !fifty_frames.contains(&touch.frame)
            && !touch
                .tags
                .iter()
                .any(|tag| tag.group == "contested" && tag.value == "contested")
    };
    Ok(touches
        .iter()
        .enumerate()
        .filter(|(index, (touch, _))| {
            eligible(touch)
                && !touches.get(index + 1).is_some_and(|(next, _)| {
                    eligible(next)
                        && !ambiguous_frames.contains(&touch.frame)
                        && !ambiguous_frames.contains(&next.frame)
                        && same_possession(touch, next, &episodes, &possession, &fifties)
                })
        })
        .map(|(_, (touch, event))| Candidate {
            kind: "bang_with_time".into(),
            source_event_ids: vec![event.meta.id.clone()],
            frame: touch.frame,
            raw_time: touch.time,
            player_id: touch.player.clone(),
            features: Vec::new(),
            evidence: event.payload["payload"].clone(),
        })
        .collect())
}

fn same_possession(
    current: &Touch,
    next: &Touch,
    episodes: &[Episode],
    possession: &[Possession],
    fifties: &[Fifty],
) -> bool {
    if current.player != next.player
        || !episodes.iter().any(|episode| {
            episode.player_id == current.player
                && episode.start_frame <= current.frame
                && next.frame <= episode.end_frame
        })
        || fifties
            .iter()
            .any(|fifty| fifty.start_frame < next.frame && fifty.resolve_frame > current.frame)
    {
        return false;
    }
    let team = if current.is_team_0 {
        PossessionState::TeamZero
    } else {
        PossessionState::TeamOne
    };
    let first = possession.partition_point(|span| span.end_frame <= current.frame);
    let mut cursor = current.frame;
    for span in possession[first..]
        .iter()
        .take_while(|span| span.frame < next.frame)
    {
        if span.frame > cursor || !span.active || span.possession_state != team {
            break;
        }
        cursor = next.frame.min(span.end_frame);
    }
    cursor == next.frame
}
