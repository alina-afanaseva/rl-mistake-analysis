use super::{ReplayContext, Samples, team_basis};
use crate::feature_context::{Context, possession_state};
use crate::feature_geometry::{directed_at_goal, goal_opening_distance};
use crate::feature_input::Event;
use crate::feature_values::Values;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub(super) struct TeamDistance {
    observed: bool,
    nearest: Option<f64>,
}

pub(super) fn team_distances(
    context: &Context<'_>,
    roster: &HashMap<String, bool>,
) -> Result<Vec<[TeamDistance; 2]>, String> {
    let frames = &context.analysis.outputs.replay_data.frame_data;
    let players = frames
        .players
        .iter()
        .map(|(id, data)| {
            Ok((
                usize::from(roster[&id.to_string()]),
                data,
                context.states_by_frame("player_activity", id)?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    (0..frames.metadata_frames.len())
        .map(|frame| {
            let Some(ball) = context.ball(Some(frame)) else {
                return Ok([TeamDistance {
                    observed: false,
                    nearest: None,
                }; 2]);
            };
            let mut distances = [TeamDistance {
                observed: true,
                nearest: None,
            }; 2];
            for (team, data, activity) in &players {
                let team = *team;
                match activity[frame] {
                    Some("demolished") => continue,
                    Some("tracked") => (),
                    None => {
                        distances[team].observed = false;
                        continue;
                    }
                    Some(state) => return Err(format!("invalid player activity state: {state}")),
                }
                let Some(car) = data.frames[frame].body() else {
                    distances[team].observed = false;
                    continue;
                };
                let distance = ball.location.subtract(car.location).length();
                if !distance.is_finite() {
                    return Err("nonfinite car-ball distance".into());
                }
                distances[team].nearest = Some(
                    distances[team]
                        .nearest
                        .map_or(distance, |old| old.min(distance)),
                );
            }
            Ok(distances)
        })
        .collect()
}

fn minimum(values: impl Iterator<Item = TeamDistance>) -> Option<f64> {
    let mut nearest: Option<f64> = None;
    for value in values {
        if !value.observed {
            return None;
        }
        if let Some(distance) = value.nearest {
            nearest = Some(nearest.map_or(distance, |old| old.min(distance)));
        }
    }
    nearest
}

fn mean_distance(
    replay: &ReplayContext<'_>,
    range: Option<(usize, usize)>,
    team: bool,
) -> Option<f64> {
    let (start, end) = range?;
    let frames = &replay
        .context
        .analysis
        .outputs
        .replay_data
        .frame_data
        .metadata_frames;
    let duration = frames[end].time - frames[start].time;
    if duration == 0.0 {
        return None;
    }
    let distance = |frame: usize| {
        let value = replay.team_distances[frame][usize::from(team)];
        value.nearest.filter(|_| value.observed)
    };
    let mut previous = distance(start)?;
    let mut integral = 0.0;
    for frame in start + 1..=end {
        let current = distance(frame)?;
        let elapsed = frames[frame].time - frames[frame - 1].time;
        integral += (previous + current) * elapsed / 2.0;
        previous = current;
    }
    Some(integral / duration)
}

pub(super) fn features(
    replay: &ReplayContext<'_>,
    touch: &Event,
    samples: &Samples,
    values: &mut Values,
) -> Result<(), String> {
    let context = &replay.context;
    let team = touch.boolean("is_team_0")?;
    let frame = touch.frame("frame")?;
    let first = samples.first_touch.as_ref().map(|stage| stage.frame);
    let second = samples.second_touch.as_ref().map(|stage| stage.frame);
    values.float(
        "play_elapsed_s",
        Some(touch.number("time")? - samples.play_start),
    )?;
    values.vector(
        "candidate_outgoing_ball_velocity",
        context
            .ball(Some(frame))
            .and_then(|ball| ball.linear_velocity),
        Some(team_basis(team)),
    )?;
    for (name, attacking_team) in [("own", !team), ("opponent", team)] {
        values.boolean(
            &format!("candidate_directed_at_{name}_goal"),
            context
                .ball(Some(frame))
                .map(|ball| directed_at_goal(ball, attacking_team))
                .transpose()?
                .flatten(),
        );
    }
    for (name, range) in [
        ("candidate_to_first_touch", first.map(|end| (frame, end))),
        ("at_first_touch", first.map(|frame| (frame, frame))),
        ("first_to_second_touch", first.zip(second)),
        ("at_second_touch", second.map(|frame| (frame, frame))),
    ] {
        for (relation, other_team) in [("own_team", team), ("opponent", !team)] {
            values.float(
                &format!("{relation}_ball_distance_{name}"),
                range.and_then(|(start, end)| {
                    minimum(
                        (start..=end)
                            .map(|frame| replay.team_distances[frame][usize::from(other_team)]),
                    )
                }),
            )?;
            if name == "candidate_to_first_touch" || name == "first_to_second_touch" {
                values.float(
                    &format!("{relation}_ball_distance_mean_{name}"),
                    mean_distance(replay, range, other_team),
                )?;
            }
        }
        for (goal, attacking_team) in [("own", !team), ("opponent", team)] {
            values.float(
                &format!("ball_{goal}_goal_distance_{name}"),
                goal_minimum(context, range, attacking_team)?,
            )?;
        }
    }
    let extrema = second.and_then(|end| {
        let position = context
            .ball(Some(frame))?
            .location
            .project(team_basis(team));
        let mut minimum = position;
        let mut maximum = position;
        for frame in frame + 1..=end {
            let position = context
                .ball(Some(frame))?
                .location
                .project(team_basis(team));
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(position[axis]);
                maximum[axis] = maximum[axis].max(position[axis]);
            }
        }
        Some((minimum, maximum))
    });
    for (axis, name) in ["toward", "side", "up"].iter().enumerate() {
        values.float(
            &format!("exchange_ball_position_min_{name}"),
            extrema.map(|(min, _)| min[axis]),
        )?;
        values.float(
            &format!("exchange_ball_position_max_{name}"),
            extrema.map(|(_, max)| max[axis]),
        )?;
    }
    event_features(context, touch, samples, values)?;
    let duration = samples
        .second_touch
        .as_ref()
        .map(|stage| possession_durations(context, touch.number("time")?, stage.raw_time, team))
        .transpose()?
        .flatten();
    for (index, state) in ["own_team", "opponent", "neutral"].iter().enumerate() {
        values.float(
            &format!("exchange_possession_{state}_s"),
            duration.map(|duration| duration[index]),
        )?;
    }
    Ok(())
}

fn goal_minimum(
    context: &Context<'_>,
    range: Option<(usize, usize)>,
    team: bool,
) -> Result<Option<f64>, String> {
    let Some((start, end)) = range else {
        return Ok(None);
    };
    let mut nearest: Option<f64> = None;
    for frame in start..=end {
        let Some(ball) = context.ball(Some(frame)) else {
            return Ok(None);
        };
        let distance = goal_opening_distance(ball.location, team)?;
        nearest = Some(nearest.map_or(distance, |old| old.min(distance)));
    }
    Ok(nearest)
}

fn event_features(
    context: &Context<'_>,
    touch: &Event,
    samples: &Samples,
    values: &mut Values,
) -> Result<(), String> {
    let subject = touch.participant("player")?;
    let team = touch.boolean("is_team_0")?;
    let frame = touch.frame("frame")?;
    let end = samples.second_touch.as_ref();
    let mut subject_bumped = false;
    let mut any_bump = false;
    let mut subject_whiff = false;
    let mut teammate_whiff = false;
    if let Some(end) = end {
        for event in context.events("bump") {
            if (frame..=end.frame).contains(&event.frame("frame")?) {
                any_bump = true;
                subject_bumped |= event.participant("victim")? == subject;
            }
        }
        for event in context.events("whiff") {
            let event_frame = event.frame("frame")?;
            if event_frame > frame
                && event_frame <= end.frame
                && event.text("kind")? == "whiff"
                && event.number("resolved_time")? <= end.raw_time
            {
                let is_subject = event.participant("player")? == subject;
                subject_whiff |= is_subject;
                teammate_whiff |= !is_subject && event.boolean("is_team_0")? == team;
            }
        }
    }
    for (name, found) in [
        ("subject_bumped_during_exchange", subject_bumped),
        ("any_bump_during_exchange", any_bump),
        ("subject_whiff_during_exchange", subject_whiff),
        ("teammate_whiff_during_exchange", teammate_whiff),
    ] {
        values.boolean(name, end.map(|_| found));
    }
    Ok(())
}

fn possession_durations(
    context: &Context<'_>,
    start: f64,
    end: f64,
    team: bool,
) -> Result<Option<[f64; 3]>, String> {
    let mut spans = context
        .events("possession")
        .iter()
        .map(|event| Ok((event.number("time")?, event.number("end_time")?, *event)))
        .collect::<Result<Vec<_>, String>>()?;
    spans.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut duration = [0.0; 3];
    let mut cursor = start;
    for (span_start, span_end, event) in spans {
        let left = start.max(span_start);
        let right = end.min(span_end);
        if left >= right {
            continue;
        }
        if left < cursor {
            return Err("overlapping possession spans during exchange".into());
        }
        if left > cursor {
            return Ok(None);
        }
        let index = match possession_state(event.text("possession_state")?, team)? {
            "own_team" => 0,
            "opponent" => 1,
            "neutral" => 2,
            _ => unreachable!(),
        };
        duration[index] += right - left;
        cursor = right;
    }
    Ok((cursor == end).then_some(duration))
}
