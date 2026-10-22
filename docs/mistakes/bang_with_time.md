# Banging the ball away with time

Commands run from the repository root. Shared formats and training APIs are described in the [format guide](../formats.md) and [training guide](../training.md).

## Candidates

```sh
target/debug/rl-mistake-analysis candidates bang_with_time < analysis.json > candidates.json
target/debug/rl-mistake-analysis contract bang_with_time
```

Candidates come from SubtrActor's ball-touch events, ordered by frame and touch ID. Group consecutive touches by one player when they fall within one recorded period of that player's possession. Their team must have possession throughout, with play active. Keep the last touch in each group.

A touch by another player, several players touching in one frame, a contested touch, or a fifty between touches ends the group. A gap or change in possession also ends it. Remove touches tagged `contested=contested` and touches on the frames recorded as either team's first touch in a fifty.

For example, continuous possession across touches at frames 10, 20, and 30 keeps frame 30. A fifty at frame 20 leaves frames 10 and 30 as candidates.

Each candidate is identified by the replay checksum, frame, touch ID, and player: `(replay_sha256, frame, touch_id, player_id)`. The output uses schema `bang_with_time.candidates.v1` and keeps the original touch in `evidence`, including its `touch_id`, along with the source event ID, time, and supplier revision.

## Annotations

Each review supplies `kind: bang_with_time`, the exact touch key, `raw_time`, native `evidence`, binary `label`, and `engine_revision`. `source_event_ids` and `reject_reason` are optional. The judged player is the native toucher.

```sh
target/debug/rl-mistake-analysis validate-annotations reviews.json
python -m training.bang_dataset features.parquet reviews.json training.parquet
```

The join matches replay checksum, frame, touch ID, player ID, and replay timestamp. Supplied source event IDs must match the extracted event. It selects reviewed rows and adds the label, annotation revision, and rejection reason while preserving the feature contract and column order.

## Context extraction

`bang_with_time.context.v1` contains 107 typed measurements per selected touch. Extraction uses a combined `events,replay_data` analysis. `--player PLAYER_ID_JSON` selects one judged player; the full selection contains every candidate's feature row.

```sh
target/debug/rl-mistake-analysis bang-features examples/bang_sampling.json < analysis.json > features.json
target/debug/rl-mistake-analysis bang-feature-contract
python -m training.bang_features data/analyses features.parquet \
  --engine target/debug/rl-mistake-analysis \
  --sampling examples/bang_sampling.json
```

Each row carries its exact touch key, timestamp, source event, sample references, and typed measurements. `samples.closest_opponent_id` identifies the player used for the opponent profile. The Parquet builder stores the shared feature contract and extraction provenance described in the [format guide](../formats.md#feature-and-training-tables).

## Frame selection

Frame selection chooses which replay frames supply the measurements. The configuration is called `sampling` in files and commands. The [configuration file](../../examples/bang_sampling.json) records:

```json
{"method": "native_touch_stages"}
```

The candidate is the selected terminal touch. The previous stage is the preceding distinct contact frame; the first and second later stages are the next two distinct contact frames. A stage retains every simultaneous toucher. Stage lookup uses the original native touch timeline, including contested contacts.

Measurements stay within the observed live-play segment: from the kickoff's `live_action_start_time` or observed segment start through the next goal, kickoff start, or replay end. Goal-boundary touches are included. Delays use elapsed replay seconds from native frame timestamps.

Ball positions use contact frames. Incoming speed uses the preceding frame in the same segment. Outgoing speed uses the touch's native `sample_frame`; `sample_time` must match that frame's timestamp. Event contact timestamps must match their native frames.

Team axes place the arena center at the origin. `toward` points toward the opposing goal, `side` is the perpendicular ground axis, and `up` is world height. Both ground axes rotate with the judged player's team. Distances and speeds use full 3D native coordinates; headings use normalized car quaternions.

Surrounding-player measurements use the preceding frame and native tracked-activity spans. Approaching counts include tracked teammates other than the subject and tracked opponents. The opponent profile selects the smallest 3D car-to-ball distance, with lexicographically smallest serialized player ID resolving ties. Its position, heading, velocity, and boost refer to that selected player and frame.

Unavailable stages, bodies, states, or required velocities produce nullable measurements. Unknown opponent activity or an unavailable tracked opponent body makes its team's count and the closest-opponent profile unavailable. Known tracked teams have a count of the cars meeting the approach condition, including zero. Numeric observations must be finite and representable as float32.

## Feature calculations

`stage` means `first_touch` or `second_touch`; `axis` means `toward`, `side`, or `up`. The subject is the judged player. Columns retain the types and nullability declared by `bang-feature-contract`. Event flags describe the presence of the specified event within an observed interval; exchange flags require the second stage as their endpoint.

| Features | Type | Calculation | Source |
| --- | --- | --- | --- |
| `candidate_ball_position_axis` | nullable float32 | Native ball center at the candidate contact frame, expressed in team axes. | `replay_data.frame_data.ball_data` |
| `candidate_incoming_ball_speed` | nullable float32 | Length of the native ball velocity in the preceding frame. | `replay_data.frame_data.ball_data` |
| `candidate_outgoing_ball_speed` / `stage_outgoing_ball_speed` | nullable float32 | Length of the native ball velocity at the touch's classification sample frame. Simultaneous touches share that sample frame. | `events.touch.sample_frame`, `replay_data.frame_data.ball_data` |
| `closest_opponent_position_relative_ball_axis` | nullable float32 | Selected opponent's car center minus the ball center in the preceding frame, expressed in team axes. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `closest_opponent_heading_axis` | nullable float32 | Selected opponent's forward unit vector from its normalized native quaternion in the preceding frame, expressed in team axes. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `closest_opponent_velocity_relative_ball_axis` | nullable float32 | Selected opponent's velocity minus the ball velocity in the preceding frame, expressed in team axes. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `closest_opponent_boost_before_pct` / `subject_boost_before_pct` | nullable float32 | Latest native boost-track point at or before the preceding frame, multiplied by `100/255`. | `events.accumulation_tracks.boost_amount` |
| `approaching_teammate_count` / `approaching_opponent_count` | nullable uint32 | Count tracked cars with a strictly positive dot product between their own velocity and their car-to-ball vector. Teammates are the subject's other team members. | `events.replay_meta`, `events.player_activity`, `replay_data.frame_data` |
| `candidate_action_*` | bool | Native action tag: shot, save, clear, pass, boom, or unclassified. | `events.touch.tags.action` |
| `candidate_dodging` | bool | Native `dodge_state` equals `dodge`. | `events.touch.tags.dodge_state` |
| `subject_field_half_at_candidate_offensive` | nullable bool | Native `field_half` at candidate time: offensive is true and defensive is false. | `events.field_half` |
| `subject_depth_role_at_candidate_*` | nullable bool | Native `depth_role` at candidate time: most_back, most_forward, mid, or no_teammates. The other category sets all indicators false. | `events.depth_role` |
| `subject_height_band_at_candidate_*` | nullable bool | Native movement height band at the candidate frame: low_air and high_air have indicators; ground sets both false. | `events.movement` |
| `subject_ball_depth_at_candidate_*` / `subject_ball_depth_at_stage_*` | nullable bool | Subject's native `ball_depth` at the corresponding contact time: level_with_ball and ahead_of_ball have indicators; behind_ball sets both false. | `events.ball_depth` |
| `opponent_beaten_at_candidate` | bool | Native `beaten_to_ball` event at the candidate frame with the subject as winner and an opponent as the beaten player. | `events.beaten_to_ball` |
| `previous_touch_self/teammate/opponent` | nullable bool | Relationships represented among the preceding stage's touchers. Teammate means another member of the subject's team. | `events.touch` |
| `previous_touch_delay_s` | nullable float32 | Candidate time minus the preceding stage's native time. | `events.touch` |
| `stage_self/teammate/opponent` | nullable bool | Relationships represented among that stage's touchers. Simultaneous touches can set several flags. | `events.touch` |
| `stage_delay_s` | nullable float32 | Stage time minus candidate time. | `events.touch` |
| `stage_ball_position_axis` | nullable float32 | Native ball center at the stage's contact frame, expressed in team axes. | `replay_data.frame_data.ball_data` |
| `stage_incoming_ball_speed` | nullable float32 | Length of the native ball velocity in the frame immediately before the stage. | `replay_data.frame_data.ball_data` |
| `stage_action_*` | nullable bool | Every native action category represented among that stage's touchers, including unclassified. | `events.touch.tags.action` |
| `opponent_whiff_during_exchange` | nullable bool | Confirmed native opponent whiff whose closest-approach frame is after the candidate and through the second stage, with `resolved_time` at or before that stage. | `events.whiff` |
| `subject_beaten_during_exchange` | nullable bool | Native `beaten_to_ball` event after the candidate frame and through the second stage with the subject as the beaten player. | `events.beaten_to_ball` |
| `subject_demo_during_exchange` | nullable bool | Native demolition of the subject in the inclusive interval from the candidate frame through the second stage. | `events.demolition` |
| `first_touch_opponent_control` | nullable bool | An opponent touch at the first later stage has a native possession tag of control or advance. | `events.touch.tags.possession` |
| `possession_before_own_team/opponent/neutral` | nullable bool | Native possession span starting before and covering candidate time, expressed relative to the subject's team. | `events.possession` |
| `ground_carry_before` / `air_dribble_before` | bool | Corresponding native episode for the subject starts earlier in the current play and covers candidate time. | `events.ball_carry`, `events.air_dribble` |
| `receiving_pass_before` | bool | Native pass names the subject as receiver, starts before the candidate, and has a reception time at or after it. | `events.pass` |
| `next_possession_own_team/opponent/neutral/no_change` | nullable bool | Earliest changed possession span starting at or after candidate time and before play end. The no_change category records an unchanged observed interval. | `events.possession` |
| `possession_change_delay_s` | nullable float32 | Changed possession span's start time minus candidate time. | `events.possession` |
| `next_goal_own_team/opponent` | bool | Scoring team of the goal ending this play segment, expressed relative to the subject's team. | `events.goal_context` |
| `next_goal_delay_s` | nullable float32 | Goal time minus candidate time. | `events.goal_context` |
| `touches_before_next_goal` | nullable uint32 | Count distinct native touch IDs strictly between candidate and goal times. Simultaneous contacts retain their separate IDs. | `events.touch`, `events.goal_context` |
| `subject_next_touch_delay_s` | nullable float32 | Subject's first touch in a later contact frame in this play, measured from candidate time. | `events.touch` |
| `subject_displacement_to_stage_axis` | nullable float32 | Subject's stage position minus its candidate position, expressed in team axes. A continuous observed subject track within one life defines a valid displacement. | `replay_data.frame_data`, `events.demolition` |
| `subject_ball_distance_at_stage` | nullable float32 | Full 3D distance between the subject's car center and ball center at the stage's contact frame. | `replay_data.frame_data` |
| `subject_heading_at_stage_axis` | nullable float32 | Subject's forward unit vector from its normalized native quaternion at the stage's contact frame, expressed in team axes. | `replay_data.frame_data.players.rigid_body.rotation` |

## Training and scoring

### Feature selection

Models declare an ordered subset of the 107 known measurements in their manifest. The base schema is `bang_with_time.context.v1`; named selections use `bang_with_time.context.v1.NAME.v1`. Feature selection and numeric conversion use the [shared training API](../training.md#dataset-inputs).

### Scoring

```sh
target/debug/rl-mistake-analysis score MODEL_DIRECTORY bang_with_time VERSION < analysis.json
```

Rust callers receive matching candidate and feature batches from the extractor:

```rust
use rl_mistake_analysis_candidates::bang_features;
use rl_mistake_analysis_inference::Model;

let mut model = Model::load(model_directory, "bang_with_time", model_version)?;
let sampling = model.metadata().sampling.ok_or("missing context sampling contract")?;
let (candidates, features) = bang_features::extract(&analysis, sampling, None)?;
let predictions = model.score_context(candidates, features)?;
```

`analysis` is the combined export deserialized as `bang_features::ReplayAnalysis`. Extraction and scoring use the same selected touches and frame selection method, including player-specific selections.

### Model training

Prepare reviewed rows with `training.bang_dataset`, then load them with `training.dataset.load` and `arrays`. The [training guide](../training.md) covers grouped model comparison, experiment tracking, ONNX verification, and release metadata.
