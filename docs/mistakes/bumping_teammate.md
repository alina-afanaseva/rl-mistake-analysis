# Teammate bump

Commands run from the repository root. Shared formats and training APIs are described in the [format guide](../formats.md) and [training guide](../training.md).

## Candidates

```sh
target/debug/rl-mistake-analysis candidates bumping_teammate < analysis.json > candidates.json
target/debug/rl-mistake-analysis contract bumping_teammate
```

Each teammate collision has two candidates with the same source event IDs and different `player_id` values, one for each player being evaluated. The original initiator and victim roles are stored in `evidence`.

SubtrActor can report the same collision several times. Reports are ordered by frame and source event ID. Reports involving the same two players within 10 frames of the previous report count as one collision, including reports with reversed initiator and victim roles. The first report supplies the event ID, frame, time, roles, and evidence.

Each candidate is identified by the replay checksum, collision frame, both players, and the player being evaluated: `(replay_sha256, frame, initiator, victim, player_id)`. The output uses schema `bumping_teammate.native.v1` and includes eight measurements shared by both players:

| Features | Type | Calculation | Source |
| --- | --- | --- | --- |
| `contact_confidence` | float64 | Native bump `confidence`. | `events.bump.confidence` |
| `closing_speed` | float64 | Native bump `closing_speed`. | `events.bump.closing_speed` |
| `victim_impulse` | float64 | Native bump `victim_impulse`. | `events.bump.victim_impulse` |
| `contact_distance` | float64 | Native bump `contact_distance`. | `events.bump.contact_distance` |
| `strength` | float64 | Native bump `strength`. | `events.bump.strength` |
| `initiator_height` | float64 | Initiator's native position height. | `events.bump.initiator_position` |
| `victim_height` | float64 | Victim's native position height. | `events.bump.victim_position` |
| `victim_distance_to_own_goal` | float64 | Distance from the victim's native position to their own goal. | `events.bump.victim_position`, `events.bump.victim_is_team_0` |

Models using these eight measurements judge the initiator. Context models use each player's own measurements, described below.

## Annotations

Each review supplies `kind: bumping_teammate`, the exact subject key, `raw_time`, native `evidence`, binary `label`, and `engine_revision`. `source_event_ids` and `reject_reason` are optional. Each participant can have an independent verdict for the same contact.

```sh
target/debug/rl-mistake-analysis validate-annotations reviews.json
python -m training.bump_dataset features.parquet reviews.json training.parquet
```

The join matches replay checksum, frame, initiator, victim, judged player, and replay timestamp. Supplied source event IDs must match the extracted event. It selects reviewed rows and adds the label, annotation revision, and rejection reason while preserving the feature contract and column order.

## Context extraction

`bumping_teammate.context.v1` contains 105 typed measurements per participant. Extraction uses a combined `events,replay_data` analysis. `--player PLAYER_ID_JSON` selects one judged player; the full selection contains both participants' feature rows.

```sh
target/debug/rl-mistake-analysis bump-features examples/bump_sampling.json < analysis.json > features.json
target/debug/rl-mistake-analysis bump-feature-contract
python -m training.bump_features data/analyses features.parquet \
  --engine target/debug/rl-mistake-analysis \
  --sampling examples/bump_sampling.json
```

Each row carries its exact subject key, timestamp, source event, sample references, and typed measurements. `p` denotes the judged player and `q` their teammate. The Parquet builder stores the shared feature contract and extraction provenance described in the [format guide](../formats.md#feature-and-training-tables).

## Frame selection

Frame selection chooses which replay frames supply the measurements. The configuration is called `sampling` in files and commands. The [configuration file](../../examples/bump_sampling.json) records:

```json
{"method": "first_acceleration_peak_then_minimum"}
```

For a contact at native frame `n`, the before sample is frame `n−1`. For each participant, acceleration magnitude at frame `i` is `a_i = ||v_i − v_(i−1)|| / (t_i − t_(i−1))`. Starting at `n`, the first peak satisfies `a_i >= a_(i−1)` and `a_i > a_(i+1)`. The following minimum satisfies `a_i <= a_(i−1)` and `a_i < a_(i+1)`. These comparisons select the final point of an extremum plateau. Each participant has their own after sample.

The extrema search uses consecutive observed native velocities and positive elapsed times within the live-play segment. A native kickoff's `live_action_start_time` starts the segment; a goal, next kickoff, or replay end bounds it. Missing observations interrupt the search. `samples` records each participant's before, peak, and after frames and times, together with the completion status. Unavailable endpoints produce nullable measurements.

Time is elapsed replay seconds from `metadata_frames[].time`. Event joins use native payload gameplay timestamps. The horizontal `toward` axis points from each participant to their teammate at the before frame; `side` is `(-toward_y, toward_x)` and `up` is world vertical. The basis stays fixed for that participant. Distances use native game units and velocities use game units per second.

## Feature calculations

`r` means `p` or `q`. Each participant's physical measurements use their own sample frames and fixed axes. Coincident horizontal positions produce nullable horizontal components. Height change is the position vector's `up` component.

| Features | Type | Calculation | Source |
| --- | --- | --- | --- |
| `r_velocity_before_{toward,side,up}` | nullable float32 | Project native before velocity onto the fixed basis. | `replay_data.frame_data.players[].frames[].Data.rigid_body.linear_velocity` |
| `r_velocity_delta_{toward,side,up}` | nullable float32 | Project `velocity_after − velocity_before`. | `rigid_body.linear_velocity` |
| `r_position_delta_{toward,side,up}` | nullable float32 | Project `position_after − position_before`. | `rigid_body.location` |
| `r_ball_distance_before` | nullable float32 | Horizontal distance between car and ball at the before sample. | Player and ball `rigid_body.location` |
| `r_ball_distance_delta` | nullable float32 | After horizontal car–ball distance minus before distance. | Player and ball `rigid_body.location` |
| `r_heading_alignment_before` | nullable float32 | Cosine between the horizontal car-forward vector and `toward`; undefined for a vertical heading. | `rigid_body.rotation`, participant positions |
| `r_upright_after` | nullable float32 | Dot product of car-up with world-up, in `[-1, 1]`; from quaternion `(x,y,z,w)`: `1 − 2(x²+y²)/(x²+y²+z²+w²)`. | After `rigid_body.rotation` |

| Features | Type | Calculation | Source |
| --- | --- | --- | --- |
| `p_is_initiator` | bool | Subject equals `events.bump.initiator`. | `events.bump` |
| `contact_gap` | float32 | Native `events.bump.contact_distance`, measured between car hitboxes. | `events.bump.contact_distance` |
| `first_touch_{p,q,other_teammate,opponent}` | bool | Participants in the earliest post-bump touch frame, from `events.touch`; simultaneous touches can set multiple fields. | `events.touch` |
| `first_touch_delay_s` | nullable float32 | Earliest touch time minus bump time. | `events.touch` |
| `possession_before_{own_team,opponent}` | bool | `events.possession` state immediately before the bump; both false means neutral. | `events.possession` |
| `next_possession_{own_team,opponent,neutral}` | bool | First subsequent possession state different from the before state; all false records an unchanged observed interval. | `events.possession` |
| `possession_change_delay_s` | nullable float32 | Start time of that changed possession state minus bump time. | `events.possession` |
| `next_goal_{own_team,opponent}` | bool | Team of the goal ending this play segment, from `events.goal_context`. | `events.goal_context` |
| `next_goal_delay_s` | nullable float32 | Goal time minus bump time. | `events.goal_context` |
| `touches_before_next_goal` | nullable uint32 | Count distinct native `touch_id` values strictly between bump and goal. | `events.touch`, `events.goal_context` |
| `r_next_touch_delay_s` | nullable float32 | Participant's own next touch time minus bump time, for each `r`. | `events.touch` |
| `r_next_touch_action_{unclassified,shot,save,pass,clear,boom}` | bool | Native action category of the participant's next touch, including unclassified. | `events.touch.tags.action` |
| `r_whiff_before_next_touch` | bool | Native `events.whiff` with kind `whiff`, closest approach strictly after the bump and resolution strictly before the participant's own next touch or play end. | `events.whiff`, `events.touch`, `events.goal_context` |
| `r_beaten_before_next_touch` | bool | `events.beaten_to_ball` strictly after the bump and before the participant's own next touch or play end. | `events.beaten_to_ball`, `events.touch`, `events.goal_context` |
| `r_next_touch_ball_{advance,retreat}` | nullable float32 | Own next touch's native `ball_movement.advance_distance` and `retreat_distance`. | `events.touch.ball_movement` |
| `r_ground_carry_before` | bool | Participant is in an `events.ball_carry` interval that began before the bump and ends at or after it. | `events.ball_carry` |
| `r_air_dribble_before` | bool | Same containment test for `events.air_dribble`. | `events.air_dribble` |
| `r_receiving_pass_before` | bool | Participant is `events.pass.receiver`, between pass `start_time` and reception `time`. | `events.pass` |
| `r_shadowing_before` | bool | Participant is in an `events.shadow_defense` interval. | `events.shadow_defense` |
| `r_dodging_before` | nullable bool | Native `PlayerFrame.Data.dodge_active` at frame `n−1`. | `replay_data.frame_data` |
| `r_field_half_before_offensive` | nullable bool | Native `events.field_half` span immediately before the bump; false means defensive. | `events.field_half` |
| `r_ball_depth_before_{level_with_ball,ahead_of_ball}` | nullable bool | Native `events.ball_depth`; both false means behind the ball. | `events.ball_depth` |
| `r_depth_role_before_{most_back,most_forward,mid,no_teammates}` | nullable bool | Native `events.depth_role`; all false means the native `other` category. | `events.depth_role` |
| `r_height_band_before_{low_air,high_air}` | nullable bool | Native `events.movement.height_band` at frame `n−1`; both false means ground. | `events.movement` |
| `r_boost_before_pct` | nullable float32 | Latest native boost track point at or before `n−1`, multiplied by `100/255`; tracks store raw boost changes. | `events.accumulation_tracks.boost_amount` |
| `r_next_boost_pickup_delay_s` | nullable float32 | Participant's next `events.boost_pickups` time minus bump time. | `events.boost_pickups` |
| `r_next_boost_pickup_gain_pct` | nullable float32 | Native pickup `collected_amount × 100/255`. | `events.boost_pickups` |
| `r_next_pad_distance_delta` | nullable float32 | After minus before horizontal distance to the pad of that exact next pickup; join native pickup time, frame, and player to `replay_data.boost_pad_events`, then resolve its `pad_id` in `boost_pads`; an exact pickup and resolved pad identify a valid measurement. | `events.boost_pickups`, `replay_data.boost_pad_events`, `replay_data.boost_pads` |
| `r_previous_bump_age_s` | nullable float32 | Bump time minus most recent earlier `events.bump` involving the participant in either role. | `events.bump` |
| `r_previous_demolition_age_s` | nullable float32 | Bump time minus most recent earlier `events.demolition` whose victim is the participant. | `events.demolition` |

Event lookups use the current live-play segment. Next touches and pickups can share the contact's timestamp; previous-event ages use strictly earlier events. Unavailable observations produce nullable measurements. Event flags describe observed events and categories. Native rotations provide heading and uprightness; native boost-track and pickup values use the conversion `100/255`.

## Training and scoring

### Feature selection

Models declare an ordered subset of the 105 known measurements in their manifest. The base schema is `bumping_teammate.context.v1`; named selections use `bumping_teammate.context.v1.NAME.v1`. Feature selection and numeric conversion use the [shared training API](../training.md#dataset-inputs).

The selected experimental release uses `bumping_teammate.context.v1.expanded90.v1`. Its [training recipe](../../training/configs/bumping_teammate_0.1.0.json) records the ordered inputs, XGBoost settings, frame selection method, and decision threshold.

### Scoring

```sh
target/debug/rl-mistake-analysis score MODEL_DIRECTORY bumping_teammate VERSION < analysis.json
```

Rust callers select the participant candidates and extract their context measurements:

```rust
use rl_mistake_analysis_candidates::{MistakeKind, bump_features};
use rl_mistake_analysis_inference::Model;

let mut model = Model::load(model_directory, "bumping_teammate", model_version)?;
let sampling = model.metadata().sampling.ok_or("missing context sampling contract")?;
let candidates = MistakeKind::BumpingTeammate.candidates(&replay_events)?;
let features = bump_features::extract(&analysis, sampling, None)?;
let predictions = model.score_context(candidates, features)?;
```

`analysis` is the combined export deserialized as `bump_features::ReplayAnalysis`; `replay_events` contains its event facts. The scorer evaluates each participant using the selected model's contract.

Incident-vector models use `MistakeKind::BumpingTeammate.initiator_candidates(&replay_events)` with `model.score(candidates)`.

### Model training

Prepare reviewed rows with `training.bump_dataset`, then load them with `training.dataset.load` and `arrays`. The [training guide](../training.md) covers grouped model comparison, experiment tracking, ONNX verification, and release metadata.

To fit the selected XGBoost recipe, supply a reviewed context table and a JSON array of held-out replay checksums:

```sh
python -m training.train_selected training.parquet held_out_replays.json model-release \
  --source-revision YOUR_CODE_REVISION
```

The command saves the ONNX graph, catalog, evaluation report, native XGBoost model, split indices, and predictions. The published [model card](https://huggingface.co/alina-afanaseva/rl-mistake-analysis-models) describes its evaluation results.

The incident-vector preparation command builds an eight-feature table from reviewed initiator annotations and original recordings:

```sh
target/debug/rl-mistake-analysis prepare reviews.json data/replays native-training.parquet bumping_teammate
```
