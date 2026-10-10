# Teammate bump

Commands below run from the repository root.

## Compute context features

The `bump-features` command calculates the `bumping_teammate.context.v1` contract from a combined `events,replay_data` export:

```sh
target/debug/rl-mistake-analysis bump-features examples/bump_sampling.json < artifacts/match.json > artifacts/match.features.json
target/debug/rl-mistake-analysis bump-feature-contract
```

Every teammate bump produces two rows, one for each player being evaluated. Pass `--player PLAYER_ID_JSON` to select one player, including a player identified as the victim by SubtrActor. Each row contains 105 boolean, float32, and uint32 measurements, with nullable fields for unavailable observations. The sampling configuration is required and recorded in every output.

Physical measurements start at the native frame before the bump and end at that participant's first acceleration minimum following the first acceleration peak at or after the bump. Both participants use their own observed endpoint. There are no fixed time offsets, interpolation, or recovery thresholds. The output records the actual sample frames, replay timestamps, and completion status. See [the feature definitions and formulas](#teammate-bump-context-features).

Create one unlabeled Parquet table from existing combined analyses:

```sh
python -m training.bump_features data/replay_analysis artifacts/team_bump_features.parquet \
  --engine target/debug/rl-mistake-analysis \
  --sampling examples/bump_sampling.json \
  --replay-ids data/replay_ids.json
```

`--replay-ids` selects a JSON array of replay filename stems; omitting it selects every `.json` file in the supplied directory. The exporter validates all selected inputs before replacing the output. Each unique replay checksum contributes one set of rows. Identical feature batches under different filenames retain all replay IDs and source paths as aliases in the metadata; conflicting batches are errors. This table carries incident and subject identities but no labels. `prepare` retains the eight-feature native contract. `score` selects the extractor specified by the chosen model's input schema and sampling contract.

## Review and score both participants

Create eight-feature review candidates from the events output:

```sh
target/debug/rl-mistake-analysis candidates < artifacts/match.events.json > artifacts/match.candidates.json
target/debug/rl-mistake-analysis contract bumping_teammate
```

The candidate output contains one batch. Import that batch into the review application's `candidate_batch` format with the replay ID and playback clock mapping. Each collision has two candidates with the same source event IDs and different `player_id` values. Candidate identity includes the judged player. Physical initiator and victim roles remain in `evidence`; they do not determine the review verdict.

Each candidate contains the eight incident measurements listed below. Both players share these measurements; the judged `player_id` carries their different identities. Candidate extraction needs no replay frames or sampling configuration. The 105 subject-specific features are computed by `bump-features` for training and scoring.

The review application's bundle and migration exports include `annotations` for linked participant candidates, including the eight-feature candidates. Each annotation identifies the judged player explicitly, so both participants can have different decisions. Unlinked moments remain in the review export and are excluded from the training annotations. Prepare the context training table with `training.bump_dataset`, then load it with `training.dataset.load` and `training.dataset.arrays` or the existing context reader.

`training.export.model_entry` records the context feature names and sampling method in the model catalog. Score a compatible context model with:

```sh
target/debug/rl-mistake-analysis score MODEL_DIRECTORY bumping_teammate VERSION < artifacts/match.json
```

Production computes the same subject-specific measurements as `bump-features` and scores both rows independently. Nullable measurements become `NaN` at the model boundary, exactly as in training. A context model requires combined `events,replay_data`; missing inputs or incompatible contracts are errors. Existing eight-feature models and their training preparation retain their initiator-only subject convention. Training a model to judge either participant requires player-specific reviews and context features.

## Native candidate and model features

`bumping_teammate` identifies an incident by `frame`, `initiator`, and `victim` within a recording. Both participants are review subjects; opponent contacts are excluded. The candidate library calculates the following eight incident measurements, in the order defined by `bumping_teammate.native.v1`. Existing eight-feature training and scoring use these measurements to judge the initiator:

1. `contact_confidence`
2. `closing_speed`
3. `victim_impulse`
4. `contact_distance`
5. `strength`
6. `initiator_height`
7. `victim_height`
8. `victim_distance_to_own_goal`

The first five copy measurements from the supplier's bump event (`confidence` becomes `contact_confidence`). The last three are calculated by this project from the supplier's player positions and team information: two heights and the victim's distance to their own goal. The complete vector is this project's feature contract, rather than the raw supplier event.

## Teammate-bump context features

`bump-features` computes `bumping_teammate.context.v1` from the combined export's `events` and `replay_data` outputs. Each teammate collision produces a separate row for each participant. `p` denotes the player being evaluated; `q` denotes their teammate. Either can be SubtrActor's initiator or victim. Selecting `--player PLAYER_ID_JSON` restricts rows to that subject.

The required sampling document is:

```json
{"method": "first_acceleration_peak_then_minimum"}
```

There are no numeric sampling parameters. For each participant, let `t` and `n` be the bump's replay timestamp and native frame. The before sample is frame `n−1`. Acceleration magnitude at frame `i` is `a_i = ||v_i − v_(i−1)|| / (t_i − t_(i−1))`, using the structured rigid body's native linear velocity. Starting at `n`, find the first frame satisfying `a_i >= a_(i−1)` and `a_i > a_(i+1)`. After that peak, the after sample is the first frame satisfying `a_i <= a_(i−1)` and `a_i < a_(i+1)`. These comparisons select the last point of an equal-valued extremum plateau.

Searches require consecutive valid velocities and positive elapsed time, and cannot cross a live-play boundary. Goals end play; a kickoff's first ball touch starts it. A missing frame or velocity interrupts the search. An incomplete search has no after sample and leaves dependent measurements null. The output records each participant's before, peak, and after frame and time, plus a completion status. The replay ending and a play boundary have separate statuses. This endpoint describes an observed acceleration pattern; it does not establish recovery or causation. It has no imposed maximum duration.

Time always means `replay_data.frame_data.metadata_frames[].time`, elapsed seconds since the replay began. Match countdown, playback speed, and computer wall-clock time do not enter these calculations. Source event joins use payload gameplay timestamps, rather than the event envelope's span-end timestamp.

Each row includes `source_event_id`, bump `frame` and `raw_time`, subject `player_id`, `teammate_id`, supplier `initiator` and `victim`, and `samples`. Batch metadata includes the replay checksum, supplier revision, sampling method, and ordered feature definitions. `bump-feature-contract` exports those definitions. The Parquet exporter stores the same contract under `rl_mistake_analysis_features`, together with per-replay row counts, `analysis_files`, and filename-stem `replay_ids`. Identical batches sharing a replay checksum contribute rows once and retain every filename as an alias. Conflicting batches for the same checksum are rejected. Player IDs and sample records are JSON strings in Parquet; measurements retain their declared types and nulls.

### Physical features

For each `r` in `{p, q}`, the horizontal `toward` unit vector points from `r` to the other participant at the before frame. `side` is its perpendicular `(-toward_y, toward_x)`; `up` is world vertical. The basis stays fixed for that participant's measurements. Coincident horizontal positions make only `toward` and `side` components undefined. Distances are in game units, velocities in game units per second, and all fields below are nullable float32.

| Fields, for each `r` | Formula | Source |
| --- | --- | --- |
| `r_velocity_before_{toward,side,up}` | Project native before velocity onto the fixed basis | `replay_data.frame_data.players[].frames[].Data.rigid_body.linear_velocity` |
| `r_velocity_delta_{toward,side,up}` | Project `velocity_after − velocity_before` | Same native velocity field |
| `r_position_delta_{toward,side,up}` | Project `position_after − position_before` | `rigid_body.location` |
| `r_ball_distance_before` | Horizontal distance between car and ball at the before sample | Player and ball `rigid_body.location` |
| `r_ball_distance_delta` | After horizontal car–ball distance minus before distance | Same player and ball positions |
| `r_heading_alignment_before` | Cosine between the horizontal car-forward vector and `toward`; undefined for a vertical heading | `rigid_body.rotation`, participant positions |
| `r_upright_after` | Dot product of car-up with world-up, in `[-1, 1]`; from quaternion `(x,y,z,w)`: `1 − 2(x²+y²)/(x²+y²+z²+w²)` | After `rigid_body.rotation` |

There is one uprightness measurement per participant, with no airborne duration, recovery timer, or extra height delta. Height change is the position vector's `up` component.

### Event features

All event lookups end at the current play segment's end. A next touch or pickup can occur at the bump's timestamp. Previous-event ages use events strictly before the bump in the same play segment. A missing next or previous event produces a null delay or age. Boolean absence indicators remain false; source states unavailable at the before boundary remain null.

| Fields | Type | Definition and source |
| --- | --- | --- |
| `p_is_initiator` | bool | Subject equals `events.bump.initiator` |
| `contact_gap` | float32 | Native `events.bump.contact_distance`, measured between car hitboxes |
| `first_touch_{p,q,other_teammate,opponent}` | bool | Participants in the earliest post-bump touch frame, from `events.touch`; simultaneous touches can set multiple fields; all false means no observed touch |
| `first_touch_delay_s` | nullable float32 | Earliest touch time minus bump time |
| `possession_before_{own_team,opponent}` | bool | `events.possession` state immediately before the bump; both false means neutral |
| `next_possession_{own_team,opponent,neutral}` | bool | First subsequent possession state different from the before state; all false means no change |
| `possession_change_delay_s` | nullable float32 | Start time of that changed possession state minus bump time |
| `next_goal_{own_team,opponent}` | bool | Team of the goal ending this play segment, from `events.goal_context`; both false means no observed goal |
| `next_goal_delay_s` | nullable float32 | Goal time minus bump time |
| `touches_before_next_goal` | nullable uint32 | Count distinct native `touch_id` values strictly between bump and goal; null when no goal is observed |
| `r_next_touch_delay_s` | nullable float32 | Participant's own next touch time minus bump time, for each `r` |
| `r_next_touch_action_{unclassified,shot,save,pass,clear,boom}` | bool | Own next touch's native `action` tag; unclassified means an observed touch has no action tag; all false means no next touch |
| `r_whiff_before_next_touch` | bool | Native `events.whiff` with kind `whiff`, closest approach strictly after the bump and resolution strictly before the participant's own next touch or play end |
| `r_beaten_before_next_touch` | bool | `events.beaten_to_ball` strictly after the bump and before the participant's own next touch or play end |
| `r_next_touch_ball_{advance,retreat}` | nullable float32 | Own next touch's native `ball_movement.advance_distance` and `retreat_distance`; null when not supplied |
| `r_ground_carry_before` | bool | Participant is in an `events.ball_carry` interval that began before the bump and ends at or after it |
| `r_air_dribble_before` | bool | Same containment test for `events.air_dribble` |
| `r_receiving_pass_before` | bool | Participant is `events.pass.receiver`, between pass `start_time` and reception `time` |
| `r_shadowing_before` | bool | Participant is in an `events.shadow_defense` interval |
| `r_dodging_before` | nullable bool | Native `PlayerFrame.Data.dodge_active` at frame `n−1` |
| `r_field_half_before_offensive` | nullable bool | Native `events.field_half` span immediately before the bump; false means defensive |
| `r_ball_depth_before_{level_with_ball,ahead_of_ball}` | nullable bool | Native `events.ball_depth`; both false means behind the ball |
| `r_depth_role_before_{most_back,most_forward,mid,no_teammates}` | nullable bool | Native `events.depth_role`; all false means the native `other` category |
| `r_height_band_before_{low_air,high_air}` | nullable bool | Native `events.movement.height_band` at frame `n−1`; both false means ground |
| `r_boost_before_pct` | nullable float32 | Latest native boost track point at or before `n−1`, multiplied by `100/255`; tracks store raw boost changes |
| `r_next_boost_pickup_delay_s` | nullable float32 | Participant's next `events.boost_pickups` time minus bump time |
| `r_next_boost_pickup_gain_pct` | nullable float32 | Native pickup `collected_amount × 100/255` |
| `r_next_pad_distance_delta` | nullable float32 | After minus before horizontal distance to the pad of that exact next pickup; join native pickup time, frame, and player to `replay_data.boost_pad_events`, then resolve its `pad_id` in `boost_pads`; null for an absent or ambiguous identity |
| `r_previous_bump_age_s` | nullable float32 | Bump time minus most recent earlier `events.bump` involving the participant in either role |
| `r_previous_demolition_age_s` | nullable float32 | Bump time minus most recent earlier `events.demolition` whose victim is the participant |

The contract contains 105 measurements after expanding participant, axis, and category names. `255` is the raw boost unit maximum; quaternion coefficients and vector operations are mathematical definitions. This extractor introduces no numeric heuristic thresholds. Native event and state definitions retain SubtrActor's own detector thresholds.

The feature table is unlabeled. Attaching reviews requires the exact replay, collision, and judged-player identities. Context candidates and production inference use the same participant extractor. Native preparation and native models retain `bumping_teammate.native.v1`.

## Prepare context features for training

`training.bump_dataset` joins reviewed incidents to an existing player-specific context feature table without extracting replay data again:

```sh
python -m training.bump_dataset artifacts/team_bump_features.parquet \
  data/subject_reviews.json artifacts/team_bump_training.parquet
```

The annotation document contains `schema_version: 1` and an `annotations` list. Each review supplies `kind: "bumping_teammate"`, `replay_sha256`, `frame`, `initiator`, `victim`, the judged `player_id`, `raw_time`, integer `label` (`0` or `1`), and `engine_revision`. Player identities use the supplier's JSON objects. Copy identity fields and the replay timestamp from the extracted row. A review can also include `reject_reason` and `source_event_ids`; supplied event IDs must agree with the extracted event.

The join uses the exact replay checksum, frame, initiator, victim, and judged player, and checks the replay timestamp. Both participants can have different labels for one collision. A review without an explicit `player_id`, a duplicate review, or an unmatched review is an error. Unreviewed rows are excluded from the labeled table; they are never assigned negative labels. The output retains the original feature metadata and adds `label`, `annotation_engine_revision`, and optional rejection reasons.

The Rust annotation validator also requires `player_id` and preserves separate participant reviews. Legacy native `prepare` accepts only initiator labels; victim labels must use the context dataset builder above.

Read the labeled context table with:

```python
from training.bump_dataset import arrays, load

table, contract = load("artifacts/team_bump_training.parquet", labeled=True)
X, y, replay_groups = arrays(table, contract)
```

Columns follow `feature_definitions` in the stored contract. Boolean flags become numerical `0` and `1`; numeric measurements retain their values; null measurements become `NaN`. Incident identities, `raw_time`, sample records, and provenance columns are excluded from model inputs. Timing features declared in the contract remain model inputs. Missing values are preserved for estimators that support them, including CatBoost. An estimator that requires imputation must include its imputer inside the estimator pipeline.

Use `replay_groups` to keep all subjects and collisions from a replay together in both the holdout partition and cross-validation. Save row identities, split indices, and the ordered feature contract alongside evaluation artifacts. A numeric CatBoost model can declare `training.onnx_export.catboost` as its exporter; the tracker verifies probabilities with CPU ONNX Runtime and saves the winning graph and hyperparameters. The boolean flags are numeric inputs, so the context model can use CatBoost's numeric ONNX export. A context model's release metadata records its ordered inputs and sampling method. Production inference validates both before scoring either participant.
