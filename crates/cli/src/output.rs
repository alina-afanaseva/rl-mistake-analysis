use serde_json::{Value, json};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    Events,
    ReplayData,
    Stats,
    StatsTimeline,
    Features,
}

impl OutputKind {
    const ALL: [Self; 5] = [
        Self::Events,
        Self::ReplayData,
        Self::Stats,
        Self::StatsTimeline,
        Self::Features,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Events => "events",
            Self::ReplayData => "replay_data",
            Self::Stats => "stats",
            Self::StatsTimeline => "stats_timeline",
            Self::Features => "features",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Events => "Analysis event timeline with context and numeric tracks",
            Self::ReplayData => "Structured replay frames, metadata and basic replay events",
            Self::Stats => "Accumulated player and team statistics",
            Self::StatsTimeline => "Full statistics snapshots over time",
            Self::Features => "Numeric feature matrix with headers and replay metadata",
        }
    }
}

pub fn catalog() -> Value {
    json!({
        "outputs": OutputKind::ALL.map(|kind| json!({
            "name": kind.name(),
            "description": kind.description(),
            "implemented": true,
        })),
    })
}

pub fn selection(args: &[String]) -> Result<Vec<OutputKind>, String> {
    if args.is_empty() {
        return Ok(vec![OutputKind::Events, OutputKind::ReplayData]);
    }
    let [flag, names] = args else {
        return Err("expected --outputs followed by a comma-separated output selection".into());
    };
    if flag != "--outputs" {
        return Err(format!("unknown extraction option {flag}"));
    }
    let mut selected = Vec::new();
    for name in names.split(',').map(str::trim) {
        let kind = OutputKind::ALL
            .into_iter()
            .find(|kind| kind.name() == name)
            .ok_or_else(|| format!("unknown output {name:?}; use outputs to list output types"))?;
        if selected.contains(&kind) {
            return Err(format!("duplicate output {name:?}"));
        }
        selected.push(kind);
    }
    Ok(selected)
}
