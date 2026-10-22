use crate::feature_input::Vector;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Boolean,
    F32,
    U32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureDefinition {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Measurement {
    Boolean(bool),
    Count(u32),
    Float(f32),
}

#[derive(Default)]
pub(crate) struct Values(pub(crate) BTreeMap<String, Option<Measurement>>);

impl Values {
    pub(crate) fn boolean(&mut self, name: &str, value: Option<bool>) {
        self.0.insert(name.into(), value.map(Measurement::Boolean));
    }

    pub(crate) fn float(&mut self, name: &str, value: Option<f64>) -> Result<(), String> {
        if value.is_some_and(|value| !value.is_finite() || !(value as f32).is_finite()) {
            return Err(format!("nonfinite or unrepresentable measurement: {name}"));
        }
        self.0.insert(
            name.into(),
            value.map(|value| Measurement::Float(value as f32)),
        );
        Ok(())
    }

    pub(crate) fn category(
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

    pub(crate) fn vector(
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

    pub(crate) fn validate(&self, definitions: &[FeatureDefinition]) -> Result<(), String> {
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
