//! JSON Schemas generated from the Rust types.
//!
//! The checked-in files under `schemas/` must match what this module produces;
//! a test enforces it. Regenerate with:
//! `UPDATE_SCHEMAS=1 cargo test -p tempera-core schema`

use crate::events::Event;
use crate::spec::JobSpec;

/// JSON Schema for `tempera.yaml`.
pub fn job_spec_schema() -> String {
    pretty(&schemars::schema_for!(JobSpec))
}

/// JSON Schema for one backend event line.
pub fn event_schema() -> String {
    pretty(&schemars::schema_for!(Event))
}

fn pretty(schema: &schemars::Schema) -> String {
    let mut s = serde_json::to_string_pretty(schema).unwrap_or_default();
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use std::path::PathBuf;

    fn schemas_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../schemas")
    }

    fn check(file: &str, generated: String) {
        let path = schemas_dir().join(file);
        if std::env::var_os("UPDATE_SCHEMAS").is_some() {
            std::fs::create_dir_all(schemas_dir()).unwrap();
            std::fs::write(&path, &generated).unwrap();
            return;
        }
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            on_disk == generated,
            "{} is out of date; run `UPDATE_SCHEMAS=1 cargo test -p tempera-core schema`",
            path.display()
        );
    }

    #[test]
    fn schema_files_are_up_to_date() {
        check("tempera.schema.json", job_spec_schema());
        check("event.schema.json", event_schema());
    }

    #[test]
    fn examples_are_valid() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples");
        let mut n = 0;
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "yaml") {
                let text = std::fs::read_to_string(&path).unwrap();
                JobSpec::from_yaml(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                n += 1;
            }
        }
        assert!(n > 0, "no examples found");
    }
}
