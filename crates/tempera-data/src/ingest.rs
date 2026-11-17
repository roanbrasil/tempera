//! Ingestion: any supported file → normalized Parquet addressed by content hash.
//!
//! Layout (see docs/wiki/Contracts.md §4):
//! ```text
//! <store>/datasets/<sha256>/data.parquet
//! <store>/datasets/<sha256>/dataset.json
//! ```
//! Ingesting the same bytes twice is a no-op. CSV is streamed (sink to Parquet),
//! so files larger than RAM work.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{DataError, Result};

/// Bump when the ingestion output changes, so old caches are not reused.
pub const INGEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    Csv,
    Parquet,
    Jsonl,
}

/// What we learned while reading a CSV.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsvDialect {
    pub separator: char,
    /// `utf-8` or the encoding that was transcoded to UTF-8 (e.g. `windows-1252`).
    pub encoding: String,
    pub had_bom: bool,
}

/// Metadata stored next to `data.parquet`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    /// SHA-256 of the source bytes (plus ingest version).
    pub id: String,
    pub source: PathBuf,
    pub format: SourceFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub csv: Option<CsvDialect>,
    pub rows: u64,
    pub columns: Vec<ColumnInfo>,
    pub ingest_version: u32,
    #[serde(skip)]
    pub dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub dtype: String,
}

impl Dataset {
    pub fn data_path(&self) -> PathBuf {
        self.dir.join("data.parquet")
    }

    /// Lazily scans the normalized data.
    pub fn scan(&self) -> Result<LazyFrame> {
        scan_parquet(&self.data_path())
    }

    /// Loads a dataset previously ingested into `dir`.
    pub fn load(dir: &Path) -> Result<Self> {
        let text = fs::read_to_string(dir.join("dataset.json"))?;
        let mut ds: Dataset = serde_json::from_str(&text)?;
        ds.dir = dir.to_path_buf();
        Ok(ds)
    }
}

pub fn scan_parquet(path: &Path) -> Result<LazyFrame> {
    Ok(LazyFrame::scan_parquet(
        PlRefPath::try_from_path(path)?,
        ScanArgsParquet::default(),
    )?)
}

/// Ingests `source` into `<store>/datasets/<hash>/`. Returns the existing
/// dataset if these exact bytes were ingested before.
pub fn ingest(source: &Path, store: &Path) -> Result<Dataset> {
    if !source.is_file() {
        return Err(DataError::NotFound(source.to_path_buf()));
    }
    let format = detect_format(source)?;
    let id = hash_file(source)?;
    let dir = store.join("datasets").join(&id);
    if dir.join("dataset.json").is_file() && dir.join("data.parquet").is_file() {
        tracing::debug!(%id, "dataset already ingested");
        return Dataset::load(&dir);
    }
    fs::create_dir_all(&dir)?;
    let tmp_out = dir.join("data.parquet.tmp");

    let csv = match format {
        SourceFormat::Csv => {
            let (dialect, utf8_path) = prepare_csv(source, &dir)?;
            let lf = LazyCsvReader::new(PlRefPath::try_from_path(&utf8_path)?)
                .with_has_header(true)
                .with_separator(dialect.separator as u8)
                .with_infer_schema_length(Some(10_000))
                .with_try_parse_dates(true)
                .with_truncate_ragged_lines(true)
                .finish()?;
            sink_parquet(lf, &tmp_out)?;
            if utf8_path != source {
                let _ = fs::remove_file(&utf8_path);
            }
            Some(dialect)
        }
        SourceFormat::Parquet => {
            sink_parquet(scan_parquet(source)?, &tmp_out)?;
            None
        }
        SourceFormat::Jsonl => {
            let lf = LazyJsonLineReader::new(PlRefPath::try_from_path(source)?)
                .with_infer_schema_length(std::num::NonZeroUsize::new(10_000))
                .finish()?;
            sink_parquet(lf, &tmp_out)?;
            None
        }
    };
    fs::rename(&tmp_out, dir.join("data.parquet"))?;

    let lf = scan_parquet(&dir.join("data.parquet"))?;
    let schema = lf.clone().collect_schema()?;
    let rows = lf
        .select([len().alias("n")])
        .collect()?
        .column("n")?
        .u32()?
        .get(0)
        .unwrap_or(0) as u64;
    let columns = schema
        .iter()
        .map(|(name, dtype)| ColumnInfo {
            name: name.to_string(),
            dtype: dtype.to_string(),
        })
        .collect();
    let ds = Dataset {
        id,
        source: source.to_path_buf(),
        format,
        csv,
        rows,
        columns,
        ingest_version: INGEST_VERSION,
        dir: dir.clone(),
    };
    let tmp_meta = dir.join("dataset.json.tmp");
    fs::write(&tmp_meta, serde_json::to_string_pretty(&ds)?)?;
    fs::rename(tmp_meta, dir.join("dataset.json"))?;
    Ok(ds)
}

/// Streams a lazy frame into a Parquet file.
pub fn sink_parquet(lf: LazyFrame, out: &Path) -> Result<()> {
    lf.sink(
        SinkDestination::File {
            target: SinkTarget::Path(PlRefPath::try_from_path(out)?),
        },
        FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions::default())),
        UnifiedSinkArgs::default(),
    )?
    .collect_with_engine(Engine::Streaming)?;
    Ok(())
}

fn detect_format(path: &Path) -> Result<SourceFormat> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "csv" | "tsv" | "txt" => Ok(SourceFormat::Csv),
        "parquet" | "pq" => Ok(SourceFormat::Parquet),
        "jsonl" | "ndjson" => Ok(SourceFormat::Jsonl),
        _ => Err(DataError::UnsupportedFormat(path.to_path_buf())),
    }
}

/// SHA-256 of the file bytes, salted with the ingest version.
pub fn hash_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(format!("tempera-ingest-v{INGEST_VERSION}\0").as_bytes());
    let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

const SNIFF_BYTES: usize = 64 * 1024;

/// Detects BOM, encoding and separator. Returns a path to a UTF-8 file
/// without BOM: the source itself when it is already clean, otherwise a
/// transcoded copy inside `work_dir`.
fn prepare_csv(source: &Path, work_dir: &Path) -> Result<(CsvDialect, PathBuf)> {
    let mut head = vec![0u8; SNIFF_BYTES];
    let n = File::open(source)?.read(&mut head)?;
    head.truncate(n);

    let had_bom = head.starts_with(&[0xEF, 0xBB, 0xBF]);
    let body = if had_bom { &head[3..] } else { &head[..] };
    // A sniff window may cut a multi-byte char at the end; only judge complete chars.
    let utf8 = match std::str::from_utf8(body) {
        Ok(_) => true,
        Err(e) => e.error_len().is_none() && is_utf8_file(source)?,
    };
    let encoding = if utf8 {
        encoding_rs::UTF_8
    } else {
        encoding_rs::WINDOWS_1252
    };
    let (text, _, _) = encoding.decode(body);
    let separator = sniff_separator(&text);
    let dialect = CsvDialect {
        separator,
        encoding: encoding.name().to_ascii_lowercase(),
        had_bom,
    };

    if utf8 && !had_bom {
        return Ok((dialect, source.to_path_buf()));
    }
    let out = work_dir.join("source.utf8.csv");
    transcode(source, &out, encoding, had_bom)?;
    Ok((dialect, out))
}

fn is_utf8_file(path: &Path) -> Result<bool> {
    let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();
    let mut reader = BufReader::new(File::open(path)?);
    let mut buf = vec![0u8; 1 << 20];
    let mut out = String::with_capacity(1 << 21);
    loop {
        let n = reader.read(&mut buf)?;
        let last = n == 0;
        out.clear();
        let (_, _, had_errors) = decoder.decode_to_string(&buf[..n], &mut out, last);
        if had_errors {
            return Ok(false);
        }
        if last {
            return Ok(true);
        }
    }
}

fn transcode(src: &Path, dst: &Path, enc: &'static encoding_rs::Encoding, skip_bom: bool) -> Result<()> {
    let mut reader = BufReader::new(File::open(src)?);
    let mut writer = BufWriter::new(File::create(dst)?);
    let mut decoder = enc.new_decoder_without_bom_handling();
    let mut buf = vec![0u8; 1 << 20];
    let mut out = String::with_capacity(1 << 21);
    let mut first = true;
    loop {
        let n = reader.read(&mut buf)?;
        let last = n == 0;
        let mut chunk = &buf[..n];
        if first && skip_bom && chunk.starts_with(&[0xEF, 0xBB, 0xBF]) {
            chunk = &chunk[3..];
        }
        first = false;
        out.clear();
        let _ = decoder.decode_to_string(chunk, &mut out, last);
        writer.write_all(out.as_bytes())?;
        if last {
            break;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Picks the candidate separator whose per-line count is most consistent
/// (and non-zero) across the first lines, ignoring quoted sections.
fn sniff_separator(text: &str) -> char {
    const CANDIDATES: [char; 4] = [',', ';', '\t', '|'];
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).take(50).collect();
    let lines = if lines.len() > 1 {
        &lines[..lines.len() - 1]
    } else {
        &lines[..]
    }; // last may be cut
    let mut best = (',', 0usize, f64::MAX);
    for &c in &CANDIDATES {
        let counts: Vec<usize> = lines.iter().map(|l| count_unquoted(l, c)).collect();
        let Some(&first) = counts.first() else { continue };
        if first == 0 {
            continue;
        }
        let consistent = counts.iter().filter(|&&n| n == first).count();
        let score = (consistent, first);
        let spread = counts
            .iter()
            .map(|&n| (n as f64 - first as f64).abs())
            .sum::<f64>();
        if score.0 > best.1 || (score.0 == best.1 && spread < best.2) {
            best = (c, consistent, spread);
        }
    }
    best.0
}

fn count_unquoted(line: &str, sep: char) -> usize {
    let mut in_quotes = false;
    let mut n = 0;
    for ch in line.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == sep && !in_quotes {
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn sniffs_separators() {
        assert_eq!(sniff_separator("a,b,c\n1,2,3\n4,5,6\n"), ',');
        assert_eq!(sniff_separator("a;b;c\n1,5;2;3\n4;5;6\n"), ';');
        assert_eq!(sniff_separator("a\tb\n1\t2\n3\t4\n"), '\t');
        assert_eq!(sniff_separator("\"x,y\"|b\n\"1,2\"|3\n\"4\"|5\n"), '|');
    }

    #[test]
    fn ingests_csv_and_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(
            tmp.path(),
            "d.csv",
            b"id,amount,when,label\n1,10.5,2024-01-02,yes\n2,,2024-02-03,no\n",
        );
        let ds = ingest(&src, tmp.path()).unwrap();
        assert_eq!(ds.rows, 2);
        let dtypes: Vec<_> = ds.columns.iter().map(|c| c.dtype.as_str()).collect();
        assert_eq!(dtypes, ["i64", "f64", "date", "str"]);
        let modified = fs::metadata(ds.data_path()).unwrap().modified().unwrap();
        let again = ingest(&src, tmp.path()).unwrap();
        assert_eq!(again.id, ds.id);
        assert_eq!(
            fs::metadata(again.data_path()).unwrap().modified().unwrap(),
            modified
        );
    }

    #[test]
    fn handles_bom_semicolon_quotes_and_latin1() {
        let tmp = tempfile::tempdir().unwrap();
        // BOM + semicolons + quoted field with separator and newline.
        let src = write(
            tmp.path(),
            "bom.csv",
            b"\xEF\xBB\xBFname;city\n\"Silva; Ana\";S\xC3\xA3o Paulo\n\"multi\nline\";Rio\n",
        );
        let ds = ingest(&src, tmp.path()).unwrap();
        let csv = ds.csv.clone().unwrap();
        assert!(csv.had_bom);
        assert_eq!(csv.separator, ';');
        let df = ds.scan().unwrap().collect().unwrap();
        assert_eq!(df.get_column_names()[0].as_str(), "name");
        assert_eq!(
            df.column("city").unwrap().str().unwrap().get(0),
            Some("São Paulo")
        );

        // Latin-1 encoded "São" (0xE3).
        let src = write(tmp.path(), "latin.csv", b"city,n\nS\xE3o Paulo,1\nRio,2\n");
        let ds = ingest(&src, tmp.path()).unwrap();
        assert_eq!(ds.csv.as_ref().unwrap().encoding, "windows-1252");
        let df = ds.scan().unwrap().collect().unwrap();
        assert_eq!(
            df.column("city").unwrap().str().unwrap().get(0),
            Some("São Paulo")
        );
    }

    #[test]
    fn ingests_jsonl_and_parquet() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(
            tmp.path(),
            "d.jsonl",
            b"{\"a\":1,\"b\":\"x\"}\n{\"a\":2,\"b\":\"y\"}\n",
        );
        let ds = ingest(&src, tmp.path()).unwrap();
        assert_eq!(ds.rows, 2);
        let pq = tmp.path().join("copy.parquet");
        fs::copy(ds.data_path(), &pq).unwrap();
        let again = ingest(&pq, tmp.path()).unwrap();
        assert_eq!(again.rows, 2);
        assert_eq!(again.format, SourceFormat::Parquet);
    }

    #[test]
    fn rejects_unknown_format() {
        let tmp = tempfile::tempdir().unwrap();
        let src = write(tmp.path(), "d.xlsx", b"nope");
        assert!(matches!(
            ingest(&src, tmp.path()),
            Err(DataError::UnsupportedFormat(_))
        ));
    }
}
