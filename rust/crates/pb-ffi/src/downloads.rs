//! Optional models that are not inside the app: downloaded on demand from
//! the PixelBoost server (`GET /models/file/{name}`), resumable, verified
//! against the SHA-256 pinned here, then moved into place atomically.

use crate::server::ServerClient;
use crate::PbError;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;

pub struct Download {
    pub id: &'static str,
    pub file: &'static str,
    pub title: &'static str,
    pub detail: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// Pinned per app version (regenerate with `pixelboost convert` + sha256sum).
pub const DOWNLOADS: &[Download] = &[
    Download {
        id: "native2x",
        file: "native-x2.pbnn",
        title: "Native 2×",
        detail: "Exact 2× upscales of photos (Real-ESRGAN x2plus, int8). Without it 2× uses the 4× models.",
        bytes: 16_824_658,
        sha256: "9127907de5f50a4cb15dd42f4e3a79b0cd8a2cef5e33f1d8150c1fa9b83b0611",
    },
    Download {
        id: "cutout-hq",
        file: "cutout-isnet.pbnn",
        title: "HQ Cutout",
        detail: "IS-Net (Apache-2.0): much finer hair and holes in subject cutouts. Runs on this iPhone.",
        bytes: 44_107_566,
        sha256: "3793908bb60a83daec7ece8d79ca70cb90f3870083df0cc625e596af1761d890",
    },
];

#[derive(uniffi::Record, Clone)]
pub struct ModelDownload {
    pub id: String,
    pub title: String,
    pub detail: String,
    pub bytes: u64,
    pub installed: bool,
    /// Bytes of an interrupted download already on disk.
    pub partial: u64,
}

#[uniffi::export(with_foreign)]
pub trait ModelProgress: Send + Sync {
    fn on_progress(&self, received: u64, total: u64);
}

#[derive(uniffi::Object)]
pub struct ModelStore {
    dir: PathBuf,
}

fn sha256_file(p: &std::path::Path) -> Result<String, PbError> {
    let mut f = std::fs::File::open(p).map_err(|e| PbError::from(e.to_string()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf).map_err(|e| PbError::from(e.to_string()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[uniffi::export]
impl ModelStore {
    /// `dir`: where downloaded models live (Application Support). Also
    /// registers it with the engine so downloaded models load like bundled.
    #[uniffi::constructor]
    pub fn new(dir: String) -> Arc<Self> {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        pb_core::set_extra_model_dirs(vec![dir.clone()]);
        Arc::new(ModelStore { dir })
    }

    pub fn list(&self) -> Vec<ModelDownload> {
        DOWNLOADS
            .iter()
            .map(|d| ModelDownload {
                id: d.id.into(),
                title: d.title.into(),
                detail: d.detail.into(),
                bytes: d.bytes,
                installed: self.dir.join(d.file).exists(),
                partial: std::fs::metadata(self.dir.join(format!("{}.part", d.file))).map(|m| m.len()).unwrap_or(0),
            })
            .collect()
    }

    pub fn installed(&self, id: String) -> bool {
        DOWNLOADS.iter().find(|d| d.id == id).map(|d| self.dir.join(d.file).exists()).unwrap_or(false)
    }

    /// Download (or resume) and verify. Blocking: call off the main thread.
    pub fn download(&self, server: Arc<ServerClient>, id: String, progress: Arc<dyn ModelProgress>) -> Result<(), PbError> {
        let d = DOWNLOADS.iter().find(|d| d.id == id).ok_or_else(|| PbError::from(format!("unknown model {id}")))?;
        let t0 = std::time::Instant::now();
        let tmp = self.dir.join(format!("{}.dl", d.file));
        // download_to keeps its partial data in "<file>.part".
        let had = std::fs::metadata(self.dir.join(format!("{}.part", d.file))).map(|m| m.len()).unwrap_or(0);
        let r = server.download_to(&format!("/models/file/{}", d.file), &tmp, &|got, total| progress.on_progress(got, if total > 0 { total } else { d.bytes }));
        let outcome = match r {
            Ok(_) => {
                let sum = sha256_file(&tmp)?;
                if sum != d.sha256 {
                    let _ = std::fs::remove_file(&tmp);
                    Err(PbError::from(format!("{} failed verification (checksum mismatch); try again", d.title)))
                } else {
                    std::fs::rename(&tmp, self.dir.join(d.file)).map_err(|e| PbError::from(e.to_string()))
                }
            }
            Err(e) => Err(e),
        };
        pb_core::telemetry::event("models", "download")
            .since(t0)
            .tag("model", d.id)
            .tag("outcome", if outcome.is_ok() { "ok" } else { "error" })
            .tag("error", outcome.as_ref().err().map(|e| e.to_string()).unwrap_or_default())
            .metric("bytes", d.bytes as f64)
            .metric("resumed_from", had as f64)
            .emit();
        outcome
    }

    pub fn remove(&self, id: String) -> Result<(), PbError> {
        let d = DOWNLOADS.iter().find(|d| d.id == id).ok_or_else(|| PbError::from(format!("unknown model {id}")))?;
        let _ = std::fs::remove_file(self.dir.join(format!("{}.part", d.file)));
        match std::fs::remove_file(self.dir.join(d.file)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(PbError::from(e.to_string())),
        }
    }

    pub fn disk_bytes(&self) -> u64 {
        DOWNLOADS.iter().filter_map(|d| std::fs::metadata(self.dir.join(d.file)).ok()).map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_hashes_match_the_repo_files() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../Models/downloads");
        for d in DOWNLOADS {
            let p = dir.join(d.file);
            if !p.exists() {
                eprintln!("{} not here: skipping", d.file);
                continue;
            }
            assert_eq!(std::fs::metadata(&p).unwrap().len(), d.bytes, "{}", d.file);
            assert_eq!(sha256_file(&p).unwrap(), d.sha256, "{}", d.file);
        }
    }
}
