//! `-hf user/repo[:QUANT]`, llama.cpp's one-command model fetch.
//!
//! The parsing and the cache live in [`frink_models::hub`], which the
//! server uses too; this is the CLI's progress reporting around them.
//! Kept as one function rather than two so `frink run` and
//! `frink serve` cannot drift about where a model lands or what they
//! print when it is already there.

/// Resolves a `-hf` reference to a local path, downloading it once.
///
/// Progress goes to stderr so a caller redirecting stdout gets the
/// model's output and not a progress bar.
pub fn resolve(spec: &str, file: Option<&str>, offline: bool) -> anyhow::Result<String> {
    let mut hf = frink_models::hub::HfRef::parse(spec);
    if let Some(f) = file {
        hf.file = Some(f.to_string());
    }
    eprintln!(
        "frink: resolving {} {}{}",
        hf.repo,
        if offline {
            "in the cache"
        } else {
            "on the Hub"
        },
        hf.quant
            .as_deref()
            .map(|q| format!(" ({q})"))
            .unwrap_or_default()
    );

    let mut last = std::time::Instant::now();
    let mut draw = move |done: u64, total: Option<u64>| {
        if last.elapsed() < std::time::Duration::from_millis(200) {
            return;
        }
        last = std::time::Instant::now();
        let mib = done as f64 / 1024.0 / 1024.0;
        match total {
            Some(t) if t > 0 => eprint!(
                "\r  {mib:>9.1} MiB  {:5.1}%",
                (done as f64 / t as f64) * 100.0
            ),
            _ => eprint!("\r  {mib:>9.1} MiB"),
        }
    };

    let (path, downloaded) = hf
        .ensure_local_with(offline, &mut draw)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if downloaded {
        eprintln!();
        eprintln!("frink: downloaded {}", path.display());
    } else {
        eprintln!("frink: using cached {}", path.display());
    }
    Ok(path.to_string_lossy().into_owned())
}
