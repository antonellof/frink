//! The one outbound HTTP client in `frink-server`, used only by
//! `/admin/download`.
//!
//! ## Why a client dependency at all
//!
//! The workspace had no HTTP *client* before this: `axum`/`hyper` are
//! the server side, and `frink_models::hf_pull` shells out to the
//! `huggingface_hub` CLI, which is not installed on most machines, is
//! not resumable under our control, and reports progress only to its
//! own terminal. A download the UI can show a real progress bar for
//! needs the byte counter in-process, so `ureq` (blocking, rustls) is
//! added -- deliberately with `default-features = false` so it brings
//! rustls and webpki-roots, most of which the TLS-serving side already
//! pulled in, and not a second async runtime.
//!
//! Blocking on purpose: every call here runs on `spawn_blocking`
//! alongside the file writes it feeds, so the Tokio reactor is never
//! parked on a socket read.
//!
//! ## Redirects and resume
//!
//! `…/resolve/main/<file>` redirects to a CDN host. Whether a `Range`
//! header survives that hop is not something this code assumes: it asks
//! for a range and then believes the *status*. Only a `206` counts as a
//! resume; a `200` means the server is sending the whole file from byte
//! zero, and the caller truncates rather than appending a second copy of
//! the prefix onto the first.

use std::io::Read;
use std::net::{SocketAddr, ToSocketAddrs};
use std::time::Duration;

/// Hub base URL. `HF_ENDPOINT` is the same variable the official client
/// reads, for mirrors and air-gapped caches. It is operator
/// configuration, never taken from a request body.
fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim_end_matches('/').to_string())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

fn token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]
        .iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|t| !t.trim().is_empty())
}

/// Resolves a host and returns its addresses **IPv4 first**.
///
/// Not cosmetic. `huggingface.co` publishes eight AAAA records and four
/// A records, and the client tries them in order with a *halving* share
/// of the connect deadline each time. On a host with no IPv6 route --
/// which is most laptops, most CI, and every container run without
/// `--ipv6` -- every AAAA attempt burns until it times out, and the
/// budget is exhausted before the first working A record is reached.
/// The observable symptom is a download that sits at zero bytes and
/// then fails with "connection timed out" on a host where `curl` to the
/// same URL succeeds instantly, because curl does Happy Eyeballs and
/// this client does not.
///
/// Ordering rather than filtering: an IPv6-only host still gets its
/// AAAA records, just after the A records, and a connect to an
/// unreachable IPv4 address fails immediately (`ENETUNREACH`) rather
/// than hanging, so the cost of the wrong guess is bounded in the
/// direction that matters.
fn resolve_ipv4_first(netloc: &str) -> std::io::Result<Vec<SocketAddr>> {
    let mut addrs: Vec<SocketAddr> = netloc.to_socket_addrs()?.collect();
    // Stable sort: the resolver's own ordering is preserved inside each
    // family, so DNS round-robin still spreads load.
    addrs.sort_by_key(|addr| !addr.is_ipv4());
    Ok(addrs)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        // Enough for a cold CDN edge to answer, short enough that a
        // black-holed address does not hold a blocking thread for
        // anything like a user's patience.
        .timeout_connect(Duration::from_secs(10))
        .resolver(resolve_ipv4_first)
        .build()
}

fn with_auth(req: ureq::Request) -> ureq::Request {
    match token() {
        Some(t) => req.set("Authorization", &format!("Bearer {t}")),
        None => req,
    }
}

/// Turns a `*` glob into a concrete filename by asking the Hub what the
/// repo contains.
///
/// Sorted, first match wins, so the choice is deterministic for the
/// same repo rather than depending on the order the API happens to
/// return. A repo with several matching quantizations is ambiguous by
/// construction; the caller can name one exactly to be sure.
/// Where a `-hf` download lands, and where a second run finds it
/// already there.
///
/// `FRINK_CACHE`, else `$XDG_CACHE_HOME/frink`, else `~/.cache/frink`.
/// llama.cpp does the same thing with `LLAMA_CACHE`, and for the same
/// reason: a model fetched by `-hf` is not part of the project you are
/// standing in, so it does not belong in `./models`.
pub fn cache_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("FRINK_CACHE") {
        return std::path::PathBuf::from(dir);
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return std::path::PathBuf::from(dir).join("frink");
    }
    match std::env::var_os("HOME") {
        Some(home) => std::path::PathBuf::from(home).join(".cache").join("frink"),
        // No HOME is a real case in a container. A relative path is a
        // worse cache than an absolute one and a better failure than a
        // panic.
        None => std::path::PathBuf::from(".frink-cache"),
    }
}

/// A `-hf` argument: `user/repo`, optionally `:QUANT`.
///
/// llama.cpp's spelling, so a command copied off a model card runs
/// unchanged: `-hf TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF:Q4_K_M`.
/// The tag after the colon is a QUANT LABEL, not a git revision, which
/// is worth saying because the same `repo:thing` shape means a revision
/// almost everywhere else.
///
/// Without a tag the pattern is every GGUF in the repo, which resolves
/// only when there is exactly one and otherwise refuses by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfRef {
    pub repo: String,
    /// An exact filename, llama.cpp's `-hff`. When set it wins over
    /// `quant`: the user named the file, so there is nothing left to
    /// resolve and nothing to guess.
    pub file: Option<String>,
    /// The quant label, upper-cased for reporting. Matching is
    /// case-insensitive: repos spell it `Q4_K_M` and `q4_k_m` about
    /// equally often, and a case-sensitive match would 404 on half of
    /// them for a reason the user cannot see.
    pub quant: Option<String>,
}

impl HfRef {
    /// Splits on the LAST colon, so a repo id containing one is still
    /// parsed the way the user meant.
    pub fn parse(spec: &str) -> Self {
        match spec.rsplit_once(':') {
            Some((repo, quant)) if !repo.is_empty() && !quant.is_empty() => HfRef {
                repo: repo.to_string(),
                file: None,
                quant: Some(quant.to_ascii_uppercase()),
            },
            // A TRAILING colon is a typo, not a tag, and the colon is
            // dropped rather than carried into the repo id. Keeping it
            // sent `owner/repo:` to the Hub and came back a 404 about a
            // repo the user can see exists.
            Some((repo, _)) if !repo.is_empty() => HfRef {
                repo: repo.to_string(),
                file: None,
                quant: None,
            },
            _ => HfRef {
                repo: spec.to_string(),
                file: None,
                quant: None,
            },
        }
    }

    /// The filename glob this reference resolves through.
    pub fn pattern(&self) -> String {
        match &self.quant {
            Some(q) => format!("*{q}*.gguf"),
            None => "*.gguf".to_string(),
        }
    }

    /// Resolves to one filename in the repo.
    ///
    /// An explicit `file` short-circuits the Hub listing entirely: the
    /// user named it, so asking the API which files exist would only
    /// add a way to fail.
    pub fn resolve(&self) -> Result<String, String> {
        match &self.file {
            Some(f) => Ok(f.clone()),
            None => resolve_glob_ci(&self.repo, &self.pattern()),
        }
    }
}

impl HfRef {
    /// The local path this reference resolves to, downloading it into
    /// [`cache_dir`] only when it is not already there.
    ///
    /// Returns `(path, downloaded)` so a caller can say "using cached"
    /// rather than printing a progress bar that never moves.
    pub fn ensure_local(
        &self,
        on_progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(std::path::PathBuf, bool), String> {
        self.ensure_local_with(false, on_progress)
    }

    /// Where this repo's files live in [`cache_dir`].
    ///
    /// Under `hub/` rather than directly in the cache root, which
    /// already holds `instances/` (the running-instance registry). A
    /// repo named `instances` would otherwise land on top of it.
    pub fn cache_path(&self) -> std::path::PathBuf {
        cache_dir().join("hub").join(self.repo.replace('/', "__"))
    }

    /// [`Self::ensure_local`], or with `offline` llama.cpp's `--offline`
    /// (`common/arg.cpp`: "forces use of cache, prevents network
    /// access"): the file or quant tag is resolved against what
    /// [`Self::cache_path`] already holds, and nothing is listed or
    /// fetched from the Hub. A reference the cache cannot answer is
    /// refused naming the directory it looked in.
    pub fn ensure_local_with(
        &self,
        offline: bool,
        on_progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<(std::path::PathBuf, bool), String> {
        let dir = self.cache_path();
        if offline {
            return self.resolve_in_dir(&dir).map(|p| (p, false));
        }
        let filename = self.resolve()?;
        let path = dir.join(&filename);
        if path.is_file() {
            return Ok((path, false));
        }
        let path = fetch_to_dir_with_progress(&self.repo, &filename, &dir, on_progress)?;
        Ok((path, true))
    }
}

impl HfRef {
    /// The cached file this reference names, from `dir` alone: the exact
    /// `-hff` filename when one was given, else the first GGUF whose name
    /// matches [`Self::pattern`] without regard to case, in sorted order,
    /// which is how [`resolve_glob_ci`] picks from the Hub's listing. A
    /// `.partial` download is not a match: it is not a whole file.
    pub fn resolve_in_dir(&self, dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
        let not_cached = || {
            format!(
                "--offline: {}{} is not in the cache ({}); run once without --offline to fetch it",
                self.repo,
                self.file
                    .as_deref()
                    .map(|f| format!(" / {f}"))
                    .or_else(|| self.quant.as_deref().map(|q| format!(":{q}")))
                    .unwrap_or_default(),
                dir.display()
            )
        };
        if let Some(f) = &self.file {
            let path = dir.join(f);
            return if path.is_file() {
                Ok(path)
            } else {
                Err(not_cached())
            };
        }
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map_err(|_| not_cached())?
            .filter_map(|e| {
                let e = e.ok()?;
                e.file_type()
                    .ok()?
                    .is_file()
                    .then(|| e.file_name().into_string().ok())?
            })
            .collect();
        names.sort();
        let lowered = self.pattern().to_ascii_lowercase();
        names
            .into_iter()
            .find(|n| glob_matches(&lowered, &n.to_ascii_lowercase()))
            .map(|n| dir.join(n))
            .ok_or_else(not_cached)
    }
}

/// [`resolve_glob`], matching without regard to case.
///
/// Separate from `resolve_glob` rather than replacing it: the exact
/// pattern a caller passes to `frink download` should mean what it
/// says, and only the `-hf` quant tag is case-insensitive.
pub fn resolve_glob_ci(repo: &str, pattern: &str) -> Result<String, String> {
    let names = list_files(repo)?;
    let lowered = pattern.to_ascii_lowercase();
    names
        .iter()
        .find(|n| glob_matches(&lowered, &n.to_ascii_lowercase()))
        .cloned()
        .ok_or_else(|| {
            let ggufs: Vec<&str> = names
                .iter()
                .filter(|n| n.to_ascii_lowercase().ends_with(".gguf"))
                .map(String::as_str)
                .collect();
            if ggufs.is_empty() {
                format!("no file in {repo} matches '{pattern}', and the repo holds no GGUF at all")
            } else {
                // Listing what IS there turns "no match" into an
                // answerable question: the usual cause is a quant this
                // repo does not publish, and the user cannot see the
                // repo's file list from a command line.
                format!(
                    "no file in {repo} matches '{pattern}'. That repo publishes: {}",
                    ggufs.join(", ")
                )
            }
        })
}

/// Every root-level filename in `repo`.
fn list_files(repo: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/api/models/{repo}", endpoint());
    let response = with_auth(agent().get(&url))
        .call()
        .map_err(|e| format!("listing {repo} on the Hub failed: {e}"))?;
    let body = response
        .into_string()
        .map_err(|e| format!("reading the file list for {repo}: {e}"))?;
    let value: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("the Hub's file list for {repo} was not JSON: {e}"))?;
    let mut names: Vec<String> = value
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|e| e.get("rfilename")?.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    // A repo may hold shards in subdirectories; a target this server
    // can safely write is a bare name in the repo root.
    names.retain(|n| !n.contains('/'));
    Ok(names)
}

pub fn resolve_glob(repo: &str, pattern: &str) -> Result<String, String> {
    let url = format!("{}/api/models/{repo}", endpoint());
    let response = with_auth(agent().get(&url))
        .call()
        .map_err(|e| format!("listing {repo} on the Hub failed: {e}"))?;
    let body = response
        .into_string()
        .map_err(|e| format!("reading the file list for {repo}: {e}"))?;
    let value: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("the Hub's file list for {repo} was not JSON: {e}"))?;
    let mut names: Vec<String> = value
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|e| e.get("rfilename")?.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
        .into_iter()
        // A repo may hold shards in subdirectories; a target this
        // server can safely write is a bare name in the repo root.
        .filter(|n| !n.contains('/'))
        .find(|n| glob_matches(pattern, n))
        .ok_or_else(|| format!("no file in {repo} matches '{pattern}'"))
}

/// An open response body plus what the caller needs to write it.
pub struct HubFile {
    pub body: Box<dyn Read + Send>,
    /// Total size of the *whole* file when the server states one, not
    /// of the remaining range: the caller's progress counter is
    /// cumulative. `None` when there is no usable `Content-Length`,
    /// which is a real case and means an indeterminate progress bar
    /// rather than a fabricated total.
    pub total_bytes: Option<u64>,
    /// True only on a `206`. See the module docs.
    pub resumed: bool,
}

/// Opens `repo`/`filename` for reading, asking to resume at
/// `resume_from` when that is non-zero.
pub fn open_file(repo: &str, filename: &str, resume_from: u64) -> Result<HubFile, String> {
    let url = format!("{}/{repo}/resolve/main/{filename}", endpoint());
    let mut req = with_auth(agent().get(&url));
    if resume_from > 0 {
        req = req.set("Range", &format!("bytes={resume_from}-"));
    }
    let response = req
        .call()
        .map_err(|e| format!("downloading {filename} from {repo} failed: {e}"))?;

    let status = response.status();
    let resumed = status == 206 && resume_from > 0;
    let total_bytes = if resumed {
        content_range_total(response.header("content-range"))
    } else {
        response
            .header("content-length")
            .and_then(|v| v.trim().parse::<u64>().ok())
    };
    Ok(HubFile {
        body: Box::new(response.into_reader()),
        total_bytes,
        resumed,
    })
}

/// Total size out of `Content-Range: bytes 100-999/1000`. `None` for
/// the `*/unknown` form, which is legal and means exactly that.
fn content_range_total(header: Option<&str>) -> Option<u64> {
    let total = header?.rsplit('/').next()?.trim();
    total.parse::<u64>().ok()
}

/// Matches a `*`-glob against a filename. `*` matches any run of
/// characters including none; every other character is literal.
///
/// Enough for the Hub filename patterns people actually type
/// (`*.gguf`, `*Q4_K_M*.gguf`) and small enough to read in one go.
pub fn glob_matches(pattern: &str, name: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == name;
    }
    let mut rest = name;
    if let Some(first) = segments.first() {
        let Some(stripped) = rest.strip_prefix(first) else {
            return false;
        };
        rest = stripped;
    }
    if let Some(last) = segments.last() {
        let Some(stripped) = rest.strip_suffix(last) else {
            return false;
        };
        // A pattern like "a*b" over "ab" leaves nothing between: fine.
        rest = stripped;
    }
    for middle in segments
        .iter()
        .skip(1)
        .take(segments.len().saturating_sub(2))
    {
        match rest.find(*middle) {
            Some(at) => rest = &rest[at + middle.len()..],
            None => return false,
        }
    }
    true
}

/// Resolve `pattern` in `repo` and write the matching file into `dir`,
/// resuming an interrupted download rather than starting over.
///
/// Returns the path written. Progress goes to `on_progress` so a CLI
/// can draw a bar and a server can stay silent.
pub fn fetch_to_dir_with_progress(
    repo: &str,
    pattern: &str,
    dir: &std::path::Path,
    mut on_progress: impl FnMut(u64, Option<u64>),
) -> Result<std::path::PathBuf, String> {
    use std::io::{Read, Write};

    let filename = resolve_glob(repo, pattern)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let final_path = dir.join(&filename);

    // Resume needs the partial bytes to survive an interruption, but a
    // partial file under the final name would later be opened as if it
    // were a whole GGUF. So it lands beside the real name and is
    // renamed only once the last byte is in.
    let partial = dir.join(format!("{filename}.partial"));
    if final_path.exists() {
        return Ok(final_path);
    }
    let resume_from = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

    let mut hub = open_file(repo, &filename, resume_from)?;
    // A server that ignores `Range` answers 200 with the whole file.
    // Appending in that case would corrupt it, so what actually
    // happened is read off the response, not off what was asked for.
    let mut done = if hub.resumed { resume_from } else { 0 };
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(hub.resumed)
        .truncate(!hub.resumed)
        .open(&partial)
        .map_err(|e| format!("{}: {e}", partial.display()))?;

    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = hub
            .body
            .read(&mut buf)
            .map_err(|e| format!("reading {filename} from {repo} failed: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .map_err(|e| format!("{}: {e}", partial.display()))?;
        done += n as u64;
        on_progress(done, hub.total_bytes);
    }
    out.flush().map_err(|e| e.to_string())?;
    // The rename must not outrun the bytes it publishes.
    out.sync_all().map_err(|e| e.to_string())?;
    drop(out);

    // A short file is a truncated download. Renaming it would hand the
    // loader a GGUF that ends in the middle of a tensor, so it stays
    // under the partial name and the next run resumes it.
    if let Some(total) = hub.total_bytes {
        if done != total {
            return Err(format!(
                "{filename} stopped at {done} of {total} bytes. The partial file is kept \
                 at {}, so running this again resumes rather than restarting.",
                partial.display()
            ));
        }
    }

    std::fs::rename(&partial, &final_path).map_err(|e| e.to_string())?;
    Ok(final_path)
}

/// [`fetch_to_dir_with_progress`] without the progress callback.
pub fn fetch_to_dir(
    repo: &str,
    pattern: &str,
    dir: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    fetch_to_dir_with_progress(repo, pattern, dir, |_, _| {})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_content_range_yields_the_whole_file_size_not_the_slice() {
        assert_eq!(content_range_total(Some("bytes 100-999/1000")), Some(1000));
    }

    #[test]
    fn an_unknown_content_range_total_is_not_invented() {
        assert_eq!(content_range_total(Some("bytes 0-99/*")), None);
        assert_eq!(content_range_total(None), None);
        assert_eq!(content_range_total(Some("garbage")), None);
    }

    #[test]
    fn resolution_puts_every_ipv4_address_ahead_of_every_ipv6_one() {
        // localhost resolves to both families on most hosts; the shape
        // of the answer is what matters, not which addresses appear.
        let addrs = resolve_ipv4_first("localhost:443").expect("localhost must resolve");
        assert!(!addrs.is_empty());
        let first_v6 = addrs.iter().position(|a| !a.is_ipv4());
        let last_v4 = addrs.iter().rposition(|a| a.is_ipv4());
        if let (Some(first_v6), Some(last_v4)) = (first_v6, last_v4) {
            assert!(
                last_v4 < first_v6,
                "an IPv6 address was ordered ahead of an IPv4 one: {addrs:?}"
            );
        }
    }

    #[test]
    fn the_endpoint_has_no_trailing_slash_to_double_up() {
        // Default, and any override, must join cleanly with "/api/...".
        assert!(!endpoint().ends_with('/'));
    }
}

#[cfg(test)]
mod hf_ref_tests {
    use super::*;

    /// llama.cpp's `-hf user/repo:QUANT`. The tag is a quant label, not
    /// a git revision, which is the opposite of what `repo:thing` means
    /// nearly everywhere else.
    #[test]
    fn a_quant_tag_is_split_off_and_becomes_a_glob() {
        let r = HfRef::parse("TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF:Q4_K_M");
        assert_eq!(r.repo, "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(r.pattern(), "*Q4_K_M*.gguf");
    }

    /// Without a tag the whole string is the repo. Sending
    /// `repo:Q4_K_M` to the Hub as a repo id is what produced a bare
    /// `401`, which reads like an auth failure and is not one.
    #[test]
    fn a_bare_repo_keeps_its_whole_name_and_matches_every_gguf() {
        let r = HfRef::parse("bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(r.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(r.quant, None);
        assert_eq!(r.pattern(), "*.gguf");
    }

    /// A trailing or leading colon is not a tag. Treating `repo:` as
    /// "quant is empty string" would build the pattern `**.gguf` and
    /// match a file the user did not ask for.
    #[test]
    fn a_degenerate_colon_is_not_a_tag() {
        assert_eq!(HfRef::parse("owner/repo:").quant, None);
        assert_eq!(HfRef::parse("owner/repo:").repo, "owner/repo");
        assert_eq!(HfRef::parse(":Q4_K_M").quant, None);
    }

    /// Case-insensitive on purpose: repos spell the quant `Q4_K_M` and
    /// `q4_k_m` about equally often, and a case-sensitive match would
    /// fail on half of them for a reason invisible from a command line.
    #[test]
    fn a_lowercase_tag_matches_an_uppercase_filename() {
        let r = HfRef::parse("owner/repo:q4_k_m");
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
        assert!(glob_matches(
            &r.pattern().to_ascii_lowercase(),
            &"SmolLM2-135M-Instruct-Q4_K_M.gguf".to_ascii_lowercase()
        ));
    }

    /// A quant label is a substring of a longer one, so the glob must
    /// not let `Q4_K_M` answer a request for `Q4_K`.
    #[test]
    fn a_tag_does_not_match_a_longer_quant_by_accident() {
        let asked = HfRef::parse("owner/repo:Q4_K_M");
        assert!(!glob_matches(
            &asked.pattern().to_ascii_lowercase(),
            &"model-Q4_K_S.gguf".to_ascii_lowercase()
        ));
    }

    /// The cache is per repo, and under `hub/` rather than the cache
    /// root, which already holds `instances/`.
    #[test]
    fn the_cache_path_is_namespaced_under_hub() {
        let dir = cache_dir();
        assert!(
            !dir.as_os_str().is_empty(),
            "a cache dir must always resolve, even with no HOME"
        );
    }

    /// An explicit `--hf-file` wins over the quant tag and skips the
    /// Hub listing entirely: the user named the file, so asking the API
    /// which files exist only adds a way to fail.
    fn cache_with(files: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "frink-hub-offline-{}-{}",
            std::process::id(),
            files.join("+").len()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for f in files {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        dir
    }

    /// `--offline` resolves a quant tag against the cache directory
    /// alone, case-insensitively, skipping a half-written `.partial`.
    #[test]
    fn offline_resolves_a_quant_tag_from_the_cache_alone() {
        let dir = cache_with(&["m-q4_k_m.gguf.partial", "m-Q4_K_M.gguf", "m-Q8_0.gguf"]);
        let got = HfRef::parse("org/repo:q4_k_m")
            .resolve_in_dir(&dir)
            .unwrap();
        assert_eq!(got, dir.join("m-Q4_K_M.gguf"));
        let mut exact = HfRef::parse("org/repo");
        exact.file = Some("m-Q8_0.gguf".into());
        assert_eq!(exact.resolve_in_dir(&dir).unwrap(), dir.join("m-Q8_0.gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A reference the cache cannot answer is refused naming where it
    /// looked, never sent to the network.
    #[test]
    fn offline_refuses_what_the_cache_does_not_hold() {
        let dir = cache_with(&["m-Q8_0.gguf"]);
        let err = HfRef::parse("org/repo:Q4_K_M")
            .resolve_in_dir(&dir)
            .unwrap_err();
        assert!(
            err.contains("--offline") && err.contains("org/repo:Q4_K_M"),
            "{err}"
        );
        assert!(err.contains(&dir.display().to_string()), "{err}");
        let missing = dir.join("nope");
        assert!(HfRef::parse("org/repo").resolve_in_dir(&missing).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_explicit_file_short_circuits_resolution() {
        let mut r = HfRef::parse("owner/repo:Q4_K_M");
        r.file = Some("exact-name.gguf".into());
        assert_eq!(r.resolve().unwrap(), "exact-name.gguf");
    }
}
