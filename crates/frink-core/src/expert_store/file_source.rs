//! [`FileRangeSource`]: one file, one contiguous byte range per expert.

use std::collections::HashMap;
use std::io;
#[cfg(not(unix))]
use std::sync::Mutex;

use super::{ExpertKey, ExpertSource};

/// A file-backed [`ExpertSource`]: each expert is one contiguous
/// `(offset, len)` byte range in a single file, read with positional
/// reads (`pread`-style on unix, so concurrent misses never contend on
/// a shared seek cursor). This is the portable buffered-I/O default
/// the storage design starts from; `O_DIRECT`/`F_NOCACHE`/`io_uring`
/// variants are future work behind the same trait and must produce
/// identical bytes.
pub struct FileRangeSource {
    file: std::fs::File,
    ranges: HashMap<ExpertKey, (u64, usize)>,
    /// Non-unix fallback only: serializes seek+read pairs.
    #[cfg(not(unix))]
    seek_lock: Mutex<()>,
}

impl FileRangeSource {
    pub fn new(file: std::fs::File, ranges: HashMap<ExpertKey, (u64, usize)>) -> Self {
        FileRangeSource {
            file,
            ranges,
            #[cfg(not(unix))]
            seek_lock: Mutex::new(()),
        }
    }
}

impl ExpertSource for FileRangeSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        self.ranges.get(&key).map(|&(_, len)| len)
    }

    fn read_expert(&self, key: ExpertKey) -> io::Result<Vec<u8>> {
        let &(offset, len) = self
            .ranges
            .get(&key)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("{key:?}")))?;
        let mut buf = vec![0u8; len];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut buf, offset)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let _guard = self.seek_lock.lock().unwrap_or_else(|p| p.into_inner());
            let mut f = &self.file;
            f.seek(SeekFrom::Start(offset))?;
            f.read_exact(&mut buf)?;
        }
        Ok(buf)
    }
}
