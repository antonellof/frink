//! The ORDER the sampler chain runs in, as llama.cpp's `--samplers`
//! spells it, and the refusal for every sampler frink does not have.
//!
//! # Why the order is a parameter and not a constant
//!
//! llama.cpp lets a caller reorder its chain (`--samplers`,
//! `--sampler-seq`, and the server's `samplers` request field). frink
//! ran one fixed order, so a command that worked upstream either could
//! not be expressed here or -- worse -- was accepted with the field
//! dropped and answered under a different chain.
//!
//! The order is not cosmetic. `sampler_chain` models the SHRINKING
//! candidate list llama.cpp passes down the chain, and each filter
//! renormalises over the survivors, so moving one step changes which
//! candidates exist for the next. frink has already shipped the bug:
//! temperature used to run FIRST, and top-p then summed probabilities
//! temperature had already reshaped, keeping a different candidate set
//! for identical flags.
//!
//! # The one table
//!
//! A list of names in the CLI and a second list in the server is this
//! repo's dominant defect shape: two structures that must agree with
//! nothing enforcing it. So there is exactly one list, in the
//! `sampler_names!` invocation below, and the enum, the canonical
//! spellings, the alias table and the parser are all generated from it.
//! Adding a name is one row.
//!
//! [`SamplerName::implemented`] is the second half of the guarantee: an
//! EXHAUSTIVE `match`, no `..`, that must say for every name whether
//! frink runs it or why it does not. A name added to the table without
//! a verdict does not compile.
//!
//! # Partial support, stated
//!
//! frink implements llama.cpp's whole default chain: `penalties`,
//! `dry`, `top_n_sigma`, `top_k`, `typ_p`, `top_p`, `min_p`, `xtc` and
//! `temperature`. `mirostat` and `infill` are named in the table purely
//! so that asking for one is a refusal that says WHICH sampler is
//! missing, rather than an unknown-name error or, far worse, a chain
//! quietly built without it. A caller who asked for `mirostat` and was
//! served a chain without it got a different sampler than they
//! requested and no way to tell.

use std::fmt;
use std::str::FromStr;

/// Generates the name table: the enum, `ALL`, the canonical spelling of
/// each variant, and the parser -- from ONE list of rows.
///
/// Each row is `Variant => 'c', "canonical" | "alias" | ...`. The
/// aliases are llama.cpp's own (`common_sampler_type_from_name`'s
/// `sampler_alt_name_map`), so a command line that works upstream parses
/// here, and the character is `common_sampler_type_to_chr`'s, the
/// spelling `--sampler-seq` takes (`'?'` where upstream has none).
macro_rules! sampler_names {
    ($($variant:ident => $chr:literal, $canonical:literal $(| $alias:literal)* ),+ $(,)?) => {
        /// Every sampler name llama.cpp's `--samplers` accepts.
        ///
        /// Membership here says the name is REAL, not that frink
        /// implements it; [`SamplerName::implemented`] decides that.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum SamplerName {
            $($variant),+
        }

        impl SamplerName {
            /// Every name, in llama.cpp's own default chain order.
            pub const ALL: &'static [SamplerName] = &[$(SamplerName::$variant),+];

            /// llama.cpp's canonical spelling, which is also what
            /// [`SamplerOrder`] prints.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(SamplerName::$variant => $canonical),+
                }
            }

            /// One name, canonical spelling or llama.cpp alias.
            ///
            /// `name` is expected already trimmed and lowercased; see
            /// [`SamplerOrder::from_names`].
            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($canonical $(| $alias)* => Some(SamplerName::$variant),)+
                    _ => None,
                }
            }

            /// One `--sampler-seq` character (`common/sampling.cpp`,
            /// `common_sampler_type_to_chr`).
            pub fn from_chr(c: char) -> Option<Self> {
                if c == '?' {
                    return None;
                }
                $(if c == $chr {
                    return Some(SamplerName::$variant);
                })+
                None
            }
        }
    };
}

sampler_names! {
    Penalties   => 'e', "penalties",
    Dry         => 'd', "dry",
    TopNSigma   => 's', "top_n_sigma" | "top-n-sigma",
    TopK        => 'k', "top_k" | "top-k",
    TypP        => 'y', "typ_p" | "typ-p" | "typ" | "typical" | "typical_p" | "typical-p",
    TopP        => 'p', "top_p" | "top-p" | "nucleus",
    MinP        => 'm', "min_p" | "min-p",
    Xtc         => 'x', "xtc",
    Temperature => 't', "temperature" | "temp",
    Mirostat    => '?', "mirostat",
    Infill      => 'i', "infill",
}

/// A step the frink chain can actually run.
///
/// [`SamplerOrder`] holds these rather than [`SamplerName`]s so that an
/// unimplemented sampler is unrepresentable once the order exists: the
/// applier's `match` is total over what it can be handed, with no
/// `unreachable!` standing in for a check that happened somewhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChainStep {
    /// The repetition / presence / frequency penalties.
    Penalties,
    /// The DRY sequence-repetition penalty; see [`crate::dry`].
    Dry,
    TopNSigma,
    TopK,
    /// Locally typical sampling, llama.cpp's `typ_p`.
    TypP,
    TopP,
    MinP,
    /// "Exclude top choices", llama.cpp's `xtc`.
    Xtc,
    Temperature,
}

impl ChainStep {
    /// The name this step is spelled with. Exhaustive, so a step added
    /// without a name does not compile.
    pub const fn name(self) -> SamplerName {
        match self {
            ChainStep::Penalties => SamplerName::Penalties,
            ChainStep::Dry => SamplerName::Dry,
            ChainStep::TopNSigma => SamplerName::TopNSigma,
            ChainStep::TopK => SamplerName::TopK,
            ChainStep::TypP => SamplerName::TypP,
            ChainStep::TopP => SamplerName::TopP,
            ChainStep::MinP => SamplerName::MinP,
            ChainStep::Xtc => SamplerName::Xtc,
            ChainStep::Temperature => SamplerName::Temperature,
        }
    }

    /// Every step this engine can run, DERIVED from the name table
    /// rather than restated beside it.
    ///
    /// A hand-written list here would be a second structure that had to
    /// agree with [`SamplerName::implemented`], which is the defect
    /// shape this whole module is built to avoid. Derived, "is a step"
    /// and "has a name that maps to it" are the same statement, and
    /// `tests::every_step_the_name_table_yields_names_itself_back`
    /// closes the loop in the other direction.
    pub fn all() -> Vec<ChainStep> {
        SamplerName::ALL
            .iter()
            .filter_map(|n| n.implemented().ok())
            .collect()
    }
}

impl SamplerName {
    /// The step frink runs for this name, or the reason it has none.
    ///
    /// EXHAUSTIVE ON PURPOSE, with no `..`: a name added to the table
    /// above stops this crate compiling here until someone states
    /// whether frink implements it. The alternative -- a `_ => Err(..)`
    /// arm -- would let a sampler frink *does* have be added to the
    /// table and silently refused, which is the same class of silence
    /// this module exists to close.
    pub const fn implemented(self) -> Result<ChainStep, &'static str> {
        match self {
            SamplerName::Penalties => Ok(ChainStep::Penalties),
            SamplerName::Dry => Ok(ChainStep::Dry),
            SamplerName::TopNSigma => Ok(ChainStep::TopNSigma),
            SamplerName::TopK => Ok(ChainStep::TopK),
            SamplerName::TypP => Ok(ChainStep::TypP),
            SamplerName::TopP => Ok(ChainStep::TopP),
            SamplerName::MinP => Ok(ChainStep::MinP),
            SamplerName::Xtc => Ok(ChainStep::Xtc),
            SamplerName::Temperature => Ok(ChainStep::Temperature),
            SamplerName::Mirostat => Err(
                "mirostat is not implemented. It is not a chain member upstream either \
                 (llama.cpp spells it `--mirostat` and it REPLACES the chain), so there \
                 is no position in this order that would honour it",
            ),
            SamplerName::Infill => Err(
                "the infill sampler is not implemented: it needs the model's FIM tokens \
                 and a whitespace-aware candidate merge that this engine does not have",
            ),
        }
    }
}

/// Why a caller-supplied sampler order was refused.
///
/// Every variant names the sampler it is about. A refusal that said only
/// "bad sampler list" would leave the caller guessing which of five
/// names this engine disliked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplerOrderError {
    /// No sampler by this name exists in llama.cpp either.
    Unknown(String),
    /// A real llama.cpp sampler that frink does not implement.
    Unimplemented {
        name: &'static str,
        reason: &'static str,
    },
    /// The same sampler named twice.
    Duplicate(&'static str),
    /// `penalties` somewhere other than the front. See
    /// [`SamplerOrder::from_names`].
    PenaltiesNotFirst,
    /// A chain with no `temperature` step. See
    /// [`SamplerOrder::from_names`].
    TemperatureMissing,
    /// An empty chain: `--samplers ""`.
    Empty,
}

impl fmt::Display for SamplerOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SamplerOrderError::Unknown(name) => write!(
                f,
                "unknown sampler `{name}`. This engine accepts {}",
                SamplerName::ALL
                    .iter()
                    .map(|n| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            SamplerOrderError::Unimplemented { name, reason } => write!(
                f,
                "sampler `{name}` is not implemented in frink: {reason}. It is refused \
                 rather than skipped, because a chain built without a sampler you asked \
                 for is a different sampler and you would have no way to tell. \
                 Implemented: {}",
                SamplerOrder::implemented_names().join(", ")
            ),
            SamplerOrderError::Duplicate(name) => write!(
                f,
                "sampler `{name}` is named twice; each sampler may appear at most once"
            ),
            SamplerOrderError::PenaltiesNotFirst => write!(
                f,
                "`penalties` must be the FIRST sampler in the chain. frink applies the \
                 repetition / presence / frequency penalties to the whole vocabulary \
                 before the candidate list exists, so a `penalties` placed after a \
                 truncation filter would penalise a different candidate set than the one \
                 you asked for. Put it first, or leave it out to disable the penalties"
            ),
            SamplerOrderError::TemperatureMissing => write!(
                f,
                "the chain must include `temperature`. frink decides greedy-versus-sampled \
                 from the temperature BEFORE the chain runs -- on Metal at `temp <= 0` the \
                 decoder folds the argmax into the GPU stack and hands the sampler a single \
                 precomputed token id, so there is no candidate list left for a chain \
                 without a temperature step to filter. Dropping `temperature` from the list \
                 buys nothing anyway: it is exactly `--temp 1.0` with the step kept"
            ),
            SamplerOrderError::Empty => write!(
                f,
                "the sampler list is empty; name at least one of {}",
                SamplerOrder::implemented_names().join(", ")
            ),
        }
    }
}

impl std::error::Error for SamplerOrderError {}

/// A validated sampler chain: the steps frink will run, in the order it
/// will run them.
///
/// Fixed-capacity and `Copy` because [`crate::sampling::SamplingParams`]
/// is cloned per request and read per token; a `Vec` here would be an
/// allocation on that path for at most
/// [`SamplerName::ALL`]`.len()` entries.
#[derive(Debug, Clone, Copy)]
pub struct SamplerOrder {
    steps: [ChainStep; SamplerName::ALL.len()],
    len: usize,
}

/// llama.cpp's default chain, verbatim: `penalties, dry, top_n_sigma,
/// top_k, typical_p, top_p, min_p, xtc, temperature`
/// (`common/common.h:259-269`), with **temperature last**.
///
/// The four steps frink gained in `feat/sampler-parity` are all
/// NO-OPS at their neutral defaults (`dry_multiplier 0.0`,
/// `top_n_sigma -1.0`, `typical_p 1.0`, `xtc_probability 0.0`), so
/// adding them to the default chain changes no run that did not
/// configure them -- asserted bit-for-bit by
/// `sampling::tests::the_default_order_is_the_chain_frink_already_ran`,
/// which compares against the five-step chain frink used to run.
///
/// This is the single definition of "the default". Changing it changes
/// every run that did not pass `--samplers`.
const DEFAULT_STEPS: [ChainStep; 9] = [
    ChainStep::Penalties,
    ChainStep::Dry,
    ChainStep::TopNSigma,
    ChainStep::TopK,
    ChainStep::TypP,
    ChainStep::TopP,
    ChainStep::MinP,
    ChainStep::Xtc,
    ChainStep::Temperature,
];

impl Default for SamplerOrder {
    fn default() -> Self {
        let mut steps = [ChainStep::Temperature; SamplerName::ALL.len()];
        steps[..DEFAULT_STEPS.len()].copy_from_slice(&DEFAULT_STEPS);
        SamplerOrder {
            steps,
            len: DEFAULT_STEPS.len(),
        }
    }
}

impl SamplerOrder {
    /// The steps, in order.
    pub fn steps(&self) -> &[ChainStep] {
        &self.steps[..self.len]
    }

    /// Whether the penalties are part of this chain. `false` means the
    /// caller left `penalties` out, which llama.cpp reads as "do not
    /// penalise", so frink must not penalise either.
    pub fn has_penalties(&self) -> bool {
        self.steps().contains(&ChainStep::Penalties)
    }

    /// The canonical spellings of every sampler this engine implements,
    /// derived from the one table rather than restated.
    pub fn implemented_names() -> Vec<&'static str> {
        SamplerName::ALL
            .iter()
            .filter(|n| n.implemented().is_ok())
            .map(|n| n.as_str())
            .collect()
    }

    /// Parse a chain from already-split names.
    ///
    /// Each name is trimmed and lowercased first: a spelling is not a
    /// semantic, so `Top_K` cannot mean anything but `top_k` and
    /// refusing it would be a false refusal.
    ///
    /// Four things are refused, all BY NAME:
    ///
    /// * a name llama.cpp does not define either;
    /// * a real llama.cpp sampler frink does not implement;
    /// * the same sampler twice, which llama.cpp tolerates and which
    ///   here would silently mean "run it once" for the idempotent
    ///   filters and "square it" for temperature;
    /// * `penalties` anywhere but first -- see
    ///   [`SamplerOrderError::PenaltiesNotFirst`];
    /// * a chain with no `temperature` -- see
    ///   [`SamplerOrderError::TemperatureMissing`].
    pub fn from_names<I, S>(names: I) -> Result<Self, SamplerOrderError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut steps = [ChainStep::Temperature; SamplerName::ALL.len()];
        let mut len = 0usize;
        for raw in names {
            let name = raw.as_ref().trim().to_ascii_lowercase();
            let parsed = SamplerName::from_name(&name)
                .ok_or_else(|| SamplerOrderError::Unknown(raw.as_ref().trim().to_string()))?;
            let step = parsed
                .implemented()
                .map_err(|reason| SamplerOrderError::Unimplemented {
                    name: parsed.as_str(),
                    reason,
                })?;
            if steps[..len].contains(&step) {
                return Err(SamplerOrderError::Duplicate(parsed.as_str()));
            }
            if step == ChainStep::Penalties && len > 0 {
                return Err(SamplerOrderError::PenaltiesNotFirst);
            }
            // `len` cannot reach the capacity: the array is as long as
            // the whole name table and duplicates are refused above.
            steps[len] = step;
            len += 1;
        }
        if len == 0 {
            return Err(SamplerOrderError::Empty);
        }
        if !steps[..len].contains(&ChainStep::Temperature) {
            return Err(SamplerOrderError::TemperatureMissing);
        }
        Ok(SamplerOrder { steps, len })
    }
}

impl SamplerOrder {
    /// llama.cpp's `--sampler-seq`: one character per sampler
    /// (`edskypmxt`), the same chain `--samplers` spells by name.
    ///
    /// An unknown character is an ERROR here where upstream logs a
    /// warning and drops it, because a chain silently missing the step
    /// the caller typed is the failure this module exists to prevent.
    pub fn from_chars(chars: &str) -> Result<Self, SamplerOrderError> {
        let names = chars
            .trim()
            .chars()
            .map(|c| {
                SamplerName::from_chr(c)
                    .map(SamplerName::as_str)
                    .ok_or_else(|| SamplerOrderError::Unknown(c.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        SamplerOrder::from_names(names)
    }
}

/// `;`-separated, exactly as llama.cpp's `--samplers` takes it.
impl FromStr for SamplerOrder {
    type Err = SamplerOrderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.trim().is_empty() {
            return Err(SamplerOrderError::Empty);
        }
        SamplerOrder::from_names(s.split(';'))
    }
}

/// Round-trips through [`FromStr`], so the CLI can print the default it
/// applied and have that string mean the same chain.
impl fmt::Display for SamplerOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for step in self.steps() {
            if !first {
                f.write_str(";")?;
            }
            first = false;
            f.write_str(step.name().as_str())?;
        }
        Ok(())
    }
}

/// Equality and hashing are over the LIVE steps only. The backing array
/// is fixed-capacity, so the slots past `len` hold filler that two equal
/// chains need not agree about -- and the response cache keys on this,
/// where a spurious inequality would just miss and a spurious equality
/// would serve one caller's answer to another.
impl PartialEq for SamplerOrder {
    fn eq(&self, other: &Self) -> bool {
        self.steps() == other.steps()
    }
}

impl Eq for SamplerOrder {}

impl std::hash::Hash for SamplerOrder {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.steps().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default chain is llama.cpp's default chain, spelled out.
    ///
    /// The distribution-level proof that adding the four new steps
    /// changed nothing is
    /// `sampling::tests::the_default_order_is_the_chain_frink_already_ran`;
    /// this is the cheap statement of the order itself, so a reordering
    /// of `DEFAULT_STEPS` is visible in one line of diff.
    #[test]
    fn the_default_chain_is_llama_cpps_default_chain() {
        assert_eq!(
            SamplerOrder::default().to_string(),
            "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
        );
        assert!(SamplerOrder::default().has_penalties());
    }

    /// Every step the name table can produce names itself back, and
    /// every step [`ChainStep::all`] lists is one the table produces.
    ///
    /// This is the closing half of the "one table" guarantee. The
    /// compiler already forces [`ChainStep::name`] to cover every
    /// variant and [`SamplerName::implemented`] to cover every name; what
    /// neither forces is that a step given a name is a name that maps
    /// BACK to it. A `ChainStep::Xtc => SamplerName::TopK` typo compiles,
    /// and would make `xtc` build a chain running top-k twice.
    #[test]
    fn every_step_the_name_table_yields_names_itself_back() {
        let steps = ChainStep::all();
        assert!(!steps.is_empty());
        for step in &steps {
            assert_eq!(
                step.name().implemented(),
                Ok(*step),
                "{step:?} is spelled `{}`, which maps to a different step",
                step.name().as_str()
            );
        }
        // And no two steps share a spelling, which would make the
        // round-trip above pass while one step was unreachable.
        let mut names: Vec<&str> = steps.iter().map(|s| s.name().as_str()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "two steps share a name");
    }

    /// Every step this engine implements is in the DEFAULT chain, and in
    /// llama.cpp's order.
    ///
    /// A step that were implemented, parseable and left out of the
    /// default would run only for callers who named it explicitly --
    /// which is llama.cpp's shape for `mirostat` and NOT for anything in
    /// its default chain, so it would be a silent divergence from
    /// upstream for every unconfigured run.
    #[test]
    fn the_default_chain_contains_every_step_this_engine_implements() {
        let mut implemented = ChainStep::all();
        let mut defaulted = SamplerOrder::default().steps().to_vec();
        assert_eq!(
            defaulted, implemented,
            "the default chain and the implemented set must be the same steps \
             in the same order"
        );
        implemented.sort_by_key(|s| s.name().as_str());
        defaulted.sort_by_key(|s| s.name().as_str());
        assert_eq!(defaulted, implemented);
    }

    /// Every name in the generated table parses back to itself, which is
    /// what makes `ALL`, `as_str` and `from_name` one table rather than
    /// three that must agree.
    #[test]
    fn every_name_in_the_table_parses_back_to_itself() {
        for &name in SamplerName::ALL {
            assert_eq!(
                SamplerName::from_name(name.as_str()),
                Some(name),
                "{name:?} does not round-trip through its own spelling"
            );
        }
        // And no two variants share a spelling, which would make the
        // round-trip above pass while one name was unreachable.
        let mut spellings: Vec<&str> = SamplerName::ALL.iter().map(|n| n.as_str()).collect();
        spellings.sort_unstable();
        let before = spellings.len();
        spellings.dedup();
        assert_eq!(before, spellings.len(), "two names share a spelling");
    }

    /// A step and its name agree in both directions, so the applier's
    /// `ChainStep` and the caller's `SamplerName` cannot drift.
    #[test]
    fn every_implemented_name_round_trips_through_its_step() {
        for &name in SamplerName::ALL {
            if let Ok(step) = name.implemented() {
                assert_eq!(step.name(), name, "{name:?} maps to a step named otherwise");
            }
        }
    }

    /// llama.cpp's own aliases parse, so a command line that works
    /// upstream works here.
    #[test]
    fn llama_cpp_aliases_parse_to_the_canonical_name() {
        for (alias, expected) in [
            ("top-k", SamplerName::TopK),
            ("top-p", SamplerName::TopP),
            ("nucleus", SamplerName::TopP),
            ("min-p", SamplerName::MinP),
            ("temp", SamplerName::Temperature),
            ("typical", SamplerName::TypP),
        ] {
            assert_eq!(SamplerName::from_name(alias), Some(expected), "{alias}");
        }
        // Case and surrounding whitespace are spelling, not meaning.
        assert_eq!(
            " Top_K ; TEMPERATURE ".parse::<SamplerOrder>().unwrap(),
            SamplerOrder::from_names(["top_k", "temperature"]).unwrap()
        );
    }

    /// A name nobody defines is refused, and the refusal REPEATS THE
    /// NAME. A caller who typed `top_kk` must not have to guess which of
    /// their five names this engine disliked.
    #[test]
    fn an_unknown_sampler_is_refused_by_name() {
        let err = "top_k;top_kk;temperature"
            .parse::<SamplerOrder>()
            .expect_err("top_kk is not a sampler");
        assert_eq!(err, SamplerOrderError::Unknown("top_kk".to_string()));
        assert!(err.to_string().contains("top_kk"), "{err}");
    }

    /// The samplers llama.cpp has and frink does not are refused BY
    /// NAME, with the reason, rather than dropped from the chain.
    ///
    /// A caller who asked for `mirostat` and was handed a chain without
    /// it was given a different sampler and served a 200. This is the
    /// whole reason those names are in the table at all.
    #[test]
    fn a_real_but_unimplemented_sampler_is_refused_with_its_reason() {
        for name in ["mirostat", "infill"] {
            let err = format!("top_k;{name};temperature")
                .parse::<SamplerOrder>()
                .expect_err(&format!("`{name}` must be refused, not skipped"));
            assert!(
                matches!(err, SamplerOrderError::Unimplemented { name: n, .. } if n == name),
                "`{name}` was refused as {err:?}, which does not name it as a real \
                 llama.cpp sampler frink lacks"
            );
        }
        // Every name the table calls unimplemented is refused, and every
        // name it calls implemented is accepted -- so the table and the
        // parser cannot disagree about which half a name is in.
        for &name in SamplerName::ALL {
            // `temperature` is required in every chain, so a one-name
            // probe for anything else has to carry it.
            let probe: Vec<&str> = if name == SamplerName::Temperature {
                vec!["temperature"]
            } else {
                vec![name.as_str(), "temperature"]
            };
            let accepted = SamplerOrder::from_names(probe).is_ok();
            assert_eq!(
                accepted,
                name.implemented().is_ok(),
                "{name:?}: the parser and `implemented()` disagree"
            );
        }
    }

    /// Same as above, read off the error rather than the panic, so the
    /// MESSAGE is asserted and not just the failure.
    #[test]
    fn an_unimplemented_sampler_names_itself_and_says_what_is_implemented() {
        let err = "top_k;mirostat"
            .parse::<SamplerOrder>()
            .expect_err("no mirostat");
        assert_eq!(
            err,
            SamplerOrderError::Unimplemented {
                name: "mirostat",
                reason: SamplerName::Mirostat.implemented().unwrap_err(),
            }
        );
        let message = err.to_string();
        assert!(message.contains("`mirostat`"), "{message}");
        assert!(
            message.contains("top_k"),
            "must list what IS there: {message}"
        );
        // Unknown and unimplemented are different verdicts: one says
        // "no such sampler", the other "that sampler exists and frink
        // does not have it". Collapsing them would tell a caller their
        // valid llama.cpp flag was a typo.
        assert!(!message.contains("unknown"), "{message}");
    }

    #[test]
    fn the_same_sampler_twice_is_refused() {
        assert_eq!(
            "top_k;top_p;top_k"
                .parse::<SamplerOrder>()
                .expect_err("dup"),
            SamplerOrderError::Duplicate("top_k")
        );
        // An alias is the same sampler.
        assert_eq!(
            "top-k;top_k".parse::<SamplerOrder>().expect_err("dup"),
            SamplerOrderError::Duplicate("top_k")
        );
    }

    /// `penalties` after a truncation filter would penalise a candidate
    /// set that had already been cut, which is not what frink runs, so
    /// it is refused instead of quietly re-interpreted.
    #[test]
    fn penalties_anywhere_but_first_is_refused() {
        assert_eq!(
            "top_k;penalties".parse::<SamplerOrder>().expect_err("late"),
            SamplerOrderError::PenaltiesNotFirst
        );
        assert!("penalties;top_k;temperature"
            .parse::<SamplerOrder>()
            .is_ok());
        // Absent is fine and means "do not penalise".
        let no_penalties = "top_k;temperature".parse::<SamplerOrder>().unwrap();
        assert!(!no_penalties.has_penalties());
    }

    #[test]
    fn an_empty_chain_is_refused_rather_than_read_as_the_default() {
        assert_eq!(
            "".parse::<SamplerOrder>().expect_err("empty"),
            SamplerOrderError::Empty
        );
        assert_eq!(
            "   ".parse::<SamplerOrder>().expect_err("blank"),
            SamplerOrderError::Empty
        );
        // A stray separator is an empty NAME, which is unknown rather
        // than an empty chain.
        assert_eq!(
            "top_k;;top_p".parse::<SamplerOrder>().expect_err("stray"),
            SamplerOrderError::Unknown(String::new())
        );
    }

    /// Equality ignores the fixed-capacity array's filler, or the
    /// response cache would treat two identical chains as different keys
    /// depending on how they were built.
    #[test]
    fn two_chains_with_the_same_steps_are_equal_and_hash_alike() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = "top_k;temperature".parse::<SamplerOrder>().unwrap();
        let b = SamplerOrder::from_names(["top-k", "temp"]).unwrap();
        assert_eq!(a, b);
        let hash = |o: &SamplerOrder| {
            let mut h = DefaultHasher::new();
            o.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&a), hash(&b));
        // And a reordering is NOT equal, or the cache key would serve
        // one chain's answer for another's request.
        let reversed = "temperature;top_k".parse::<SamplerOrder>().unwrap();
        assert_ne!(a, reversed);
        assert_ne!(hash(&a), hash(&reversed));
    }

    /// The default round-trips through its own printed form, so the
    /// string a CLI banner shows can be pasted back into `--samplers`.
    #[test]
    fn the_printed_chain_parses_back_to_the_same_chain() {
        let default = SamplerOrder::default();
        assert_eq!(
            default.to_string().parse::<SamplerOrder>().unwrap(),
            default
        );
    }

    /// `--sampler-seq` used to be an alias of `--samplers`, so llama.cpp's
    /// own default `edskypmxt` was refused as one unknown sampler name.
    /// The characters are `common_sampler_type_to_chr`'s.
    #[test]
    fn sampler_seq_characters_spell_the_same_chain_as_the_names() {
        assert_eq!(
            SamplerOrder::from_chars("ekpmt").unwrap(),
            "penalties;top_k;top_p;min_p;temperature".parse().unwrap()
        );
        assert!(matches!(
            SamplerOrder::from_chars("kq"),
            Err(SamplerOrderError::Unknown(c)) if c == "q"
        ));
        assert!(
            SamplerOrder::from_chars("kdt").is_ok(),
            "dry is implemented"
        );
        assert!(
            SamplerOrder::from_chars("kit").is_err(),
            "infill is refused by name"
        );
    }
}
