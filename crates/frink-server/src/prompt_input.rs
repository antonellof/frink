//! What a generation prefills: text the tokenizer turns into ids, or
//! ids a route has already assembled.
//!
//! Every generating route hands the decode TEXT, and the decode
//! tokenizes it with `parse_special` and a BOS where the vocabulary adds
//! one (`tokenize_input_prompts(..., true, true)` upstream). llama.cpp's
//! `/infill` is the one route that cannot: its prompt is FIM control
//! tokens between pieces tokenized separately and WITHOUT
//! `parse_special` (`format_prompt_infill`), and no text re-tokenizes to
//! that sequence: a merge across a piece boundary, or a prefix that
//! happens to spell `<|fim_prefix|>`, would give different ids. So the
//! prompt is a parameter of the one decode path rather than a second
//! path that takes ids.

use frink_models::tokenizer::{prepend_bos, SpecialTokens};
use frink_models::TextTokenizer;

/// A generation's prompt.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Prompt<'a> {
    /// Text, tokenized by the decode with `parse_special` and a BOS
    /// where this vocabulary adds one.
    Text(&'a str),
    /// Ids prefilled exactly as given, BOS included or not: the route
    /// that assembled them already decided.
    Ids(&'a [usize]),
}

impl<'a> Prompt<'a> {
    /// The ids the decode prefills.
    pub(crate) fn tokens<T: TextTokenizer + ?Sized>(
        self,
        tokenizer: &T,
        bos_id: Option<usize>,
    ) -> Vec<usize> {
        match self {
            Prompt::Text(text) => {
                let mut tokens = tokenizer.encode(text, SpecialTokens::Parse);
                prepend_bos(&mut tokens, bos_id);
                tokens
            }
            Prompt::Ids(ids) => ids.to_vec(),
        }
    }

    /// The rendered text, for the two checks that read it: whether the
    /// prompt already opened a reasoning block, which arms a reasoning
    /// budget and counts reasoning tokens.
    ///
    /// Empty for an id prompt. That is not a guess about what the ids
    /// say: `run_generation_emit` refuses an id prompt that carries a
    /// reasoning format or budget, so neither check is reached with one.
    pub(crate) fn text(self) -> &'a str {
        match self {
            Prompt::Text(text) => text,
            Prompt::Ids(_) => "",
        }
    }

    pub(crate) fn is_ids(self) -> bool {
        matches!(self, Prompt::Ids(_))
    }
}

impl<'a> From<&'a str> for Prompt<'a> {
    fn from(text: &'a str) -> Self {
        Prompt::Text(text)
    }
}

impl<'a> From<&'a String> for Prompt<'a> {
    fn from(text: &'a String) -> Self {
        Prompt::Text(text)
    }
}

impl<'a> From<&'a Vec<usize>> for Prompt<'a> {
    fn from(ids: &'a Vec<usize>) -> Self {
        Prompt::Ids(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ServerTokenizer;

    /// Text is tokenized and given its BOS. Ids are taken as they are:
    /// the route that assembled them decided whether a BOS belongs.
    ///
    /// Sabotage: prepend the BOS to an id prompt too; this goes red.
    #[test]
    fn text_gets_a_bos_and_ids_are_prefilled_verbatim() {
        let tok = ServerTokenizer::Byte;
        assert_eq!(Prompt::Text("ab").tokens(&tok, Some(1)), vec![1, 97, 98]);
        let ids = vec![200, 97];
        assert_eq!(Prompt::from(&ids).tokens(&tok, Some(1)), vec![200, 97]);
        assert!(Prompt::from(&ids).is_ids());
        assert_eq!(Prompt::from(&ids).text(), "");
    }
}
