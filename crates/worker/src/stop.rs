//! Stop strings: generation ends before any of them, and they aren't
//! returned. Text that might be the start of a stop string is held back until
//! it's clear either way.

#[derive(Default)]
pub struct StopScanner {
    stops: Vec<String>,
    held: String,
}

impl StopScanner {
    pub fn new(stops: Vec<String>) -> Self {
        StopScanner {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
        }
    }

    /// Add generated text. Returns the text safe to emit, and whether a stop
    /// string was hit (generation should end; nothing after it is emitted).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        self.held.push_str(text);
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            let out = self.held[..at].to_string();
            self.held.clear();
            return (out, true);
        }
        // Keep the longest suffix that could still grow into a stop string.
        let keep = self
            .stops
            .iter()
            .flat_map(|s| {
                (1..s.len())
                    .filter(|&n| s.is_char_boundary(n))
                    .map(move |n| &s[..n])
            })
            .filter(|prefix| self.held.ends_with(prefix))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, false)
    }

    /// Whatever was held back, at the end of generation.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_and_holds_back() {
        let mut s = StopScanner::new(vec!["<|im_end|>".into(), "\n\nUser:".into()]);
        assert_eq!(s.push("Hello"), ("Hello".into(), false));
        assert_eq!(s.push(" world<|im"), (" world".into(), false));
        assert_eq!(s.push("_end|> trailing"), ("".into(), true));

        let mut s = StopScanner::new(vec!["<|im_end|>".into()]);
        assert_eq!(s.push("a<"), ("a".into(), false));
        assert_eq!(s.push("b"), ("<b".into(), false));
        assert_eq!(s.push("<|"), ("".into(), false));
        assert_eq!(s.finish(), "<|");

        let mut none = StopScanner::new(vec![]);
        assert_eq!(none.push("x"), ("x".into(), false));
    }

    #[test]
    fn earliest_stop_wins_and_multibyte_is_safe() {
        let mut s = StopScanner::new(vec!["B".into(), "AB".into(), "€€".into()]);
        assert_eq!(s.push("xxAB"), ("xx".into(), true));
        let mut s = StopScanner::new(vec!["€€".into()]);
        assert_eq!(s.push("a€"), ("a".into(), false));
        assert_eq!(s.push("€b"), ("".into(), true));
    }
}
