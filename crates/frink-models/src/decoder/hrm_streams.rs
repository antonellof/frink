//! HRM-Text's two residual streams as the layer-walking bodies see them
//! (`crate::hrm`).

use super::Decoder;

impl Decoder {
    /// HRM-Text's two streams for a decode of `rows` rows, or `None`
    /// for every other architecture (`crate::hrm`).
    ///
    /// Called by every body that walks layers, so the state is built
    /// in one place from one rule; a body that forgot it would run the
    /// stacks on a single stream and answer fluently.
    pub(crate) fn hrm_streams(&self, embedded: &[f32]) -> Option<crate::hrm::HrmStreams> {
        let init = self.hrm_z_l_init.as_ref()?;
        crate::hrm::hrm_schedule(self.config.layer_loops)?;
        Some(crate::hrm::HrmStreams::new(embedded, init))
    }

    /// The residual layer `l` reads: `zH + zL` at a stack boundary,
    /// and whatever the previous layer produced everywhere else.
    pub(crate) fn hrm_stack_input(
        &self,
        streams: Option<&crate::hrm::HrmStreams>,
        l: usize,
        hidden: &mut Vec<f32>,
    ) {
        let Some(streams) = streams else { return };
        let Some(loops) = self.config.layer_loops else {
            return;
        };
        if loops.stack_starts_at(l) {
            *hidden = streams.stack_input();
        }
    }

    /// Store a finished stack's output in the stream it writes.
    ///
    /// Runs AFTER the FFN body's weightless pass norm
    /// (`Decoder::apply_loop_norm`), because `hrm-text.cpp:162` norms
    /// inside `build_stack` and `:186,193` store what it returned.
    pub(crate) fn hrm_store(
        &self,
        streams: Option<&mut crate::hrm::HrmStreams>,
        l: usize,
        hidden: &[f32],
    ) {
        let Some(streams) = streams else { return };
        let Some(loops) = self.config.layer_loops else {
            return;
        };
        if let Some(stream) = loops.stream_after(l) {
            streams.store(stream, hidden);
        }
    }
}
