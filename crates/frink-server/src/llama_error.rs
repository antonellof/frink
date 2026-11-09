//! llama-server's error body, for the routes that answer in its shape.
//!
//! `format_error_response` (`tools/server/server-common.cpp`) writes
//! `{"error": {"code", "message", "type"}}`, and the HTTP status is the
//! `code`. The kind names the status and the type together here, so a
//! route cannot pair upstream's `not_supported_error` with a 400.

use axum::http::StatusCode;
use axum::Json;

use crate::ApiError;

/// upstream's `error_type`, for the kinds these routes answer with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LlamaErrorKind {
    /// `ERROR_TYPE_INVALID_REQUEST`: 400.
    InvalidRequest,
    /// `ERROR_TYPE_NOT_SUPPORTED`: 501.
    NotSupported,
    /// `ERROR_TYPE_UNAVAILABLE`: 503.
    Unavailable,
}

impl LlamaErrorKind {
    fn status_and_type(self) -> (StatusCode, &'static str) {
        match self {
            LlamaErrorKind::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            LlamaErrorKind::NotSupported => (StatusCode::NOT_IMPLEMENTED, "not_supported_error"),
            LlamaErrorKind::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "unavailable_error"),
        }
    }
}

/// One error, in upstream's body.
pub(crate) fn llama_error(kind: LlamaErrorKind, message: impl Into<String>) -> ApiError {
    let (status, kind) = kind.status_and_type();
    (
        status,
        Json(serde_json::json!({"error": {
            "code": status.as_u16(),
            "message": message.into(),
            "type": kind,
        }})),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three kinds, each with upstream's status and type string.
    #[test]
    fn each_kind_carries_upstreams_code_and_type() {
        for (kind, code, ty) in [
            (LlamaErrorKind::InvalidRequest, 400, "invalid_request_error"),
            (LlamaErrorKind::NotSupported, 501, "not_supported_error"),
            (LlamaErrorKind::Unavailable, 503, "unavailable_error"),
        ] {
            let (status, body) = llama_error(kind, "m");
            assert_eq!(status.as_u16(), code);
            assert_eq!(body.0["error"]["code"], code);
            assert_eq!(body.0["error"]["type"], ty);
            assert_eq!(body.0["error"]["message"], "m");
        }
    }
}
