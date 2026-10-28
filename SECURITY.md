# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private
vulnerability reporting: open the
[Security tab of antonellof/frink](https://github.com/antonellof/frink/security)
and choose "Report a vulnerability". Do not open a public issue for a
security problem.

Include the frink version (`frink --version`), the command or request
that triggers the problem, and a file or steps to reproduce it if you
can. We will acknowledge the report, work on a fix in a private
advisory, and credit you in it unless you ask us not to.

## Supported versions

Only the latest release receives security fixes. All crates in the
workspace share one version.

## Scope

- `frink-server` authentication: `FRINK_API_KEY` is the bearer token
  required by every route except `/health`, including the `/admin`
  routes. A way to reach a protected route without it is in scope.
- TLS: `FRINK_TLS_CERT` / `FRINK_TLS_KEY`. Without them the server
  speaks plain HTTP.
- The bind check: the server refuses to start on a non-loopback address
  (or one it cannot confirm is loopback) when no API key is set.
  `FRINK_ALLOW_UNAUTHENTICATED_REMOTE=1` turns that check off, for
  deployments where a reverse proxy handles authentication. Running
  with it set exposes the API to anyone who can reach the address; that
  is the documented behaviour, not a vulnerability. A way past the
  check without the override is in scope.
- Parsing untrusted model files: GGUF and safetensors headers, tensor
  tables and tokenizer metadata. A crafted file that causes memory
  unsafety, an out-of-bounds read, or an unbounded allocation is in
  scope.

See [`docs/CONFIG.md`](docs/CONFIG.md) for the server's security
variables.
