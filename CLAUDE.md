# Project Context — Rust + Embedded AI (Jetson Orin NX)

## Stack
- Language: Rust (idiomatic, safe — prefer `Result` over `unwrap`/`panic`)
- Target: Jetson Orin NX (ARM64, CUDA/TensorRT for inference)
- AI crates: `ort` (ONNX Runtime) for vision, `candle` for LLM/audio
- Build/test: `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt`

## Common commands
- Build: `cargo build`
- Test: `cargo test`
- Lint: `cargo clippy -- -D warnings`
- Format check: `cargo fmt --check`

---

## Team Workflow (the agents)

This project uses a three-role agent team. Work always flows through a Pull
Request — no agent commits directly to `main`.

```
rust-dev  ──>  rust-qa  ──>  rust-senior
(branch +     (test the     (review diff +
 PR)           PR)           merge or reject)
```

### Roles
1. **rust-dev** — implements the change on a feature branch, pushes, and opens a PR.
2. **rust-qa** — checks out the PR branch, runs the test/lint suite, posts findings on the PR. Read-only on code.
3. **rust-senior** — senior reviewer. Reads the diff and QA's findings, then either **merges** the PR or **rejects** it back to the dev. Only this agent is allowed to merge.

### Branch & PR conventions
- Branch names: `feat/<short-desc>`, `fix/<short-desc>`, `refactor/<short-desc>`
- One PR per logical change; keep them small and reviewable
- PR title: imperative mood (e.g. "Add YOLO ONNX inference to capture pipeline")
- PR body: what changed, why, and how it was tested

### Merge rules (enforced by rust-senior)
- Merge ONLY if QA reported a pass AND review finds no blocking issues
- Never merge with failing tests, failing clippy, or unresolved review comments
- Use squash merge and delete the branch: `gh pr merge <n> --squash --delete-branch`
- Never force-push to `main`; never merge your own un-reviewed work

### Standard invocation
```
Use the rust-dev subagent to <task>, then rust-qa to test the PR,
then rust-senior to review and merge.
```

> Prerequisite: the GitHub CLI (`gh`) must be installed and authenticated
> (`gh auth login`) so the agents can open, comment on, and merge PRs.