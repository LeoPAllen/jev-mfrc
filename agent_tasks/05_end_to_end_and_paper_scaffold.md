# Task 05 — end-to-end and paper scaffold

Goal: leave a compact research package that a human can understand, run, and write from.

Focus on the full repository, but prefer deletion/simplification over new machinery.

Check:
- fresh setup instructions are correct;
- `./continue_build.sh` is the only build-loop command a user needs;
- `./run.sh --stage data|dev|test|sensitivity|analyze|all` is coherent and resumable;
- `all` advances one scientific phase per invocation and stops at the human instrument gate;
- generated results and paper outline use measured, non-causal language;
- literature/doc claims are sourced or explicitly marked as needing a source;
- no leftover references require deleted harness scripts or obsolete provenance gates.

Run the full tests, shell syntax, documentation check, and a synthetic no-network analysis smoke test if practical. Append a short note to `BUILD_LOG.md`, then stop.
