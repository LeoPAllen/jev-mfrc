# Luna build rules

This is a small research repository. Optimize for study quality, legibility, and correct results — not software-process ceremony.

1. Work on exactly the assigned task, then stop.
2. Read the task file first. Inspect only the files needed to solve it.
3. Prefer the smallest clear change. Do not introduce frameworks, services, databases, or dependencies unless the task truly requires them.
4. Run the task's stated checks before finishing. Add a focused regression test when you fix a real bug.
5. `docs/RESEARCH_SPEC.md` and `docs/METHODS.md` describe the intended study. If code and design conflict, surface the conflict rather than silently changing the scientific question.
6. Never use held-out outcomes to choose prompt wording, thresholds, exclusions, or claims.
7. Raw MFRC data are immutable after download. Derived files may be rebuilt freely from raw data.
8. Paid JEV calls belong in the deterministic runner, not ad-hoc scripts.
9. Do not read or print provider secrets. The scientific runner reads `TYPESAFE_API_KEY` from the shell environment.
10. Human judgment is required for the instrument approval gate and final manuscript claims.
11. Briefly append what you changed and any unresolved issue to `BUILD_LOG.md`, then stop.
12. Do not mark your own task complete in `state/tasks.json`; the wrapper does that only after focused checks pass.

Scientific guardrails: human annotations are a trained-rater reference distribution rather than ontological truth; JEV probability is a probability of the bounded yes/no judgment, not moral intensity; MFRC source-cell analyses describe this selected sample, not Reddit population prevalence.
