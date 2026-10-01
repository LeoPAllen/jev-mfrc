# Literature anchors and engineering guidance

This file is a bounded source list for the MVP. Agents drafting the paper should not invent citations beyond it.

## Measurement and management / IS methods

1. MacKenzie, S. B., Podsakoff, P. M., & Podsakoff, N. P. (2011). Construct Measurement and Validation Procedures in MIS and Behavioral Research: Integrating New and Existing Techniques. *MIS Quarterly*, 35(2), 293–334. Core implication: define the construct and measurement model before treating observed indicators as valid measures.
2. Qiao, M., & Huang, K.-W. (2021). Correcting Misclassification Bias in Regression Models with Variables Generated via Data Mining. *Information Systems Research*, 32(2), 462–480. Core implication: error in machine-generated variables can distort downstream inference; classifier metrics and inferential quality are not identical objectives.
3. Carlson, N. A., & Burbano, V. (2026). The use of LLMs to annotate data in management research: Foundational guidelines and warnings. *Strategic Management Journal*, 47(3), 699–725. Core implication: prompt/model choices can materially alter labels and downstream conclusions; use human-coded validation, sensitivity analysis, and transparent documentation.
4. Son, M., & Lee, P. (2026). Can Generative Large Language Models Serve as Raters for Test Development? A Systematic Evaluation Across Tasks, Models, and Inference Configurations. *Organizational Research Methods*, OnlineFirst. Core implication: treat generative models as raters whose reliability/validity depend on task and inference configuration.

## Disagreement / soft labels

5. Mostafazadeh Davani, A., Díaz, M., & Prabhakaran, V. (2022). Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations. *Transactions of the Association for Computational Linguistics*, 10, 92–110. Core implication: disagreement in subjective labels can contain systematic information lost by majority voting.
6. Fornaciari, T., Uma, A., Paun, S., Plank, B., Hovy, D., & Poesio, M. (2021). Beyond Black & White: Leveraging Annotator Disagreement via Soft-Label Multi-Task Learning. *NAACL*. Core implication: soft label distributions can preserve information discarded by hard aggregation.
7. Weerasooriya, T. C., et al. (2023). Disagreement Matters: Preserving Label Diversity by Jointly Modeling Item and Annotator Label Distributions with DisCo. *Findings of ACL*. Core implication: a single aggregated label can obscure minority judgments; observed label distributions are themselves meaningful objects.

## MFRC

8. Trager, J. P., et al. (2026). The Moral Foundations Reddit Corpus. *LREC-COLING 2026*, 6383–6407. The paper describes 16,123 comments, revised six-foundation coding, trained annotators, confidence, and a deliberately morally enriched sampling design. The public Hugging Face release must still be audited because its current row/annotator/source counts do not perfectly match publication prose.
9. Moral Foundations Coding Guide-2, appendix to the MFRC work. The six domains are Care, Equality, Proportionality, Loyalty, Authority, and Purity; annotation asks whether text communicates a moral concern/belief/attitude/emotion in each domain, with Thin Morality handled separately.

## Engineering / agent workflow guidance

10. OpenAI (2026), “Harness engineering: leveraging Codex in an agent-first world.” Practical implication: work depth-first in small building blocks; keep repository knowledge as the system of record; use a short `AGENTS.md` as a map; enforce important constraints mechanically rather than only in prose.
11. OpenAI, “How OpenAI uses Codex.” Practical implication: issue-like prompts with file paths and acceptance criteria work well; use a lightweight task queue and persistent repository instructions.
12. Simon Willison (2026), “First run the tests,” *Agentic Engineering Patterns*. Practical implication: existing executable tests orient coding agents and provide essential evidence that generated code actually runs.
13. GitHub (2025–2026), coding-agent guidance including WRAP. Practical implication: atomic tasks with crisp acceptance criteria outperform broad mixed requests, especially for lower-cost models.
14. Cookiecutter Data Science, “Opinions.” Practical implication: immutable raw data, reproducible source-code pipelines, cached expensive intermediates, environment capture, and tests are appropriate defaults for small research software projects.

15. OpenAI GPT-6 Luna model documentation (current 2026). Engineering implication: `gpt-6-luna` is the focused/high-volume GPT-6 model and the model API supports reasoning effort through `max`. This scaffold requests Luna + max explicitly for each bounded build task.
16. OpenAI Codex automation/configuration guidance (current 2026). Engineering implication: `codex exec` is intended for scripted runs; workspace-write plus a noninteractive approval policy is an appropriate bounded coding setup. Keep the harness simple and let a run fail visibly if the local client/account cannot honor the requested model or effort.
17. OpenAI/community Codex reports (2026) document rapidly changing reasoning-effort behavior across client versions. Engineering implication: do not build scientific provenance around the coding agent's token-level execution details. Scientific reproducibility belongs in the data, instrument, provider outputs, and analysis decisions; coding-agent quality is enforced by scoped tasks and executable tests.
18. TypeSafe System One OpenAPI (current 2026). Engineering implication: `GET /v1/models` returns names/aliases accepted in the request `model` field; Noul is the probability of yes/true; and a response `model` may differ from the requested alias. The contract does not say that the response-model string is itself requestable, so record it for provenance/drift detection rather than resubmitting it as an undocumented pin.

## Framing discipline

The paper's durable contribution is not “JEV is good.” It is a test of whether native probabilistic AI judgments preserve measurement information that hard-label workflows discard, and whether uncertainty can prioritize human review. JEV is one implementation of that methodological opportunity.

## Verified source links

- MacKenzie et al. (2011): https://aisel.aisnet.org/misq/vol35/iss2/5/
- Qiao & Huang (2021): https://doi.org/10.1287/isre.2020.0977
- Carlson & Burbano (2026): https://doi.org/10.1002/smj.70023
- Son & Lee (2026): https://doi.org/10.1177/10944281261475596
- Trager et al. (2026): https://aclanthology.org/2026.lrec-1.507/
- OpenAI GPT-6 Luna: https://developers.openai.com/api/docs/models/gpt-6-luna
- TypeSafe System One API: https://api.typesafe.ai/docs
