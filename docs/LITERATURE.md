# Literature anchors and source notes

These sources bound the paper scaffold. Each research claim below links to its publisher or archival source. Verify wording against the full text before using it in the manuscript; mark any additional claim "source needed" until it is supported.

## Measurement and management / IS methods

1. MacKenzie, S. B., Podsakoff, P. M., & Podsakoff, N. P. (2011). Construct Measurement and Validation Procedures in MIS and Behavioral Research: Integrating New and Existing Techniques. *MIS Quarterly*, 35(2), 293–334. Establishes construct-definition, measurement-model, and validation procedures. [MIS Quarterly article](https://aisel.aisnet.org/misq/vol35/iss2/5/)
2. Qiao, M., & Huang, K.-W. (2021). Correcting Misclassification Bias in Regression Models with Variables Generated via Data Mining. *Information Systems Research*, 32(2), 462–480. Examines how generated-variable misclassification affects downstream regression inference. [INFORMS DOI](https://doi.org/10.1287/isre.2020.0977)
3. Carlson, N. A., & Burbano, V. (2026). The Use of LLMs to Annotate Data in Management Research: Foundational Guidelines and Warnings. *Strategic Management Journal*, 47(3), 699–725. Discusses annotation design, validation, and sensitivity to implementation choices. [Wiley DOI](https://doi.org/10.1002/smj.70023)
4. Son, M., & Lee, P. (2026). Can Generative Large Language Models Serve as Raters for Test Development? A Systematic Evaluation Across Tasks, Models, and Inference Configurations. *Organizational Research Methods*, OnlineFirst. Evaluates reliability and validity across rating tasks, models, prompts, and inference settings. [SAGE DOI](https://doi.org/10.1177/10944281261475596)

## Disagreement and soft labels

5. Mostafazadeh Davani, A., Díaz, M., & Prabhakaran, V. (2022). Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations. *Transactions of the Association for Computational Linguistics*, 10, 92–110. Studies multi-annotator approaches and the information in subjective disagreement. [ACL Anthology](https://aclanthology.org/2022.tacl-1.6/)
6. Fornaciari, T., Uma, A., Paun, S., Plank, B., Hovy, D., & Poesio, M. (2021). Beyond Black & White: Leveraging Annotator Disagreement via Soft-Label Multi-Task Learning. In *Proceedings of NAACL-HLT 2021*, 2591–2597. Evaluates soft-label learning as a way to retain annotator-distribution information. [ACL Anthology](https://aclanthology.org/2021.naacl-main.204/)
7. Weerasooriya, T. C., Ororbia, A. G., Bhensadadia, R. B., KhudaBukhsh, A. R., & Homan, C. M. (2023). Disagreement Matters: Preserving Label Diversity by Jointly Modeling Item and Annotator Label Distributions with DisCo. *Findings of ACL 2023*, 4679–4695. Models item and annotator label distributions. [ACL Anthology](https://aclanthology.org/2023.findings-acl.287/)

## MFRC

8. Trager, J. P., et al. (2026). The Moral Foundations Reddit Corpus. *Proceedings of LREC 2026*, 6383–6407. Describes MFRC and the Moral Foundations Coding Guide-2. [LREC proceedings paper](https://lrec.elra.info/lrec2026-main-507)
9. The MFRC paper's appendix contains the Coding Guide-2 used to define the six focal domains: Care, Equality, Proportionality, Loyalty, Authority, and Purity. The guide asks whether text communicates a moral concern, belief, attitude, or emotion; Thin Morality is outside this study. Check the appendix wording before quoting it. [MFRC paper and appendix](https://lrec.elra.info/lrec2026-main-507)

The paper describes comments and subreddit counts, while the public `train_dedup` release is an annotation-row table. At the time of this review (2026-10-01), its dataset card reported 53,827 rows and 11 subreddit values; audit and cite the exact downloaded revision because release metadata can change. [MFRC paper](https://lrec.elra.info/lrec2026-main-507), [Hugging Face dataset card](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)

## Engineering / agent workflow guidance

This is repository guidance, not manuscript literature. Workflow decisions are described in `docs/ENGINEERING.md`. For the JEV response and model-list contract used by the client, consult the [TypeSafe System One API documentation](https://api.typesafe.ai/docs).

## Framing discipline

The paper tests whether JEV's native probabilities preserve information relative to prespecified hard-label and sample-cell summaries. It does not establish that JEV is generally valid. Human votes are trained-rater reference judgments; the design does not identify why raters disagree. The review simulation is oracle/reference replacement and cannot establish labor savings. MFRC source-cell results concern this selected sample, not population prevalence or causal effects.
