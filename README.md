# The-NLP

ANLP Monsoon 2026 project of team deathmachine069: *Investigating Whether Inductive Backdoors Are a LoRA Artifact*.

- **[RESULTS.md](RESULTS.md): all experiments, outputs, findings and next steps**
- **[report/final_report.pdf](report/final_report.pdf): short report with every figure** ([LaTeX source](report/final_report.tex))
- [`projectmid/`](projectmid/): code, configs, data and results ([README](projectmid/README.md), [short overview](projectmid/idea.md))
- [`projectmid/results/final/`](projectmid/results/final/): complete experiment (rank sweep, all-linear placement sweep, optimizer ablation, SVD / Eckart-Young-Mirsky analysis), figures in `figures/`
- [`projectmid/results/pilots/`](projectmid/results/pilots/): pilot runs (recovered from Weights & Biases)
- `deathmachine069-Proposal.pdf`, `inductive_backdoors_lora_proposal.tex`: project proposal
- `subliminal_learning_litreview.tex`: literature review
- `jarvislabs-mech-interp-remote-gpu-guide.md`: GPU machine setup

Training data: [Betley et al. 2025](https://github.com/JCocola/weird-generalization-and-inductive-backdoors) (US PRESIDENTS), copied verbatim into `projectmid/data/`.
