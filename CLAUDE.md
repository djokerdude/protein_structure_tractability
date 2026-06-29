# CLAUDE.md

Context and working conventions for this repo. Read this before making changes.

## What this is

`tractable` assesses how amenable a protein is to experimental structure
determination. Input a protein name → resolve it → pull experimental structures
(RCSB/SIFTS), AlphaFold confidence, and UniProt annotations → compute coverage
and a transparent score → return a typed `TractabilityReport` with model-written
reasoning and a recommended experimental strategy.

It is an **LLM agent over deterministic tools**, built for a role focused on
agentic biomedical data acquisition, ETL, and QC.

## The one rule that governs the design

**The LLM never invents numbers.** Every numeric field in the report
(coverage %, domain solved/unsolved, score) is produced by pure code in
`compute.py` / `scoring.py` from real database records. The model's only
writable fields are `reasoning` and `recommended_strategy`, and it must ground
them strictly in the already-computed facts it is handed.

When implementing the agent, pass it the *computed facts*, never the raw API
blobs to summarize into numbers.

## Layout

```
src/tractable/
  schema.py     typed report model — the contract; conform to it
  compute.py    deterministic residue geometry (DONE, tested)
  scoring.py    transparent additive rubric (DONE, tested)
  tools/        one cached, provenance-stamped fn per data source (interfaces DONE)
  qc.py         cross-source anomaly detection                    (TODO)
  agent.py      Anthropic tool-use loop + entity resolution       (TODO)
  render.py     report → markdown/text                            (TODO)
  cli.py        `tractable "BRCA1"` entrypoint                     (TODO)
tests/          offline unit tests (no network)
```

## Conventions

- Python ≥ 3.10, typed throughout. `pydantic` v2 for models.
- **Tools must be cached on disk** (keyed by request) so tests run offline and we
  respect each service's rate limits / usage policy. Commit representative
  cached responses under `tests/fixtures/`.
- **Every datum carries `Provenance`** (source, identifier, retrieved_at).
- Pure logic stays free of network/LLM/pydantic imports so it's unit-testable in
  isolation (see `compute.py` — operates on plain `(start, end)` tuples).
- Hand-roll the agent tool-use loop against the Anthropic SDK; do not pull in a
  framework. The orchestration and prompts should stay inspectable.
- Keep `reasoning`/`recommended_strategy` as the *only* LLM-authored fields.

## Run

```bash
pip install -e ".[dev]"
pytest                 # must stay green and offline
ruff check . && mypy src
```

## Immediate next tasks (in suggested order)

1. **`qc.py`** — cross-source consistency checks (e.g. UniProt vs NCBI length
   mismatch → `QCFlag`). Pure logic over tool outputs; fully testable offline.
   Highest-value next piece and the most direct match to the target role.
2. **Tool implementations** — `search_uniprot`, `get_uniprot_entry`,
   `get_pdb_structures`, `get_sifts_coverage`, `get_alphafold_plddt`,
   `get_ncbi_record`, each with on-disk caching + provenance. Save real
   responses as fixtures as you go.
3. **`agent.py`** — Anthropic tool-use loop: entity resolution
   (disambiguate organism/isoform, ask the user when genuinely ambiguous),
   then orchestrate acquisition, then hand computed facts to the model for
   `reasoning` + `recommended_strategy`.
4. **`render.py` + `cli.py`** — report → the text format shown in the README.
5. **Rubric calibration** — assemble a small labelled set of known-tractable vs.
   known-intractable targets; tune the weights in `scoring.py`; bump
   `RUBRIC_VERSION`.

## Watch out for

- SIFTS is the source of truth for residue-level coverage. Do not infer coverage
  from PDB entry titles/metadata.
- UniProt numbering is 1-indexed inclusive; keep ranges consistent end-to-end.
- Don't let `agent.py` compute or restate numbers — it routes and narrates only.
