# Literature review: uncertainty-gated memory *write* admission

**Companion to** `../../TWO_STAGE_GEOMETRY_MARGIN_ENTROPY_PLAN.md` (§21). Distilled from the
2026-07-20 literature survey; every citation below is load-bearing for a specific design
decision in the plan, annotated with what transfers and what does not.

## 1. The write-gating gap (novelty claim)

All verified uncertainty-gated RAG systems gate **read**, not write:

| System | Mechanism | arXiv |
|---|---|---|
| FLARE | token-probability threshold triggers retrieval | 2305.06983 |
| DRAGIN | entropy × attention triggers retrieval | 2403.10081 |
| Self-RAG | learned reflection tokens | 2310.11511 |
| Adaptive-RAG | external query-complexity classifier | 2403.14403 |
| SeaKR | Gram-determinant of hidden states across 20 samples | 2406.19215 |

None gates *write*. Existing write policies are per-write **LLM judgments**:

- **Mem0** (2504.19413): ADD/UPDATE/DELETE/NOOP judge — we adopt its action taxonomy,
  implemented deterministically.
- **A-Mem** (2502.12110): neighbor-retrieval-at-write skeleton — structural ancestor of
  our Stage-1 neighborhood evaluation.
- **Generative Agents** (2304.03442): LLM-scored importance at write time.

SeaKR's "insert candidate, measure uncertainty change" is the nearest conceptual ancestor
of our provisional ΔH — ours is deterministic and retrieval-side, not sampled and
hidden-state-side. **The deterministic, calibrated, LLM-free write gate is the thesis's
novelty claim.** One further targeted search before the defense is pre-registered in
Phase 8.

## 2. Entropy caution

- **Yadkori et al.** (2406.02543): entropy conflates aleatoric answer multiplicity with
  epistemic error; margins disambiguate. This is the principled defense of the plan's
  *joint* margin–entropy rule, and exactly the observed `user_value_question` failure
  mode of the old Stage 2 (plan §4.2): a probe that legitimately has several good
  answers in the store reads as "uncertain".
- **Kuhn / Farquhar semantic entropy** (2302.09664; Nature 2024): motivates meaning-level
  uncertainty, but its sampling+NLI recipe is what our ablation campaign rejected
  (online NLI cost with 66 % neutral punts).
- **SEP** (2406.15927): legitimizes cheap deterministic proxies for semantic entropy.
- **AdaRAGUE** (2501.12835): simple uncertainty signals ≥ complex pipelines at scale.

## 3. Score-distribution statistics (QPP pedigree)

- **NQC** (Shtok et al., TOIS 2012) and successors: post-retrieval query-performance
  prediction from score-distribution statistics. Our `disp` (std/mean of top-k scores)
  and softmax entropy features are QPP statistics applied at *write* time; `n_eff_norm`
  borrows their background-score normalization idea for store-size comparability.

## 4. Abstention / selective prediction

- **Chow** (1970): cost-ratio rejection thresholds — basis for the Option-B threshold rule.
- **Geifman & El-Yaniv** (NeurIPS 2017): risk-coverage curves / SGR — the calibration
  ceremony (plan §13.3) and the justification metric for a v2 in which ABSTAIN suppresses.
- **Kamath et al.** (ACL 2020): calibrate selective prediction under category shift —
  our backend/scenario strata.

## 5. Interference and write-parsimony

- **Radovanović et al.** (JMLR 2010): hubness — a centroid-near vector crowds top-k lists;
  exactly what `ΔH_neighbor`/`churn` measure at write time. k-occurrence skew is reported
  as a diagnostic.
- Embedding-capacity limits (2508.21038) and more-documents-harm results (2503.04388):
  justify write-parsimony as a goal in itself.

## 6. Stage 0 machinery

- **Mu & Viswanath** (ICLR 2018): All-But-The-Top embedding post-processing — primary
  citation for the Stage-0 ABTT whitening.

## 7. Transfer discipline

Nothing is copied wholesale from read-time RAG. Every borrowed element is re-derived on
deterministic retrieval-side quantities: probes are template-generated from user text
(never sampled, never candidate-derived), entropy is over retrieval score distributions
(never token distributions), and every threshold is calibrated out-of-sample on harvest
data (plan §13).
