# RAG program, campaign C2 (CAPTURE × READ) — pre-registration

**Frozen 2026-08-08, BEFORE the campaign runs and BEFORE C1 unblinds.**
Launch of C2 is gated on C1: `read_verbatim` must beat **both** of its
controls on vector (point estimate) for the read half to be worth pairing;
if C1 fails, C2 is re-planned and this document is amended, not silently
replaced.

## Method under test — M2, entry-budget-aware packed capture

`QwenRagHandler` write side (`RAG_WRITE=pack`): buffer no-write prereq turn
texts; emit one `archival_memory_add` per ~2000-char packed blob. The
binding archival budget is ENTRY COUNT (50), not entry length — offline,
packing lifts vector carried-rate 0.748 → 0.884 and r@5 0.703 → 0.819 vs
one-turn-per-entry, while sentence chunking collapses to 0.342
(`gov_logs/hnav_rag/falsifiers.json`, m2 GO frozen before compute).

Fully causal: the buffer holds only past turns; the chain-final tail
(< one blob) is never written. The live carried-rate is therefore expected
slightly BELOW the offline 0.884; the offline→live gap is itself a
pre-registered quantity of interest.

## Arms (5 replicates, `memory_kv,memory_vector`, temperature 0.001)

| arm | env | purpose |
|---|---|---|
| `baseline` | — | reproduced fair baseline, co-run |
| `pack_write` | `RAG_WRITE=pack, RAG_MODE=none` | capture WITHOUT read: predicted ≈ 0 by the tier-conditional correction — this arm exists to test that prediction |
| `read_only` | `RAG_MODE=read_verbatim` | C1's winner re-run, the read half alone |
| `pack_read` | `RAG_WRITE=pack, RAG_MODE=read_verbatim` | **the method** |
| `pack_shuf_read` | `RAG_WRITE=pack_shuffled, RAG_MODE=read_verbatim` | mechanism control: identical write volume/count, scrambled word order — separates FACT capture from TEXT VOLUME (note: shuffling preserves the word multiset, so short verbatim golds may survive; this control also measures exactly that bag-of-words channel) |
| `per_turn_read` | `RAG_WRITE=per_turn, RAG_MODE=read_verbatim` | granularity contrast: retires alt5R by comparison (packing claim ≠ capture claim) |

## Hypotheses (frozen)

- **H-C2a (primary).** `pack_read` > `baseline` on vector accuracy, matched
  questions; CI excluding zero AND direction consistent in ≥ 4/5 replicates.
- **H-C2b (interaction).** ΔAcc(`pack_write` vs `baseline`) ≈ 0 AND
  ΔAcc(`pack_read` vs `read_only`) − ΔAcc(`pack_write` vs `baseline`) ≥
  +0.03 → capture and read are multiplicative, not additive.
- **H-C2c (mechanism).** `pack_read` > `pack_shuf_read`. If the shuffled
  control matches, the gain is text volume / bag-of-words presence, and must
  be reported as such.
- **H-C2d (granularity).** `pack_read` > `per_turn_read` on vector, in the
  direction of the offline +13.6pp carried-rate gap.
- **H-C2e (offline mechanism check).** `hnav_answer_index --validate`
  by-tier on the `pack_write`/`pack_read` final stores shows carried-rate
  near the offline prediction (0.88 minus tail loss) and, in `pack_read`,
  p(correct | archival-only) far above the baseline 0.000.

## Analysis (frozen)

As C1: `analyze_gov_replicates.py --reference-arm baseline` (+ secondary
reference `read_only` for the interaction), exact McNemar + cluster
bootstrap, Holm per family, student-included headline, per-replicate table,
strict answer-field regrade, step-budget guard, §22 cost accounting.
`RAG_TOP_K` stays 5. Nothing is tuned in this campaign; every knob is frozen
at its falsifier value (pack_len 2000, cap 50).
