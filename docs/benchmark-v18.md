# Development benchmark v1.8

Version 1.8 is a frozen development configuration, not a production-certified
benchmark. The corpus and lock are local artifacts and are not supplied in a
public clone. Historical scoring versions retain their existing rules.

## Version contract

- The frozen development scenes use transcripts decoded from video audio,
  with the model revision and files identified in each observation artifact.
  Transcripts are not derived from answer labels.
- The 1.8 scorer inherits 1.7 anti-flooding, temporal and public-identity corrections.
  Unknown/missing speaker labels no longer invalidate otherwise correct dialogue:
  speaker identity is not a scored dimension. Text, time and duplicate penalties
  remain unchanged. Weights, thresholds and costs are unchanged.
- Frozen development evaluations use an explicit identity for code, input media,
  labels, transcript artifacts and runtime versions. The historical development
  budget is 100k visual units, 120 audio seconds and 20k transcript characters.
  Private corpus locks and campaign tooling are not part of the public package.

## Evidence ceiling

This is a fixed engineering benchmark, not an independently certified corpus.
These sources are already exposed; native text/audio/cuts may be incompletely
annotated. Report family scores and examples with this limitation. Never suppress
correct native observations in the miner simply to fit incomplete labels, add
scene-ID rules, access truth from miner code, or optimize against reserved final
answers. The independent corpus reservation remains untouched.

Retain attempts, failures, costs and source revisions when comparing candidates.
A development improvement does not establish generalization or public reward
safety. Follow [independent evaluation requirements](evaluation-v2.md) for those
claims. Use the [local quickstart](workflows.md) for a public-clone smoke test.
