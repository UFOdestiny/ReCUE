"""ReCUE: judge-free, single-trace uncertainty for reasoning LLMs.

Two orthogonal views over one completed reasoning trace:
  ARC  (Answer Re-Commitment)      -- re-elicit a short answer at the completed
                                      prefix and measure agreement / likelihood /
                                      confidence relative to the returned answer.
  TUP  (Trace Uncertainty Profile) -- passive token-level uncertainty already
                                      available from the primary generation.

A shared logistic head fuses both into a correctness-ranking score. Correctness
labels come from a deterministic verifier (math_verify / exact-match) -- there is
no LLM judge anywhere in the pipeline. All paths are read from the repo-root .env.
"""
