# Hand-written reward specs

These are baseline reward specifications for the v0 prompt set, written
by hand. They serve two purposes:

1. **Reproducibility without API spend.** Run them via
   `p2p train-spec examples/specs/<file>.json` to reproduce a known
   baseline without hitting the LLM.
2. **A point of comparison for the post.** When the LLM produces a
   reward for "run forward", we can show the LLM's spec next to the
   hand-written baseline and discuss the differences.

| File                      | Intended behavior                |
| ------------------------- | -------------------------------- |
| forward_locomotion.json   | Run forward as fast as possible  |
| backward_locomotion.json  | Run backward as fast as possible |
| stand_still.json          | Stand still, stay upright        |

To train against one:

```
p2p train-spec examples/specs/forward_locomotion.json --quick
```
