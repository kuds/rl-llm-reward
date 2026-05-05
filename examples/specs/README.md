# Hand-written reward specs

These are baseline reward specifications, written by hand. They serve
two purposes:

1. **Reproducibility without API spend.** Run them via
   `p2p train-spec --env <env> examples/specs/<file>.json` to reproduce
   a known baseline without hitting an LLM.
2. **A point of comparison for the post.** When the LLM produces a
   reward for "run forward", we can show the LLM's spec next to the
   hand-written baseline and discuss the differences.

## HalfCheetah

| File                      | Intended behavior                |
| ------------------------- | -------------------------------- |
| forward_locomotion.json   | Run forward as fast as possible  |
| backward_locomotion.json  | Run backward as fast as possible |
| stand_still.json          | Stand still, stay upright        |

```
p2p train-spec examples/specs/forward_locomotion.json --quick
```

## Hopper

| File                          | Intended behavior                  |
| ----------------------------- | ---------------------------------- |
| hopper_forward.json           | Hop forward without falling        |
| hopper_hop_in_place.json      | Hop in place as high as possible   |

```
p2p train-spec --env hopper examples/specs/hopper_forward.json --quick
```

## Ant

| File                          | Intended behavior                  |
| ----------------------------- | ---------------------------------- |
| ant_forward.json              | Walk forward steadily              |
| ant_stand.json                | Stand still, stay upright          |

```
p2p train-spec --env ant examples/specs/ant_forward.json --quick
```
