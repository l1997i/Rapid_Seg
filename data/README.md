# Data

This directory holds the SemanticKITTI data (not tracked in git).

## Official layout (`--dataset full`)

Download [SemanticKITTI](http://semantic-kitti.org/dataset.html) and place it as:

```
data/semantickitti_full/
  sequences/
    00/ velodyne/*.bin     # float32 [x, y, z, remission]
        labels/*.label     # uint32  (sem & 0xFFFF) | (inst << 16)
    01/ ...
    ...
```

Raw semantic ids are mapped to the 19 evaluation classes via the official
`learning_map`, using the official splits (train 00-07,09,10 · val 08 ·
test 11-21).

```bash
python scripts/train.py --dataset full --data_root data/semantickitti_full
```

## Preprocessed `.pth` subset (`--dataset sp`)

Place preprocessed frames as `data/semantickitti_sp/<seq>/*.pth` (each a dict
with `coord`, `remission`, `semantic_label`).

```bash
python scripts/train.py --dataset sp --data_root data/semantickitti_sp
```
