# Dataset acquisition

Download the official MS COCO 2014 validation images and caption annotations from
https://cocodataset.org/#download. Extract `val2014.zip` to `data/val2014/` and
`annotations_trainval2014.zip` to `data/annotations/` so that
`data/annotations/captions_val2014.json` exists.

The loader selects one caption per image, removes empty captions and captions
containing a CIFAR-10 class name as a standalone word, then samples the configured
number of unique image/caption pairs with a local seeded RNG. The same sampled
captions and real reference images are used for every condition. No new dataset is
collected. Set `coco.count: 0` and `generation.coco_count: 0` for runs without COCO;
FID and CLIPScore are then omitted.

For the proposal's classifier, get `resnet18.pt` from Huy Phan's
[PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10) pretrained
weights archive. This repository can stream the official archive and extract
the ResNet-18 state dict:

```bash
python scripts/download_classifier.py
```

The script uses the author's Google Drive mirror because the Box URL in the
author's README currently returns 404. The archive is about 979 MB.
