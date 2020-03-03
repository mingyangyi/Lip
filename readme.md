## Repository Overview
| File | Description |
| ------ | ------ |
| `main.py` | Main train and test file |
| `macer.py` | MACER algorithm |
| `model.py` | Network architectures |
| `rs/*.py` | Randomized smoothing |
| `visualize/plotcurves.m` | Result visualization |

## Make sure you meet package requirements by running:
```
pip install -r requirements.txt
```

## Example

Here we will show how to train provably l2-robust CIFAR10 and Imgeanet model. We will use &sigma;=0.25 as an example.

### Train CIFAR10
```
python main.py --dataset cifar10 --lr 0.01 --batch_size 64 --training_method macer --sigma 0.25 --lbd 12 --gauss_num 16 --label_smoothing True
```

### Trrain Imagenet
python main.py --dataset imagenet --lr 0.1 --batch_size 256 --data_dir /blob_data/data/imagenet --training_method macer --epochs 120 --sigma 0.25 --lbd 6 --gauss_num 2 --label_smoothing True
