# Notes

config file: `transfuser/config.py`

train: 

```shell
python train.py --id transfuser --batch_size 56 --logdir transfuser/log/exp

tensorboard --logdir=~/colin/transfuser/transfuser/log/exp --port=17777
```

evaluation:

```shell
bash leaderboard/scripts/run_evaluation.sh
```
