# Polyp Localization and Detection
Polyp Detection Using SSD (Single Shot Multibox Detector)


## Polyp Detection

The task here is to train object detection network. We used SSD (Single Shot Multibox detectot to evaluate our results).

To train SSD 7 , run the file `train_ssd7.py`
To test saved SSD 7 , run the file `test_ssd7.py`

To train SSD 300 , run the file `train_ssd300.py`
To test SSD 300 , run the file `test_ssd300.py`

To train SSD 512 , run the file `train_ssd512.py`
To test SSD 512 , run the file `test_ssd512.py`

SSD-300 gives best results so far.


### How to run train_ssd7.py / train_ssd300.py / train_ssd512.py
```bash
usage: train_ssd300.py [-h] [-d DATASET] [--gpu GPU] [-b BATCH_SIZE]
                [-e EPOCHS]  [-lr LEARNING_RATE] [-m MODEL_NAME]
                [-tf TF_LOGS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        dataset, {'polyps'} (default: polyps)
  --gpu GPU             ID of the GPU to train on (or '' to train on CPU)
                        (default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 50)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs (default: 100)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        initial learning rate (default: 0.0001)
  -m MODEL_NAME, --model_name MODEL_NAME
                        name of the model (default: 'default')
  -tf TF_LOGS, --tf_logs TF_LOGS
                        path for tensorboaard loggging (default: 'tf_logs')
```
## Sample Command
```bash
python3 train_ssd300.py -d polyps_rcnn -b 16  -e 20
```

### How to run test_ssd7.py / test_ssd300.py / test_ssd512.py
```bash
usage: train_ssd300.py [-h] [-d DATASET] [--gpu GPU] [-b BATCH_SIZE]
                [-lr LEARNING_RATE] [-m MODEL_NAME]
                [-tf TF_LOGS]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        dataset, {'polyps'} (default: polyps)
  --gpu GPU             ID of the GPU to train on (or '' to train on CPU)
                        (default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 1)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        initial learning rate (default: 0.0001)
  -m MODEL_NAME, --model_name MODEL_NAME
                        name of the model (default: 'default')
  -tf TF_LOGS, --tf_logs TF_LOGS
                        path for tensorboaard loggging (default: 'tf_logs')
```
## Sample Command
```bash
python3 test_ssd300.py -m ssd300 -b 16
```