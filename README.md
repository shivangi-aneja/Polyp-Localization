For setting up environment, please read `SETUP.md`

# Polyp Localization and Detection
Polyp localization Using Fully Convolutional Networks

For our localization and detection task, we used ColonCVC dataset and ETIS-Larib dataset.


## 1. Polyp Localization

The task here is to train Fully Convolutional Network (FCN-8s)  to create segmentation masks for the polyps.
To train FCN-8s, run `main.py`.
To evaluate/test the models, run `predict_masks.py`


## 2. Polyp Detection

The task here is to train object detection network. We used SSD (Single Shot Multibox detectot to evaluate our results).
Code is available for both `Faster R-CNN` and `SSD`.

To train faster R-CNN, run the file `keras_train_frcnn.py`
To test faster R-CNN, run the file `keras_test_frcnn.py`

To run SSD, migrate to directory SSD-keras, and follow `README.md` for instructions

### How to run main.py / predict_masks.py / keras_train_frcnn.py / keras_test_frcnn.py
```bash
usage: main.py [-h] [-d DATASET] [--data-dirpath DATA_DIRPATH]
               [--n-workers N_WORKERS] [--gpu GPU] [-rs RANDOM_SEED]
               [-a ARCHITECTURE] [-l LOSS] [-b BATCH_SIZE]
               [-e EPOCHS]  [-lr LEARNING_RATE] [-opt OPTIM] [-m MODEL_NAME]
               [-r RESUME] [-tf TF_LOGS] [-wd WEIGHT_DECAY]
               [-dp DROPOUT]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        dataset, {'polyps'} (default: polyps)
  --data-dirpath DATA_DIRPATH
                        directory for storing downloaded data (default: data/)
  --n-workers N_WORKERS
                        how many threads to use for I/O (default: 4)
  --gpu GPU             ID of the GPU to train on (or '' to train on CPU)
                        (default: 0)
  -rs RANDOM_SEED, --random-seed RANDOM_SEED
                        random seed for training (default: 1)
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        network architecture name, {'fcn8s1'}
                        (default: fcn8s1)
  -l LOSS, --loss LOSS
                        loss function, {'cse'} (default: cse, mse, dse)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        input batch size for training (default: 50)
  -e EPOCHS, --epochs EPOCHS
                        number of epochs (default: 100)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        initial learning rate (default: 0.0001)
  -opt OPTIM, --optim OPTIM
                        optimizer, {'adam', 'sgd', 'adagrad', 'rms_prop'} (default: adam)
  -m MODEL_NAME, --model_name MODEL_NAME
                        name of the model (default: 'fcn8s_1')
  -r RESUME, --resume RESUME
                        flag to resume training (default: 'y')
  -tf TF_LOGS, --tf_logs TF_LOGS
                        path for tensorboaard loggging (default: 'tf_logs')
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay (default: 0)
  -dp DROPOUT, --dropout DROPOUT
                        dropout (default: 0)
```
## Sample Command
```bash
python3 main.py -d polyps -a fcn8s -b 100  -e 20
```


## 3. Misc

Code is available for both Faster R-CNN and SSD. But I used SSD for the task of detecting polyps.

All the results and models can be downloaded from [here](https://drive.google.com/file/d/1Fb9XrDYKtzJiysEi79dC_NZlsrgUr-9o/view?usp=sharing).
No pretrained models were used. Everything is trained from scratch.

The presentations can be downloaded from [here](https://drive.google.com/file/d/1C2l-P5fKZWgDAObaIJ25ryeBtg5YgRug/view?usp=sharing).