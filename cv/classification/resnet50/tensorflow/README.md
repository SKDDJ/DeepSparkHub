
## Prepare

### Install packages

```shell
pip3 install absl-py git+https://github.com/NVIDIA/dllogger#egg=dllogger
```

### Download datasets


[Downloading and converting to TFRecord format](https://github.com/kmonachopoulos/ImageNet-to-TFrecord)  or 
[here](https://github.com/tensorflow/models/tree/master/research/slim#downloading-and-converting-to-tfrecord-format)
make a file named imagenet_tfrecord, and store imagenet datasest convert to imagenet_tfrecord   



## Training

### Training on single card

```shell
bash run_train_resnet50_imagenette.sh
```

### Training on mutil-cards
```shell
bash run_train_resnet50_multigpu_imagenette.sh
```


## Result

|               | acc       |       fps |
| ---           | ---       | ---       |
|    multi_card | 0.9860    | 236.9     |