# Deep Sort with CSP
### CSP version of [DeepSORT PyTorch](https://github.com/ZQPei/deep_sort_pytorch)

### Project structure

Pretrained model should be copied to here. (Checkpoints for CSP and Deepsort)
```angular2
./checkpoints
```
Dataset files in .npz for faster loading 
```angular2
./data_cache
```
You can put your own dataloader here. (and import them!) 
```angular2
./load_data
```
Pretrained ResNet50 model should be put here for training
```angular2
./models
```
Tasks for project. They also provide visualization. 
```angular2
./tasks
```
The output directory of tasks. It contains sequence of output images. 
```angular2
./testset_output
```

### Usage
Training:  
1. prepare your own dataset or put dataset in data_PETS2009
2. Download pre-trained ResNet50 model. ('https://download.pytorch.org/models/resnet50-19c8e357.pth')
3. Run train_csp.py in terminal or IDE. You can adjust config (eg. image size, batch size, #gpu) in config.py. 
4. The checkpoints will be stored in ./weights

Testing
1. Put trained CSP model and Deepsort model checkpoints under ./checkpoints
2. run test_csp.py in terminal or IDE
3. You can also use Tasks.ipynb for evaluation of tasks. 

### References and Credits:
1. [Pytorch implementation of deepsort with Yolo3](https://github.com/ZQPei/deep_sort_pytorch)
2. [Center-and-Scale-Prediction-CSP-Pytorch](https://github.com/zhangminwen/Center-and-Scale-Prediction-CSP-Pytorch)
3. [Deep Sort with PyTorch](https://github.com/sayef/detectron2-deepsort-pytorch)
4. [Deepsort](https://github.com/nwojke/deep_sort)
5. [SORT](https://github.com/abewley/sort)
6. [PETS2009 Benchmark Data](http://www.cvg.reading.ac.uk/PETS2009/a.html)
7. [Ground truths for PETS2009 tasks](http://www.milanton.de/data/)
```
@inproceedings{liu2018high,
  title={High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection},
  author={Wei Liu, Shengcai Liao, Weiqiang Ren, Weidong Hu, Yinan Yu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

```

