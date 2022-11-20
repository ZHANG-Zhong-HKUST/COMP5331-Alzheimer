# model training

The model can be trained by simply running 'python train.py'. When running, make sure to put the indicator data (csv file) and image data (ADNIp directory) in the same directory as train.py.

You can also specify the output directory and hyperparameters. An example is shown below.
  
```bash
python train.py --out '/home/output_dir' --epochs 200 --batch_size 6 --l2_regular 0.00001 --lr 1e-7
```