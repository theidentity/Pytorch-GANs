# Pytorch-GANs
Implementation of DCGAN on CIFAR-10 dataset

* DCGAN architecture on PyTorch
* 32 x 32 x 3 

### Setting up the environment
* Clone the repo
* Create a virtualenvironment with python3
* Activate the environment
* Install requirements from req.txt
```
https://github.com/theidentity/Pytorch-GANs.git
python3 -m venv my_venv
pip3 install -r req.txt
source activate my_venv/bin/activate
```
### Running the program
```
python dcgan.py --epochs=10 --gpu=0 --seed=42
```
### Play around with the model
* Disciminator and Generator are in ```networks.py```
* Data input and output are in ```data_io.py```
* Custom layers are in ```pt_layers.py```

### Results
* The visualizations are in ```gen_imgs/dcgan/```
* Current model run for 10 epochs. Train longer for more complex images
* Improvment of epochs 0 to 5 to 10 :
![epoch 0](https://raw.githubusercontent.com/theidentity/Pytorch-GANs/master/gen_imgs/dcgan/00000000.png)
![epoch 5](https://raw.githubusercontent.com/theidentity/Pytorch-GANs/master/gen_imgs/dcgan/00050000.png)
![epoch 10](https://raw.githubusercontent.com/theidentity/Pytorch-GANs/master/gen_imgs/dcgan/00100000.png)


