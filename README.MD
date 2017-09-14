# SupervisedDescentMethod
This is an implementation of Supervised Descent Method algorithm for face alignment.
```
@inproceedings{xiong2013supervised,
  title={Supervised descent method and its applications to face alignment},
  author={Xiong, Xuehan and De la Torre, Fernando},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={532--539},
  year={2013}
}
```
Note: Many ideas of my implementation come from [patrikhuber](https://github.com/patrikhuber/superviseddescent) and [RoboPai](https://github.com/RoboPai/sdm), thanks their nice work. Intention of this implementation is to make codes clean and more readable.

## Requirements
* OpenCV 2.x or 3.x

## Usage
### Build
```
mkdir build
cd build
cmake ..
make
```

### Test on image
```
./examples/example test ../data/img_202.jpg
```
You will see like this:

![image](https://github.com/laughteroverflow/SupervisedDescentMethod/raw/master/data/img_202_out.jpg)

### Test on video
```
./examples/example demo
```
Run it without PATH param will use default camera.
Or you can append path of video file to the command.

### Train model
* Prepare training data like IBUG dataset
* Make a list of images(absolute paths)
* Run ./examples/example train LIST_FILE

## To do
* Some details of this implementation is not so perfect, fix it later
* Tracking
* Head pose estimation
