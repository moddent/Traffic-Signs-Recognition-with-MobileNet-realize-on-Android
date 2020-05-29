# Traffic-Signs-Recognition-with-MobileNet-realize-on-Android
Use phone to take a traffic sign picture, then recognizing it.
## Data set
[German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)  
![image](/image/dataset.jpg)
## Requirements
* Python 3.x
* Pytorch >= 1.3.0
* Android Studio >= 4.0
## Getting Started
Here is an example that uses some images.  
You can download the full image from [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).  
After downloading, you just put the image in the corresponding folder.  

To testing dataloader, run the following script from the directory:  

    python Dataloader_test.py  

Training MobileNet, directly run:  

    python main.py  
## Demo  
<p align="left">
    <img src="/image/demo1.jpg" width="200" height="400"/>
    <img src="/image/demo2.jpg" width="200" height="400"/>
    <img src="/image/demo3.jpg" width="200" height="400"/>
    <img src="/image/demo4.jpg" width="200" height="400"/>
    <img src="/image/demo5.jpg" width="200" height="400"/>
    <img src="/image/demo6.jpg" width="200" height="400"/>
    <img src="/image/demo7.jpg" width="200" height="400"/>
    <img src="/image/demo8.jpg" width="200" height="400"/>
</p>

## License
This project is released under the MIT license.   
If you have any issue, please submit an issue or contact us at:zxcz14071407@gmail.com/tyson.wang26@gmail.com
## References
MobileNet Architecture:https://github.com/marvis/pytorch-mobilenet  
APP:https://pytorch.org/mobile/android/  
