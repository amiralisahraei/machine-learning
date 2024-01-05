# Image segmentation
The project intends to construct a **U-net** model which includes **Encoder** and **Decoder**. The Encoder utilizes **MobileNetV2** as a pre-trained model to downsample and the Decoder applies **pix2pix** to upsample. In addition, the **ResNet50** is used as a pre-trained model in Decoder.

## Resource
I have applied the Image Segmentation project from the **Tensorflow** documentation.<br> 
[Resource](https://www.tensorflow.org/tutorials/images/segmentation)

## Results
As the result shows the model that trained 20 epochs and used **MobileNetV2** as decoder outperformed the other model trained 40 epochs and used **ResNet50** as decoder. However, it is worth saying
it looks the first model is going to overfit after 20th epoch but the second one is capable to train more than 40 epcohs.

  
<div align="center"><h3>MobileNetV2</h3></div>
<table>
<tr>
<td><img src="results/MobileNetV2_20_epochs.png"></td>
<td><img src="results/MobileNetV2_20_epochs2.png"></td> 
</tr>
</table>

<div align="center"><h3>ResNet50</h3></div>
<table>
<tr>
<td><img src="results/RestNet50_40_epochs.png"></td> 
<td><img src="results/RestNet50_40_epochs2.png"></td> 
</tr>
</table>




