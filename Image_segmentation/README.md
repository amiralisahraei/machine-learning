# Image segmentation
The project intends to construct a **U-net** model which includes **Encoder** and **Decoder**. The Encoder utilizes **MobileNetV2** as a pre-trained model to downsample and the Decoder applies **pix2pix** to upsample. In addition, the **ResNet50** is used as a pre-trained model in Decoder.

## Resource
I have applied the Image Segmentation project from the **Tensorflow** documentation.<br> 
[Resource](https://www.tensorflow.org/tutorials/images/segmentation)

## Results
<table>
<tr>
<h3>MobileNetV2</h3>
<td><img src="results/MobileNetV2_20_epochs.png"></td>
<td><img src="results/MobileNetV2_20_epochs2.png"></td> 
</tr>
<tr>
<h3>ResNet50</h3>
<td><img src="results/RestNet50_40_epochs.png"></td> 
<td><img src="results/RestNet50_40_epochs2.png"></td> 
</tr>
</table>



