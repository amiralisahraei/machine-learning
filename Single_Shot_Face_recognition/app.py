import numpy as np
import matplotlib.pyplot as plt
import os
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from torchvision.transforms import functional as F

from test_image_paths import list_of_test_image_paths


# Show a few faces from dataset
def show_few_faces(diretory_path):
    num_colums = 7
    num_rows = int(np.ceil(len(os.listdir(diretory_path)) / num_colums))
    plt.figure(figsize=(20, 20))
    for i, dir in enumerate(os.listdir(diretory_path)):
        for file_name in os.listdir(diretory_path + dir)[0:1]:
            image_path = diretory_path + dir + "/" + file_name
            plt.subplot(num_rows, num_colums, i + 1)
            image = Image.open(image_path)
            plt.axis("off")
            plt.title("{}".format(dir))
            plt.imshow(image)

    plt.show()


# show_few_faces("./Face_dataset/")


# Use InceptionResnetV1 as a pre-trained model to extract feature vector from each image
def faceNet_feature_extract(img_path):
    # Load the pre-trained FaceNet model
    model = InceptionResnetV1(pretrained="vggface2").eval()

    # Load and preprocess the input image
    img = Image.open(img_path)

    # Resize the image to 160x160 pixels
    img = img.resize((160, 160))

    # Convert the image to a tensor
    img_tensor = F.to_tensor(img)

    # Expand dimensions to create a batch of size 1
    img_tensor = img_tensor.unsqueeze(0)

    # Extract features using FaceNet
    with torch.no_grad():
        features = model(img_tensor)

    # Convert the features tensor to a numpy array
    feature_vector = features.detach().numpy()[0]

    return feature_vector


# Make a dictionary as key is the name of the artist and value is the corresponding feature vector
members = {}


def Store_images(names, file_names):
    for name, file_name in zip(names, file_names):
        path = f"./Face_dataset/{name}/{file_name}"
        # feature_vetor = feature_extraction(path)
        feature_vetor = faceNet_feature_extract(path)
        members[name] = feature_vetor

    return members


# Selected Images for feature extraction and a Single Shot Recongnition
list_of_artists = [
    "Angelina Jolie",
    "Brad Pitt",
    "Denzel Washington",
    "Jennifer Lawrence",
]
list_of_image_paths = [
    "001_fe3347c0.jpg",
    "001_c04300ef.jpg",
    "067_ee6435dc.jpg",
    "026_cf5be1f1.jpg",
]
dictionary = Store_images(list_of_artists, list_of_image_paths)


# Count Euclidean distance for two vectors
def distance(vector1, vector2):
    distance = np.linalg.norm(vector1 - vector2)
    return distance


# Comparison of the input images and the resource images to count their difference
def face_recognision(new_image_path, min_dist=100):
    new_img = faceNet_feature_extract(new_image_path)
    detected_member = ""

    for name in members.keys():
        difference = distance(new_img, dictionary[name])
        if difference <= min_dist:
            detected_member = name
            min_dist = difference

    return detected_member


# Test on a few new images
def test(list_of_artists):
    diretory_path = "./Face_dataset"
    num_colums = len(list_of_artists[list(list_of_artists.keys())[0]])
    num_rows = len(list(list_of_artists.keys()))
    counter = 1
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    font1 = {"color": "green"}

    for key, file_names in list_of_artists.items():
        for i, file_name in enumerate(file_names):
            image_path = diretory_path + "/" + key + "/" + file_name
            plt.subplot(num_rows, num_colums, counter)
            image = Image.open(image_path)
            detected_face = face_recognision(image_path)
            plt.axis("off")
            plt.title(
                "Actual: {} \n Detected: {}".format(key, detected_face), fontdict=font1
            )
            plt.imshow(image)
            counter += 1

    plt.show()


test(list_of_test_image_paths)
