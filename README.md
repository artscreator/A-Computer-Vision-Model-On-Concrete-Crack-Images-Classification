# Concrete-Crack-Images-Classification
 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Description
The dataset contains concrete images having cracks. The data is collected from various METU Campus Buildings.
The dataset is divided into two as negative and positive crack images for image classification.
Each class has 20000images with a total of 40000 images with 227 x 227 pixels with RGB channels.
The dataset is generated from 458 high-resolution images (4032x3024 pixel) with the method proposed by Zhang et al (2016).
High-resolution images have variance in terms of surface finish and illumination conditions.
No data augmentation in terms of random rotation or flipping is applied.

## Data source:
https://data.mendeley.com/datasets/5y9wdsg2zt/2

## Model Training accuracy & loss
![epoch_acc](https://user-images.githubusercontent.com/37768522/212000561-feb40135-bc6c-4b5d-a530-563fda767c7c.png)
![epoch_loss](https://user-images.githubusercontent.com/37768522/212000571-e11acc12-c7a5-459a-bb14-14192fe53882.png)

![loss, acc, val_loss, val_accuracy](https://user-images.githubusercontent.com/37768522/212000627-979a9d3d-e40d-4f00-ad2a-98fef8cb6338.png)
