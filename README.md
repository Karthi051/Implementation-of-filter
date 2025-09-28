<img width="389" height="411" alt="6e67f248-e7c3-4a52-bd55-e1fdf79a27d0" src="https://github.com/user-attachments/assets/ba6836f5-3d27-4683-83e7-a7da7373b544" /># Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the required libraries.

### Step2

Convert the image from BGR to RGB.
### Step3

Apply the required filters for the image separately.
### Step4

Plot the original and filtered image by using matplotlib.pyplot.
### Step5

End the program.

## Program:
### Developed By   : Dharshan V
### Register Number: 212224240035
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img_path = r"Karthi photo.jpeg"  # Change this to your correct path

if not os.path.exists(img_path):
    print(" Image not found. Check the file path.")
else:
    image1 = cv2.imread(img_path)
    if image1 is None:
        print(" Image could not be loaded (possibly corrupted or unsupported format).")
    else:
        image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        kernel = np.ones((11, 11), np.float32) / 169
        image3 = cv2.filter2D(image2, -1, kernel)

        plt.figure(figsize=(9, 9))
        plt.subplot(1, 2, 1)
        plt.imshow(image2)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(image3)
        plt.title("Average Filter Image")
        plt.axis("off")
        plt.show()




```
ii) Using Weighted Averaging Filter
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()



```
iii) Using Gaussian Filter
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()




```
iv)Using Median Filter
```Python
median = cv2.medianBlur(image2, 13)
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.title("Median Blur")
plt.axis("off")
plt.show()




```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python


kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()



```
ii) Using Laplacian Operator
```Python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()




```

## OUTPUT:
## original image:
<img width="717" height="359" alt="24ac89ee-9af0-4c1e-bad1-f1c2b921303e" src="https://github.com/user-attachments/assets/ba3410d0-31b6-46a6-9864-ec33638968da" />


### 1. Smoothing Filters

## i) Using Averaging Filter
<img width="389" height="411" alt="bd475add-a34e-4094-b7b1-60db2d360f84" src="https://github.com/user-attachments/assets/2bf8f861-94c2-472b-be52-d453166abb86" />



## ii)Using Weighted Averaging Filter
<img width="389" height="411" alt="bd475add-a34e-4094-b7b1-60db2d360f84" src="https://github.com/user-attachments/assets/19c97418-75fb-4b10-bef3-a1c26e6afdbe" />



## iii)Using Gaussian Filter
<img width="389" height="411" alt="6e67f248-e7c3-4a52-bd55-e1fdf79a27d0" src="https://github.com/user-attachments/assets/8917c62b-2bf3-44f7-ad7b-b8d4ea59fb0a" />


## iv) Using Median Filter
<img width="389" height="411" alt="616125a9-2a7a-482b-a4d4-da1f0f8e0ecc" src="https://github.com/user-attachments/assets/ffbdcc9b-0f2e-4afc-a3e7-4106a9e857a0" />

### 2. Sharpening Filters
## i) Using Laplacian Kernal
<img width="389" height="411" alt="babe24cb-dc28-4021-a5df-e1d50fde0b26" src="https://github.com/user-attachments/assets/c16be717-ec7a-4c4f-b65e-6d700f4606d0" />



## ii) Using Laplacian Operator
<img width="389" height="411" alt="29caa08c-1e1e-4f46-a31d-7c803cdc9d47" src="https://github.com/user-attachments/assets/e55aeb22-f83f-4d55-9208-449b40ab13b2" />



## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
