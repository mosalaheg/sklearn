from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

flower = load_sample_image('flower.jpg')

flower.dtype
flower.shape

plt.imshow(flower)



