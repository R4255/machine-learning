import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import datasets
from keras.preprocessing.image import ImageDataGenerator
#create a data generator object that transforms images
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
#normalize the pixel value to be between 0 and 1
train_images,test_images=train_images/255.0,test_images/255.0
datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
#picking an image to transform 
test_img=train_images[20]
img=image.img_to_array(test_img)
#convert the image to numpy array
img=img.reshape((1,)+img.shape)#reshape image
i=0
for batch in datagen.flow(img,save_prefix='test',save_format='jpeg'):
    plt.figure(i)
    plot=plt.imshow(image.img_to_array(batch[0]))
    i+=1
    if i>4:#show 4 images
        break
plt.show()
