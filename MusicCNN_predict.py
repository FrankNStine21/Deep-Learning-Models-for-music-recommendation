from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np

class_labels = {
    0: "electronic",
    1: "experimental",
    2: "folk",
    3: "hip-hop",
    4: "instrumental",
    5: "international",
    6: "pop",
    7: "rock",
}
image_height = 128
image_width = 1280

img = load_img('FMA/mel/mel/test/hip-hop/086_086416_s0.png', target_size=(image_height, image_width), color_mode='grayscale')
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255

model = load_model('MusicCNN_50.h5')
pred = model.predict(img_array)
print(pred)
pred_class_num = np.argmax(pred)
print('Prediction: ', class_labels[pred_class_num])
