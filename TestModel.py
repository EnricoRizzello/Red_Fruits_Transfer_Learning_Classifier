from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
for filename in os.listdir(os.getcwd()+"\TestImage"):
    img = image.load_img(os.getcwd()+"\TestImage\\"+filename, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x,axis=0)
    img_data = preprocess_input(x)
    model=load_model('red_fruits_model_resnet50.h5')
    model.predict(img_data)
    a=np.argmax(model.predict(img_data), axis=1)
    if(a == 0):
     print("Image: "+filename+" => apple = " + str("apple" in filename))
    if(a == 1): 
     print("Image: "+filename+" => cherry = " + str("cherry" in filename))
    if(a == 2): 
     print("Image: "+filename+" => strawberry =" + str("strawberry" in filename)) 
    if(a == 3): 
     print("Image: "+filename+" => tomato = " + str("tomato" in filename))  