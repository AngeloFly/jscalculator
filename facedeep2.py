# import the required modules
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import requests
from io import BytesIO
import numpy as np  #
# read image
#img = cv2.imread('./images/image1.jpg')

# call imshow() using plt object
#plt.imshow(img[:,:,::-1])

# display that image
#plt.show()
# download image from url
response = requests.get('https://pps.whatsapp.net/v/t61.24694-24/339615525_1009311816730203_5688752723475398652_n.jpg?ccb=11-4&oh=01_AdR6YLHDVBauhcNsCcEIG6Xu5LK6w9Afc7ogGnXdpgOGlg&oe=64454DE9')
img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

# convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
try:
    # storing the result
    result = DeepFace.analyze(img,actions=["age", "gender"], enforce_detection=True)
    #"emotion"

    # print result
    print(result)
except ValueError as err:
    if 'Face could not be detected' in str(err):
        print('Error: No face detected. Please check if the uploaded image is a face photo.')
    else:
        raise err

