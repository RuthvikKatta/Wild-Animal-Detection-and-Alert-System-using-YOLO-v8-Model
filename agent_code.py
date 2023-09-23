# @authors Katta Ruthvik, Koppula Saketh Raja, Suryaneni Rohith and Pulimi Yashwanth

import cv2
import numpy as np
from keras.models import load_model
import keras.utils as image
import telepot
import time

path_to_model = "./keras_model.h5"
bot_token = "6666638666:AAFqDSJM5Hnw-YoAhIG0WiFr69btxtFk3Yc"
chat_id = "1035950237"
class_labels = ['HUMAN', 'TIGER', 'LION', 'CHEETCH', 'HYENA', 'DONKEY', 'GOAT', 'DEER', 'HOUSE', 'ELEPHANT',
                'FOX', 'LEOPARD', 'RHINOCEROS', 'SNAKE', 'WOLF', 'CAT', 'CHIMPANZEE', 'COW', 'DOG', 'DUCK', 'MONKEY', 'ZEBRA']
wild_animals = ['TIGER', 'LION', 'CHEETCH', 'HYENA',
                'ELEPHANT', 'FOX', 'LEOPARD', 'RHINOCEROS', 'SNAKE', 'WOLF']


model = load_model("path_to_model")

# Initialize the webcam
cap = cv2.VideoCapture(0)

bot = telepot.Bot(bot_token)
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    prediction = model.predict(img_tensor)
    class_index = np.argmax(prediction[0])
    class_label = class_labels[class_index]
    cv2.putText(frame, class_label, (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow('Webcam', frame)
    if class_label in wild_animals and time.time() - last_detection_time >= 5:
        cv2.imwrite('captured_image.jpg', frame)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open('captured_image.jpg', 'rb') as photo:
            bot.sendPhoto(chat_id, photo, caption=f'''🚨 ALERT: {class_label} Detected 🚨
                        📅 Timestamp: {current_time} 🕒
                        Take immediate precautions for your safety:
                        1️⃣ Stay calm and alert.
                        2️⃣ Follow official instructions.
                        3️⃣ Take necessary precautions.
                        4️⃣ Stay informed from reliable sources.
                        5️⃣ Help others while maintaining distance.
                        Cooperate with authorities and stay safe! 
                        We will provide further updates as soon as possible.Stay safe and stay vigilant!''')
        last_detection_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
