
# Необходимо на основе OpenCV(https://opencv.org/) сделать нейросеть 
# с функцией определения эмоции(человек/собака/кошка и т.п.)прямого видеопотока.
# Далее ссылку на гитхаб присылайте для проверки.


# вообще задание немного странное, так как OpenCV это не фреймворк для создания нейросетей, а скорее как либа обработки фото и видео


# вот тут можно было обучить нейросеть на этих данных
# https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/
# я взял готовую, так как обучать долго, а надо цигель цигель

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#вырубили варнинги
import cv2 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model


# Определение функции для извлечения признаков из изображения 
def  extract_features ( image ): 
    feature = np.array(image) 
    feature = feature.reshape( 1 , 48 , 48 , 1 ) 
    return feature / 255.0 



# Загрузка предварительно обученной  модели
model = create_model()
model.load_weights('model.h5')

# Загрузка каскадного классификатора Хаара для обнаружения лиц
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



# Открыть веб-камеру (camera)
webcam = cv2.VideoCapture( 0 ) 

# Определить метки для классов эмоций
labels = { 0 : 'angry' , 1 : 'disgust' , 2 : 'fear' , 3 : 'happy' , 4 : 'neutral' , 5 : 'sad' , 6 : 'surprise' } 

while  True : 
    # Считать кадр с веб-камеры
    i, im = webcam.read() 

    # Преобразовать кадр в оттенки серого

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    try : 
        # Для каждого обнаруженного лица выполнить распознавание эмоций на лице 
        for (p, q, r, s) in faces: 
            # Извлечь область интереса (ROI), которая содержит
            image = gray[q:q + s, p:p + r] 

            # Нарисовать прямоугольник вокруг обнаруженного лица
            cv2.rectangle(im, (p, q), (p + r, q + s), ( 255 , 0 , 0 ), 2 ) 

            # Изменить размер изображения лица до требуемого входного размера (48x48)
            image = cv2.resize(image, ( 48 , 48 )) 

            # Извлечь признаки из измененного изображения лица
            img = extract_features(image) 

            # Сделать прогноз с использованием обученной модели
            pred = model.predict(img) 

            # Получить прогнозируемую метку для эмоции
            prediction_label = labels[pred.argmax()] 

            # Отобразить прогнозируемую метку эмоции рядом с обнаруженным лицом
            cv2.putText(im, f'Emotion: {prediction_label} ' , (p - 10 , q - 10 ), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2 , ( 0 , 0 , 255 )) 

        # Отобразить кадр с аннотациями в реальном времени
        cv2.imshow( "Real-time emotion recognize as test task" , im)     


        # Разорвать цикл, если нажата клавиша 'Esc' 
        if cv2.waitKey( 1 ) == 27 : 
            break 

    except cv2.error: 
        pass 

# Освободите веб-камеру и закройте все окна OpenCV
webcam.release() 
cv2.destroyAllWindows()