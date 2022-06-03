# базовый образ контейнера - это tensorflow image с вшитой библиотекой tensorflow
FROM tensorflow/tensorflow:2.3.0-gpu

# здесь происходит добавление пакетов
# model, scripts, src (основная логика работы модуля находится здесь), tfod2
ADD . /

# установка необходимых зависимостей для модуля
RUN apt update
RUN apt install -y musl-dev make cmake git gcc g++ gfortran bash curl libffi-dev ffmpeg libsm6 libxext6

# установка python библиотек
RUN pip3 install -r requirements.txt
RUN pip3 install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag rospkg roslz4 sensor_msgs geometry_msgs cv_bridge
RUN pip3 uninstall -y opencv-python
RUN pip3 install opencv-python

# запуск скрипта распознавания с отправкой в топик
CMD make run-ros