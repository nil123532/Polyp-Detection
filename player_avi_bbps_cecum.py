import ctypes
import sys
import os
import os.path
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPalette, QColor, QImage, QPixmap
from PyQt5.QtWidgets import *
import os
#os.add_dll_directory('C://Program Files//VideoLAN//VLC')
#os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')
import vlc

from bbps import BBPS_Scorer
from cecum import Cecum_Detector
from object_detector import ObjectDetector
from tracker import Tracker

import cv2
import pandas as pd
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt

import time

DATA_ROOT = 'C://Users//a1786723//Desktop//Project//data'

WIDTH = 720
HEIGHT = 576
SCALE = 128
#VIDEO_NAME = "C://Users//a1786723//Desktop//Project//data//BBPS_0_1.avi"
#VIDEO_NAME = "C://Users//a1786723//Desktop//Project//data//BBPS_2_3_1.avi"
#VIDEO_NAME = "C://Users//a1786723//Desktop//Project//data//cecum.avi"
#VIDEO_NAME = "C://Users//a1786723//Desktop//Project//data//232.avi"
#VIDEO_NAME = "C://Users//a1786723//Desktop//Project//data//cecum2.avi"
VIDEO_NAME = "C://Users//a1786723//Desktop//Project//data//BBPS_polyp.avi"

FILE_STATION = "C://Users//a1786723//Desktop//Project//data//cropped//cropped.png"

size = WIDTH * HEIGHT * 4
buf = (ctypes.c_ubyte * size)()
buf_p = ctypes.cast(buf, ctypes.c_void_p)

VideoLockCb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

videoframe = 0
obj_detector = ObjectDetector(0.2)
tracker = Tracker(min_confidence=0.05,
                  min_new_confidence=0.1,
                  exclusive_threshold=700,
                  match_threshold=300,
                  max_unseen=17,
                  smoothing_factor=0.45)
bbps_scorer = BBPS_Scorer(model_file='model/ResNet_Cifar10_BBPS_final.h5',
                          smoothing_factor=0.2)
cecum_detector = Cecum_Detector(model_file='model/ResNet_Cifar10_cecum.h5',
                                smoothing_factor=0.2,
                                threshold=0.5,
                                cecum_reached_count_threshold=25)

counter = 0

def mask(image):
    mask = np.zeros(image.shape, dtype=np.uint8)

    roi_corners = np.array([[(3,85), (45,85), (45,128), (110,128), (128,95), (128,30), (110,0), 
                         (21,0), (3,30)]], dtype=np.int32)

    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

class Player(QMainWindow):
    """A simple Media Player using VLC and Qt
    """
    def __init__(self, master=None):
        QMainWindow.__init__(self, master)
        self.setWindowTitle("Media Player")
        global mediaplayer
        # creating a basic vlc instance
        self.instance = vlc.Instance()
        # creating an empty vlc media player
        mediaplayer = self.instance.media_player_new()

        self.createUI()  
        self.isPaused = False

        chroma = "RGBA"

        # create byte objects from the strings
        chroma1 = chroma.encode('utf-8')

        chroma_p = ctypes.cast(chroma1, ctypes.c_char_p)

        vlc.libvlc_video_set_format(mediaplayer, chroma_p, WIDTH, HEIGHT, WIDTH * 4)
        vlc.libvlc_media_player_set_rate(mediaplayer, 0.25)
        vlc.libvlc_video_set_callbacks(mediaplayer, self._lockcb, None, self._display, None)

    @VideoLockCb
    def _lockcb(opaque, planes):
        planes[0] = buf_p



    @vlc.CallbackDecorators.VideoDisplayCb
    def _display(opaque, picture):
        global videoframe
        global mediaplayer
        global bbps_scorer
        global cecum_detector
        global counter

        start = time.time()

        frame_draw = Image.frombuffer("RGBA", (WIDTH, HEIGHT), buf, "raw")

        frame_draw = frame_draw.convert('RGB')
        frame_predict = frame_draw.copy()

        frame_draw = add_margin(frame_draw, 0, 0, 0, 250, (0, 0, 0))
        draw = ImageDraw.Draw(frame_draw)
        font = ImageFont.truetype("FreeMono.ttf", 35)

        bboxes = obj_detector.apply_model(frame_predict)

        objects = tracker.update(bboxes)

        frame_predict = frame_predict.crop((40, 22, 663, 554))
        #frame_predict.show()
        #frame_predict.save(FILE_STATION)
        #frame_arr = asarray(frame)
        #image = cv2.imread(FILE_STATION, cv2.IMREAD_UNCHANGED)
        image = np.array(frame_predict)
        image = image[:, :, ::-1].copy()
        image = cv2.resize(image,(SCALE, SCALE))
        #image = cv2.resize(np.array(frame),(SCALE, SCALE))
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        clean_image = np.array(mask(image_RGB))
        clean_image = clean_image.astype("float") / 255.0
        clean_image = clean_image.reshape(1, clean_image.shape[0], clean_image.shape[1], clean_image.shape[2])

        #Bowel Prep
        bbps_score = bbps_scorer.predict(clean_image)
        if  bbps_score <= 0.5:
            #print('bbps', bbps_score)
            bbps_colour = (160,27,16)
            bbps_text = 'BBPS 0-1'
        else:
            bbps_colour = (58,153,68)
            bbps_text = 'BBPS 2-3'

        draw.text((5, 0), bbps_text, font=font, fill=bbps_colour)

        #Cecum
        cecum_reached = cecum_detector.predict(clean_image)
        if cecum_reached:
            draw.text((5, 50), 'Cecum Reached', font=font, fill=(174, 204, 242))

        end = time.time()

        #print(end - start)

        for i in range(len(objects)):
            if objects[i].confidence > 0.33:
                colour = (255, 0, 0)

                if objects[i].confidence < 0.70:
                    colour = (255, 255, 0)

                if objects[i].confidence < 0.40:
                    colour = (0, 255, 0)

                draw.rectangle([objects[i].mean[0] - objects[i].mean[2] + 250,
                                objects[i].mean[1] - objects[i].mean[3],
                                objects[i].mean[0] + objects[i].mean[2] + 250,
                                objects[i].mean[1] + objects[i].mean[3]],
                               fill=None, outline=colour,
                               width=3)

                #draw.text((objects[i].mean[0] - objects[i].mean[2], objects[i].mean[1] - objects[i].mean[3]),
                #          "{:.2f}".format(objects[i].confidence), colour, font=font)

        frame_draw.save("out/{}.jpg".format(counter))
        counter += 1

        img = ImageQt(frame_draw)
        img = img.scaled(QSize(videoframe.width(), videoframe.height()), Qt.KeepAspectRatio)
        pix = QPixmap.fromImage(img)
        videoframe.setPixmap(pix)

    def createUI(self):
        """Set up the user interface, signals & slots
        """
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        global videoframe
        #global abnormal_label
        global mediaplayer

        videoframe = QLabel()
        videoframe.setMinimumSize(QSize(100, 100))
        self.palette = videoframe.palette()
        self.palette.setColor (QPalette.Window,
                               QColor(0,0,0))
        videoframe.setPalette(self.palette)
        videoframe.setAutoFillBackground(True)

        self.positionslider = QSlider(Qt.Horizontal, self)
        self.positionslider.setToolTip("Position")
        self.positionslider.setMaximum(1000)
        self.positionslider.sliderMoved.connect(self.setPosition)

        self.hbuttonbox = QHBoxLayout()
        self.playbutton = QPushButton("Play")
        self.hbuttonbox.addWidget(self.playbutton)
        self.playbutton.clicked.connect(self.PlayPause)

        self.stopbutton = QPushButton("Stop")
        self.hbuttonbox.addWidget(self.stopbutton)
        self.stopbutton.clicked.connect(self.Stop)

        self.hbuttonbox.addStretch(1)
        #self.volumeslider = QSlider(Qt.Horizontal, self)
        #self.volumeslider.setMaximum(100)
        #self.volumeslider.setValue(self.mediaplayer.audio_get_volume())
        #self.volumeslider.setToolTip("Volume")
        #self.hbuttonbox.addWidget(self.volumeslider)
        #self.volumeslider.valueChanged.connect(self.setVolume)

        #abnormal_label = QLabel("None")
        #abnormal_label.setFixedHeight(30)
        #self.hbuttonbox.addWidget(abnormal_label)

        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(videoframe)
        self.vboxlayout.addWidget(self.positionslider)
        self.vboxlayout.addLayout(self.hbuttonbox)

        self.widget.setLayout(self.vboxlayout)

        open = QAction("&Open", self)
        open.triggered.connect(self.OpenFile)
        exit = QAction("&Exit", self)
        exit.triggered.connect(sys.exit)
        menubar = self.menuBar()
        filemenu = menubar.addMenu("&File")
        filemenu.addAction(open)
        filemenu.addSeparator()
        filemenu.addAction(exit)

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.updateUI)

    def PlayPause(self):
        """Toggle play/pause status
        """
        global mediaplayer

        if mediaplayer.is_playing():
            mediaplayer.pause()
            self.playbutton.setText("Play")
            self.isPaused = True
        else:
            if mediaplayer.play() == -1:
                self.OpenFile()
                return
            mediaplayer.play()
            self.playbutton.setText("Pause")
            self.timer.start()
            self.isPaused = False

    def Stop(self):
        """Stop player
        """
        global mediaplayer
        mediaplayer.stop()
        self.playbutton.setText("Play")

    def OpenFile(self, filename=None):
        """Open a media file in a MediaPlayer
        """
        global mediaplayer
        if filename is None:
            filename = QFileDialog.getOpenFileName(self, "Open File", os.path.expanduser('~'))[0]
        if not filename:
            return

        # create the media
        if sys.version < '3':
            filename = unicode(filename)
        self.media = self.instance.media_new(filename)
        # put the media in the media player
        mediaplayer.set_media(self.media)

        # parse the metadata of the file
        self.media.parse()
        # set the title of the track as window title
        self.setWindowTitle(self.media.get_meta(0))



        # the media player has to be 'connected' to the QFrame
        # (otherwise a video would be displayed in it's own window)
        # this is platform specific!
        # you have to give the id of the QFrame (or similar object) to
        # vlc, different platforms have different functions for this
        #if sys.platform.startswith('linux'): # for Linux using the X Server
        #    self.mediaplayer.set_xwindow(videoframe.winId())
        #elif sys.platform == "win32": # for Windows
        #    self.mediaplayer.set_hwnd(videoframe.winId())
        #elif sys.platform == "darwin": # for MacOS
        #    self.mediaplayer.set_nsobject(int(videoframe.winId()))

        self.PlayPause()

    def setVolume(self, Volume):
        """Set the volume
        """
        global mediaplayer
        mediaplayer.audio_set_volume(Volume)

    def setPosition(self, position):
        """Set the position
        """
        global mediaplayer
        # setting the position to where the slider was dragged
        mediaplayer.set_position(position / 1000.0)
        # the vlc MediaPlayer needs a float value between 0 and 1, Qt
        # uses integer variables, so you need a factor; the higher the
        # factor, the more precise are the results
        # (1000 should be enough)

    def updateUI(self):
        """updates the user interface"""
        global mediaplayer
        # setting the slider to the desired position
        self.positionslider.setValue(int(mediaplayer.get_position() * 1000.0))

        if not mediaplayer.is_playing():
            # no need to call this function if nothing is played
            self.timer.stop()
            if not self.isPaused:
                # after the video finished, the play button stills shows
                # "Pause", not the desired behavior of a media player
                # this will fix it
                self.Stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = Player()
    player.show()
    player.resize(640, 480)
    if sys.argv[1:]:
        player.OpenFile(sys.argv[1])
    sys.exit(app.exec_())