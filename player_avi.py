import ctypes
import sys
import os
import os.path
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPalette, QColor, QImage, QPixmap
from PyQt5.QtWidgets import *
import vlc
#from anomaly_detector import AnomalyDetector
from object_detector import ObjectDetector
from tracker import Tracker

import time

#import cProfile

#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput

WIDTH = 720
HEIGHT = 576

size = WIDTH * HEIGHT * 4
buf = (ctypes.c_ubyte * size)()
buf_p = ctypes.cast(buf, ctypes.c_void_p)

VideoLockCb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))

videoframe = 0
abnormal_label = 0

old_time = 0

#detector = AnomalyDetector(0.33)
obj_detector = ObjectDetector(0.001)
tracker = Tracker(min_confidence=0.0575,
                  min_new_confidence=0.17,
                  exclusive_threshold=700,
                  match_threshold=295,
                  max_unseen=17,
                  smoothing_factor=0.35)


counter = 0

class Player(QMainWindow):
    """A simple Media Player using VLC and Qt
    """
    def __init__(self, master=None):
        QMainWindow.__init__(self, master)
        self.setWindowTitle("Media Player")

        # creating a basic vlc instance
        self.instance = vlc.Instance()
        # creating an empty vlc media player
        self.mediaplayer = self.instance.media_player_new()

        self.createUI()
        self.isPaused = False
        #print("test")

        chroma = "RGBA"
        # create byte objects from the strings
        chroma = chroma.encode('utf-8')
        chroma_p = ctypes.cast(chroma, ctypes.c_char_p)

        mode = "yadif"
        mode = mode.encode('utf-8')
        mode_p = ctypes.cast(mode, ctypes.c_char_p)

        vlc.libvlc_video_set_format(self.mediaplayer, chroma_p, WIDTH, HEIGHT, WIDTH * 4)
        #vlc.libvlc_video_set_deinterlace(self.mediaplayer, mode_p)
        vlc.libvlc_media_player_set_rate(self.mediaplayer, 0.5)
        vlc.libvlc_video_set_callbacks(self.mediaplayer, self._lockcb, None, self._display, None)

    @VideoLockCb
    def _lockcb(opaque, planes):
        #print("test2")
        planes[0] = buf_p

    @vlc.CallbackDecorators.VideoDisplayCb
    def _display(opaque, picture):
        global old_time
        #print(time.time() - old_time)
        old_time = time.time()

        global counter

        #print("--------------------------------")

        global tracker

        #print("test3")
        global videoframe
        global abnormal_label

        start = time.time()

        frame = Image.frombuffer("RGBA", (WIDTH, HEIGHT), buf, "raw")
        #frame.show()

        frame = frame.convert('RGB')

        #with PyCallGraph(output=GraphvizOutput()):
        bboxes = obj_detector.apply_model(frame)

        #print("bboxes: " + str(bboxes))

        objects = tracker.update(bboxes)

        #observations = observations[-20:]

        frame2 = frame.copy()
        #frame = frame.crop((590, 0, 590 + 1072, 1072))
        #frame.show()
        #(score, dirty) = detector.apply_model(frame)
        #print(score)

        draw = ImageDraw.Draw(frame2)
        font = ImageFont.truetype("FreeMono.ttf", 40)

        #draw.text((0,0), ("Abnormal " if score > 0.04 else "Normal ") + ("Dirty" if dirty else "Clean"), (255,255,255), font=font)

        #for bbox in bboxes:
        #    if bbox[1] > 0.3:
        #        colour = (255, 0, 0)
        #
        #        if (bbox[1] < 0.70):
        #            colour = (255, 255, 0)
        #
        #        if (bbox[1] < 0.40):
        #            colour = (0, 255, 0)

            #draw.rectangle([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']], fill=None, outline=(0,0,255), width=2)
        #        draw.text((bbox[2], bbox[4]), "{:.2f}".format(bbox[1]), colour, font=font)

        #global counter

            #draw.line(previous_bboxes, fill=(0, 0, 255), width=4)
        for i in range(len(objects)):
            if (objects[i].confidence >= 0.3):
                colour = (255, 0, 0)

                if (objects[i].confidence < 0.8):
                    colour = (255, 255, 0)

                if (objects[i].confidence < 0.5):
                    colour = (0, 255, 0)

                draw.rectangle([objects[i].mean[0] - objects[i].mean[2],
                                objects[i].mean[1] - objects[i].mean[3],
                                objects[i].mean[0] + objects[i].mean[2],
                                objects[i].mean[1] + objects[i].mean[3]],
                               fill=None, outline=colour,
                               width=3)

        #frame2.save("out/{}.jpg".format(counter))
        counter += 1

        end = time.time()
        #print(end - start)

        img = ImageQt(frame2)
        img = img.scaled(QSize(videoframe.width(), videoframe.height()), Qt.KeepAspectRatio)
        pix = QPixmap.fromImage(img)
        videoframe.setPixmap(pix)

        #abnormal_label.setText("{:.2f}".format(score))

        #os._exit(1)

        #(img)

    def createUI(self):
        """Set up the user interface, signals & slots
        """
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        global videoframe
        global abnormal_label
       
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

        abnormal_label = QLabel("None")
        abnormal_label.setFixedHeight(30)
        self.hbuttonbox.addWidget(abnormal_label)

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
        if self.mediaplayer.is_playing():
            self.mediaplayer.pause()
            self.playbutton.setText("Play")
            self.isPaused = True
        else:
            if self.mediaplayer.play() == -1:
                self.OpenFile()
                return
            self.mediaplayer.play()
            self.playbutton.setText("Pause")
            self.timer.start()
            self.isPaused = False

    def Stop(self):
        """Stop player
        """
        self.mediaplayer.stop()
        self.playbutton.setText("Play")

    def OpenFile(self, filename=None):
        """Open a media file in a MediaPlayer
        """
        if filename is None:
            filename = QFileDialog.getOpenFileName(self, "Open File", os.path.expanduser('~'))[0]
        if not filename:
            return

        # create the media
        if sys.version < '3':
            filename = unicode(filename)
        self.media = self.instance.media_new(filename)
        # put the media in the media player
        self.mediaplayer.set_media(self.media)

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
        self.mediaplayer.audio_set_volume(Volume)

    def setPosition(self, position):
        """Set the position
        """
        # setting the position to where the slider was dragged
        self.mediaplayer.set_position(position / 1000.0)
        # the vlc MediaPlayer needs a float value between 0 and 1, Qt
        # uses integer variables, so you need a factor; the higher the
        # factor, the more precise are the results
        # (1000 should be enough)

    def updateUI(self):
        """updates the user interface"""
        # setting the slider to the desired position
        self.positionslider.setValue(int(self.mediaplayer.get_position() * 1000.0))

        if not self.mediaplayer.is_playing():
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