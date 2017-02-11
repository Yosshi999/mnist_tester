import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from sklearn.externals import joblib
import numpy as np

# classifier
from mnistCNN import NN
classifier = joblib.load("./mnistClassifier")


class MouseWidget(QWidget):
    def __init__(self, id, parent=None):
        super(MouseWidget, self).__init__(parent)
        self.px = None
        self.py = None
        self.points = []
        self.psets = []
        self.resize(200,200)
        self.id = id
        self.mgr = parent
    
    def clearPoints(self):
        self.points = []
        self.psets = []
        self.update()

    def mousePressEvent(self, event):
        if not self.points:
            # first click
            self.mgr.onFirstClick(self.id)
        self.points.append(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        self.points.append(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        self.pressed = False
        self.psets.append(self.points)
        self.points = []
        self.update()
        self.mgr.onMouseUp(self.id)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        painter.drawRect(self.rect())

        painter.setPen(QPen(Qt.black, 8, Qt.SolidLine))

        # draw historical points
        for points in self.psets:
            painter.drawPolyline(*points)

        # draw current points
        if self.points:
            painter.drawPolyline(*self.points)
class SmallWidget(QWidget):
    def __init__(self, parent=None):
        super(SmallWidget, self).__init__(parent)
        self.points = []
        self.resize(28,28)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        painter.drawRect(self.rect())

        painter.setPen(Qt.black)
        for p in self.points:
            painter.drawPoint(p[0], p[1])


    def setPoints(self, arr):
        self.points = []
        for row in range(28):
            for line in range(28):
                if arr[row][line] > 0:
                    self.points.append((line, row))
        self.update()
        
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def onFirstClick(self, id):
        for pad in self.pad:
            if pad.id == id:
                continue
            pad.clearPoints()

    def initUI(self):
        self.pad = [MouseWidget(0, self), MouseWidget(1, self)]
        self.pad[0].move(20,20)
        self.pad[1].move(240,20)
        self.resize(460,320)
        self.setWindowTitle('mnist')
        
        self.win = [SmallWidget(self), SmallWidget(self)]
        self.win[0].move(110,230)
        self.win[1].move(330,230)

        self.lb = [QLabel(self), QLabel(self)]
        self.lb[0].move(150,230)
        self.lb[1].move(370,230)

    def onMouseUp(self, id):
        npx = np.zeros((28,28))
        pixmap = self.pad[id].grab().toImage().scaled(28,28)
        
        for _x in range(28):
            for _y in range(28):
                c = pixmap.pixel(_x,_y)
                #colors = QColor(c).getRgbF()
                if qRed(c) == 0:
                    for ix in range(-1,2):
                        for iy in range(-1,2):
                            x = _x + ix
                            y = _y + iy
                            if x < 0 or x >= 28 or y < 0 or y >= 28:
                                continue
                            npx[y][x] = 1

        npy = classifier.calc(np.reshape(npx.astype(np.float32), (1,1,28,28))).data
        self.lb[id].setText('<h1>' + np.argmax(npy).astype('str') + '</h1>')
        self.lb[id].adjustSize()
        self.win[id].setPoints(npx)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
