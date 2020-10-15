from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QToolButton, QHBoxLayout, QVBoxLayout, QGroupBox, \
    QPushButton, QSizePolicy, QComboBox, QGridLayout, QFileDialog, QLineEdit, QCheckBox, QSlider, QSpinBox, \
    QTabBar, QTabWidget, QMainWindow, QMenuBar, QMenu, QAction, QActionGroup, qApp, QScrollArea, QScrollBar, \
    QGridLayout, QDialog, QDialogButtonBox

import matplotlib.pyplot as plt
import cv2
import qtawesome as qta
from functools import partial
import os

from .finders import BaseFinder
from .targeters import BaseTargeter
from .generators import BaseGenerator


class ModuleWidget(QWidget):
    
    def __init__(self, module=None, parent=None):
        QWidget.__init__(self, parent)
        self._module = module
        if module:            
            self._module.changed.connect(self.update_image)
        
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(3, 3, 3, 3)

        self._image_widget = CVImageWidget(self)
        self.layout().addWidget(self.create_settings_widget())
        self.layout().addWidget(self._image_widget)

        self.update_image()       

    def setModule(self, module):
        self._module = module
        self._module.changed.connect(self.update_image)
        self.layout().itemAt(0).widget().setParent(None)
        self.layout().insertWidget(0, self.create_settings_widget())
        self.update_image()        

    def create_settings_widget(self):

        if self._module is None:
            return QLabel('Please select a module from the menu first.')


        settings = self._module.settings

        w = QWidget()
        w.setLayout(QHBoxLayout())
        w.layout().setContentsMargins(3, 3, 3, 3)
        w.layout().setSpacing(3)

        for sk in settings.keys():
            l = QLabel(settings[sk]['label'] + ":", w)
            control = settings[sk]['control'](w)
            for setup_lambda in settings[sk]['setup']:
                setup_lambda(control)

            w.layout().addWidget(l)
            w.layout().addWidget(control)

            spacer = QWidget(w)
            spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
            spacer.setFixedWidth(12)
            w.layout().addWidget(spacer)

            control_t = settings[sk]['control']

            if control_t == QLineEdit:
                control.setText(str(settings[sk]['value']))
                control.textEdited.connect(partial(self._module.set_setting, sk))                
            elif control_t == QCheckBox:
                control.setChecked(settings[sk]['value'])
                control.toggled.connect(partial(self._module.set_setting, sk))                
            elif hasattr(control_t, 'func') and control_t.func == QSlider:
                control.setValue(settings[sk]['value'])
                control.valueChanged.connect(partial(self._module.set_setting, sk))
            elif control_t == QComboBox:
                i = control.findData(settings[sk]['value'])
                control.setCurrentIndex(i)
                control.currentIndexChanged.connect(partial(self._module.set_setting, sk))
            elif control_t == QSpinBox:
                control.setValue(settings[sk]['value'])
                control.valueChanged.connect(partial(self._module.set_setting, sk))
            else:
                print('Unhandled type: %s'%(control_t))

        return w

    def update_image(self):
        if self._module:    
            self._image_widget.setImage(self._module.get_image())


class CVImageWidget(QWidget):

    image = None

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setVisible(False)

        self.imageLabel = QLabel(self)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        
        self.scrollArea.setWidget(self.imageLabel)        
        self.layout().addWidget(self.scrollArea)

        self.setFocusPolicy(Qt.ClickFocus)
        self.setFocus(Qt.MouseFocusReason)


    def setImage(self, cvimage):
        s = cvimage.shape
        if len(s) == 3:
            height, width, colors = cvimage.shape
            bytesPerLine = 3 * width
            img_format = QImage.Format_RGB888
        else:
            height, width = cvimage.shape
            colors = 0
            bytesPerLine = width
            img_format = QImage.Format_Grayscale8
        
        self.image = QImage(cvimage.data, width, height, bytesPerLine, img_format)
        self.scaleFactor = 1
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
        self.scrollArea.setVisible(True)        
        self.imageLabel.adjustSize()
        self.update()

    def scaleImage(self, factor):
        print('scaleFactor = %f'%self.scaleFactor)
        if (self.scaleFactor > 3 and factor > 1) or (self.scaleFactor < 0.1 and factor < 1):
            return

        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

    def adjustScrollBar(self, scrollbar, factor):
        scrollbar.setValue(int(factor * scrollbar.value() + ((factor - 1)* scrollbar.pageStep()/2)))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Equal and event.modifiers() == Qt.ControlModifier:
            self.scaleImage(1.2)
        elif event.key() == Qt.Key_Minus and event.modifiers() == Qt.ControlModifier:
            self.scaleImage(0.8)

        QWidget.keyPressEvent(self, event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        full_action = QAction('100 %', self)
        full_action.triggered.connect(self.normalSize)
        menu.addAction(full_action)

        width_action = QAction('Fit width', self)
        width_action.triggered.connect(lambda: self.fit(Qt.Horizontal))
        menu.addAction(width_action)

        height_action = QAction('Fit height', self)
        height_action.triggered.connect(lambda: self.fit(Qt.Vertical))
        menu.addAction(height_action)

        save_action = QAction('Save', self)
        save_action.triggered.connect(self.saveImage)
        menu.addAction(save_action)

        menu.exec(self.mapToGlobal(event.pos()))

    def normalSize(self):
        self.scaleFactor = 1
        self.scaleImage(1)

    def fit(self, orientation):
        
        imsize = self.imageLabel.pixmap().size()
        sasize = self.scrollArea.size()

        hfactor = sasize.width()/imsize.width()
        vfactor = sasize.height()/imsize.height()
        factor = hfactor if orientation == Qt.Horizontal else vfactor

        self.scaleFactor = 1
        self.scaleImage(factor)

    def saveImage(self):
        filename, _ = QFileDialog.getSaveFileName()

        if filename:
            self.image.save(filename)

class LACVWindow(QMainWindow):
    sourcePathLabel = None

    def __init__(self, lacv, parent=None):
        QMainWindow.__init__(self, parent)

        self.lacv = lacv
        self.setWindowTitle("LACV")
    
        self.sourceWidget = CVImageWidget(self)        
        self.findWidget = ModuleWidget(parent=self)
        self.targetWidget = ModuleWidget(parent=self)
        self.generateWidget = ModuleWidget(parent=self)
        tabWidget = QTabWidget(self)
        tabWidget.addTab(self.sourceWidget, qta.icon('fa.download'), "Source")
        tabWidget.addTab(self.findWidget, qta.icon('fa.search'), "Find")
        tabWidget.addTab(self.targetWidget, qta.icon('fa.crosshairs'), "Target")
        tabWidget.addTab(self.generateWidget, qta.icon('fa.upload'), "Generate")

        self.createMenus()

        self.setCentralWidget(tabWidget)

        self.resize(600, 600)
        
    def createMenus(self):
        # File menu
        file_menu = self.menuBar().addMenu('File')
        
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.openSource)
        file_menu.addAction(open_action)

        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(qApp.quit)
        file_menu.addAction(quit_action)

        modules_dict = {
            'Finder': self.lacv.finders, 
            'Targeter': self.lacv.targeters,
            'Generator': self.lacv.generators
        }

        for name, modules in modules_dict.items():
            menu = self.menuBar().addMenu(name)
            menu.setObjectName(name+'_menu')
            group = QActionGroup(self)
            for m in modules:
                action = QAction(m.name, self)
                action.setCheckable(True)
                action.triggered.connect(partial(self.setModule, m))
                menu.addAction(action)
                group.addAction(action)

        finder_menu = self.menuBar().findChildren(QMenu, 'Finder_menu')[0]
        finder_menu.addSeparator()

        finder_settings_action = QAction('Global settings', self)
        finder_settings_action.triggered.connect(self.showFinderSettings)
        finder_menu.addAction(finder_settings_action)
                

    def showFinderSettings(self):
        s = QDialog(self)
        s.setWindowTitle('Global finder settings')
        l = QGridLayout()
        l.setContentsMargins(3, 3, 3, 3)
        s.setLayout(QVBoxLayout())
        s.layout().setContentsMargins(3, 3, 3, 3)
        
        l.addWidget(QLabel('Filter'), 0, 0, Qt.AlignLeft)
        l.addWidget(QLabel('Enabled'), 0, 1, Qt.AlignHCenter)
        l.addWidget(QLabel('Minimum'), 0, 2, Qt.AlignHCenter)
        l.addWidget(QLabel('Maximum'), 0, 3, Qt.AlignHCenter)

        orig = self.lacv.global_finder_settings 
        d = orig.copy()

        def store_setting(name, param, value):
            d[name][param] = value

        row = 1
        for setting_name, setting_dict in d.items():
            l.addWidget(QLabel(setting_name.title()), row, 0)
            cb = QCheckBox(s)
            cb.setChecked(setting_dict['enabled'])
            cb.toggled.connect(partial(store_setting, setting_name, 'enabled'))
            l.addWidget(cb, row, 1)
            lower = QLineEdit(s)
            lower.setText(str(setting_dict['min']))
            lower.textEdited.connect(partial(store_setting, setting_name, 'min'))
            l.addWidget(lower, row, 2)
            upper = QLineEdit(s)
            upper.setText(str(setting_dict['max']))
            upper.textEdited.connect(partial(store_setting, setting_name, 'max'))
            l.addWidget(upper, row, 3)      
            row += 1      

        s.layout().addLayout(l)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, s)
        s.layout().addWidget(bb)
        bb.accepted.connect(s.accept)
        bb.rejected.connect(s.reject)

        if s.exec() == QDialog.Accepted:
            self.lacv.global_finder_settings = d
            self.lacv.finder.changed.emit()
            

    def setModule(self, m):
        if issubclass(m, BaseFinder):
            self.lacv.finder = m(self.lacv.source_image())
            self.findWidget.setModule(self.lacv.finder)
        elif issubclass(m, BaseTargeter):
            self.lacv.targeter = m(self.lacv.finder.contours(), self.lacv.source_image(), self.lacv.finder.binary_image())
            self.targetWidget.setModule(self.lacv.targeter)
        elif issubclass(m, BaseGenerator):
            self.lacv.generator = m()
            self.generateWidget.setModule(self.lacv.generator)


    def openSource(self):
        sourcePath, _ = QFileDialog.getOpenFileName(
            filter="Align files (*.Align);;Image files (*.bmp;*.jpg;*.png;*.tiff)")

        if len(sourcePath) < 1:
            print("No file selected")
            return

        self.lacv.set_source(sourcePath)
        if self.lacv.source_image is None:
            print("There was no source image. ======")
            return

        self.sourceWidget.setImage(self.lacv.source_image())
        
        hist = cv2.calcHist([self.lacv.source_image()], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
