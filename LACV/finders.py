import cv2
import numpy as np
from functools import partial

from PyQt5.QtWidgets import QCheckBox, QLineEdit, QSlider, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal, QObject

class BaseFinder(QObject):
    
    _binary_image = None

    changed = pyqtSignal()

    def __init__(self, input_image):
        QObject.__init__(self, None)
        self._input_image = input_image
        self.settings = {}

    def make_binary(self, image):
        """
        Takes in an image and returns grain boundery polygons.
        """
        pass

    def boundaries(self, base_image):
        """
        Given a base image returns an image with boundaries drawn and the accepted contours.
        """

        if self._binary_image is None:
            return base_image

        contours, hierarchy = cv2.findContours(self._binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        ex = np.zeros([100, 100], dtype=np.uint8)
        exim = cv2.ellipse(ex, (50, 50), (60, 30), 0, 0, 360, 1, 1)
        exc, _ = cv2.findContours(exim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        exc = exc[0]
        img_with_boundaries = base_image.copy()
        img_with_boundaries[:] = (255, 255, 255)
        
        good_contours = []
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] <= 0:
                try:
                    area = cv2.contourArea(cnt)
                    match = cv2.matchShapes(exc, cnt, cv2.CONTOURS_MATCH_I2, 0)            
                    circularity = 4*3.14*cv2.contourArea(cnt) / (cv2.arcLength(cnt, True)**2)
                    convexity = cv2.contourArea(cnt)/cv2.contourArea(cv2.convexHull(cnt))
                except:
                    area = 0
                    match = 0
                    circularity = 0
                    convexity = 0


                print("%f\t%f\t%f\t%f"%(match, circularity, convexity, area))

                if area < 1000:
                    continue

                #if match < 0.01 or circularity < 0.75 or convexity < 0.8:
                #    continue
                color = np.random.randint(0, 255, (3)).tolist()
                try:
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    #cv2.putText(img_with_boundaries, "%f, %f, %f"%(match, circularity, convexity), (cX, cY), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                except:
                    pass
                cv2.drawContours(img_with_boundaries, [cnt], 0, color, 8)
                good_contours.append(cnt)

        #cv2.imwrite('/Users/japetrus/Desktop/cont.png', img_with_boundaries)
        self._contours = good_contours
        return (img_with_boundaries, good_contours)

    def contours(self):
        return self._contours

    def binary_image(self):
        return self._binary_image

    def set_setting(self, setting_name, setting_value):
        self.settings[setting_name]['value'] = self.settings[setting_name]['type'](setting_value)
        self.changed.emit()

    def get_image(self):
        self.make_binary()
        return self.boundaries(self._input_image)[0]

class ThresholdFinder(BaseFinder):
    """
    A simple finder that 
    """

    name = "Smoothed Thresholding"

    settings = {
        'lower': {
            'type': int,
            'control': QLineEdit,
            'label': 'Lower',
            'value': 170,
            'setup': []
        },
        'upper': {
            'type': int,
            'control': QLineEdit,
            'label': 'Upper',
            'value': 230,
            'setup': []
        },
        'smooth': {
            'type': bool,
            'control': QCheckBox,
            'label': 'Smooth',
            'value': True,
            'setup': []
        },
        'smooth_size': {
            'type': int,
            'control': partial(QSlider, Qt.Horizontal),
            'label': 'Smoothing size',
            'value': 11,
            'setup': [lambda w: w.setMinimum(3), lambda w: w.setMaximum(101)]
        },
        'open': {
            'type': bool,
            'control': QCheckBox,
            'label': 'Open',
            'value': True,
            'setup': []            
        },
        'kernel_size': {
            'type': int,
            'control': partial(QSlider, Qt.Horizontal),
            'label': 'Opening kernel size',
            'value': 7,
            'setup': [lambda w: w.setMinimum(3), lambda w: w.setMaximum(101)]
        }
    }

    def __init__(self, input_image):
        BaseFinder.__init__(self, input_image)

    def make_binary(self):  
        imggray = cv2.cvtColor(self._input_image, cv2.COLOR_RGB2GRAY)

        if self.settings['smooth']['value'] == True:
            v = self.settings['smooth_size']['value']
            if v % 2 == 0:
                v += 1
            imggray = cv2.medianBlur(imggray, v)

        lower = self.settings['lower']['value']
        upper = self.settings['upper']['value']
        thresh = cv2.inRange(imggray, lower, upper)    
        
        if self.settings['open']['value'] == True:
            ks = self.settings['kernel_size']['value']            
            kernel = np.ones((ks, ks), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        self._binary_image = thresh
        return self._binary_image
        
class AdaptiveThresholdFinder(BaseFinder):
    """
    A finder that uses adaptive thresholding to construct
    binary image and find contours.
    """

    name = "Adaptive Thresholding"

    settings = {
        'method': {
            'type': int,
            'control': QComboBox,
            'label': 'Method',
            'value': cv2.ADAPTIVE_THRESH_MEAN_C,
            'setup': [lambda w: w.addItem("Mean", cv2.ADAPTIVE_THRESH_MEAN_C), lambda w: w.addItem("Gaussian", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)]
        },
        'block_size': {
            'type': int,
            'control': partial(QSlider, Qt.Horizontal),
            'label': 'Block size',
            'value': 21,
            'setup': [lambda w: w.setMinimum(3), lambda w: w.setMaximum(200)]
        },
        'c': {
            'type': int,
            'control': partial(QSlider, Qt.Horizontal),
            'label': 'C',
            'value': 2,
            'setup': [lambda w: w.setMinimum(-255), lambda w: w.setMaximum(255)]
        }
    }

    def __init__(self, input_image):
        BaseFinder.__init__(self, input_image)
        self.settings['block_size']['setup'][1] = lambda w: w.setMaximum(input_image.shape[0]/2.0)

    def make_binary(self):  
        imggray = cv2.cvtColor(self._input_image, cv2.COLOR_RGB2GRAY)
        imggray = cv2.medianBlur(imggray, 5)

        method = self.settings['method']['value']
        block_size = self.settings['block_size']['value']
        if block_size % 2 == 0:
            block_size += 1
        c = self.settings['c']['value']
        
        thresh = cv2.adaptiveThreshold(imggray, 255, method, cv2.THRESH_BINARY, block_size, c)
        kernel = np.ones((11, 11), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        self._binary_image = thresh
        return self._binary_image


class OtsuThresholdFinder(BaseFinder):
    """
    A finder that uses Otsu thresholding to construct
    binary image and find contours.
    """

    name = "Otsu Thresholding"

    settings = {
        'blur_size': {
            'type': int,
            'control': partial(QSlider, Qt.Horizontal),
            'label': 'Blur size',
            'value': 5,
            'setup': [lambda w: w.setMinimum(3), lambda w: w.setMaximum(200)]
        }
    }

    def __init__(self, input_image):
        BaseFinder.__init__(self, input_image)

    def make_binary(self):  
        imggray = cv2.cvtColor(self._input_image, cv2.COLOR_RGB2GRAY)
        k = self.settings['blur_size']['value']
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(imggray, (k,k), 0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        self._binary_image = th
        return self._binary_image


        