import cv2
import numpy as np
import math

from PyQt5.QtWidgets import QCheckBox, QLineEdit, QSlider, QSpinBox
from PyQt5.QtCore import Qt, pyqtSignal, QObject

import matplotlib.pyplot as plt

class BaseTargeter(QObject):

    coords = []

    changed = pyqtSignal()
    new_spot_size = pyqtSignal(str)

    def __init__(self, contours, base_image, binary_image):
        QObject.__init__(self, None)
        self._contours = contours
        self._base_image = base_image
        self._binary_image = binary_image
        self.settings = {}

    def max_spot_size(self):
        """
        Computes the maximum spot size given a set of contours 
        """
        pass

    def image_with_spots(self, image, spotsize = 30):
        if len(image.shape) == 3:
            image_copy = image.copy()
        else:
            image_copy = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        for c in self.coords:
            try: 
                x = int(c[0])
                y = int(c[1])
                cv2.circle(image_copy, (x,y), spotsize, (255, 0, 0), -1)
            except TypeError:
                pass

        return image_copy

    def set_setting(self, setting_name, setting_value):
        self.settings[setting_name]['value'] = self.settings[setting_name]['type'](setting_value)
        self.coords = []
        self.changed.emit()

    def get_image(self):
        self.compute_spots()
        return self.image_with_spots(self._base_image, spotsize=self.spot_size)

    def calculate_auto_spot_size(self):
        min_size = 1e10
        for contour in self._contours:
            try:
                el = cv2.fitEllipse(contour)
            except:
                continue
            this_min = min(el[1])
            if this_min < min_size:
                min_size = this_min
        
        min_size = round(min_size)
        self.new_spot_size.emit(str(min_size))

        return min_size

    def setup_spot_size(self):
        if self.settings['auto_spot']['value']:
            self.spot_size = self.calculate_auto_spot_size()
        else:
            self.spot_size = self.settings['spot_size']['value']

        print('Using spot size = %i'%(self.spot_size))

class CoreTargeter(BaseTargeter):

    name = 'Cores'

    settings = {
        'auto_spot': {
            'type': bool,
            'control': QCheckBox,
            'label': 'Automatic spot size',
            'value': True,
            'setup': []
        },
        'spot_size': {
            'type': int,
            'control': QLineEdit,
            'label': 'Spot size',
            'value': 30,
            'setup': []
        }
    }

    def __init__(self, contours, base_image, binary_image):        
        self.settings['spot_size']['setup'] = [lambda w: self.new_spot_size.connect(w.setText)]
        BaseTargeter.__init__(self, contours, base_image, binary_image)

    def compute_spots(self):
        self.setup_spot_size()

        dist = cv2.distanceTransform(self._binary_image, cv2.DIST_L2, 3)
        kernel = np.ones((75, 75), np.uint8)
        distdil = cv2.dilate(dist, kernel)
        localmax = (distdil == dist)
        localmax = localmax * dist
        _, localmaxb = cv2.threshold(localmax, 1, 1, cv2.THRESH_BINARY)
        localmaxb = np.array(localmaxb).astype(np.uint8)
        spotlocs = cv2.findNonZero(localmaxb)
        sizelist = [dist[spot[0][1]][spot[0][0]] for spot in spotlocs]
        sizelist = np.sort(sizelist)

        grainspots = {}
        for spot in spotlocs:
            x = spot[0][0]
            y = spot[0][1]

            for (cntindex, cnt) in enumerate(self._contours):
                if cntindex not in grainspots and cv2.pointPolygonTest(cnt, (x,y), True) > self.spot_size/2:
                    grainspots[cntindex] = (x,y)
                    break

        for key in grainspots:
            self.coords.append(grainspots[key])
        
        return self.coords
        

class MomentsTargeter(BaseTargeter):

    name = 'Moments'
    settings = {
        'auto_spot': {
            'type': bool,
            'control': QCheckBox,
            'label': 'Automatic spot size',
            'value': True,
            'setup': []
        },
        'spot_size': {
            'type': int,
            'control': QLineEdit,
            'label': 'Spot size',
            'value': 30,
            'setup': []
        }
    }

    def __init__(self, contours, base_image, binary_image):        
        BaseTargeter.__init__(self, contours, base_image, binary_image)

    def compute_spots(self):
        self.setup_spot_size()
        self.coords = []
        if self._contours is None:
            return
        for c in self._contours:
            if cv2.contourArea(c) < math.pi*(self.spot_size/2.0)**2:
                continue
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])    
            self.coords.append( [cX, cY] )

        return self.coords

class RimTargeter(BaseTargeter):

    name = 'Rims'

    settings = {
        'auto_spot': {
            'type': bool,
            'control': QCheckBox,
            'label': 'Automatic spot size',
            'value': False,
            'setup': []
        },
        'spot_size': {
            'type': int,
            'control': QSpinBox,
            'label': 'Spot size',
            'value': 30,
            'setup': [lambda w: w.setRange(5, 500)]
        }
    }

    def __init__(self, contours, base_image, binary_image):        
        BaseTargeter.__init__(self, contours, base_image, binary_image)

    def compute_spots(self):
        self.setup_spot_size()

        dist = cv2.distanceTransform(self._binary_image, cv2.DIST_L2, 3)
        
        inset = 10
        rim = cv2.inRange(dist, math.floor(self.spot_size/2.0)+inset, math.ceil(self.spot_size/2.0 + 0.5)+inset)         
        
        spotlocs = cv2.findNonZero(rim)
        spotlocs = spotlocs.reshape( (spotlocs.shape[0], spotlocs.shape[2]) )
        spotlocs = spotlocs[spotlocs[:,0].argsort()]

        grainspots = {}
        for spot in spotlocs:
            x = spot[0]
            y = spot[1]

            for (cntindex, cnt) in enumerate(self._contours):                
                if cntindex not in grainspots and cv2.pointPolygonTest(cnt, (x,y), False) == 1:
                    grainspots[cntindex] = (x,y)
                    break

        for key in grainspots:
            self.coords.append(grainspots[key])
        
        return self.coords

class SimpleBlobTargeter(BaseTargeter):

    name = 'Simple Blobs'

    def __init__(self, contours, base_image, binary_image):        
        BaseTargeter.__init__(self, contours, base_image, binary_image)

    def compute_spots(self):
        self.setup_spot_size()

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 1200
        params.maxArea = 1e6
        params.filterByCircularity = False
        params.minCircularity = 0.5
        params.filterByConvexity = False
        params.minConvexity = 0.9
        params.filterByInertia = False
        params.minInertiaRatio = 0.7
        params.minDistBetweenBlobs = 10
        params.filterByColor = False
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(self.binary_image)
        self.coords = [k.pt for k in keypoints]
        return self.coords
               

