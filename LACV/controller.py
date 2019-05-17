import os
import cv2
import xml.etree.ElementTree as ET
from math import cos, sin, radians
import numpy as np

from .finders import ThresholdFinder, AdaptiveThresholdFinder, OtsuThresholdFinder
from .targeters import CoreTargeter, RimTargeter, MomentsTargeter, SimpleBlobTargeter
from .generators import ChromiumGenerator, GeoStarGenerator

from PyQt5.QtGui import QTransform, QPolygonF
from PyQt5.QtCore import QPointF

class LACVController:
    
    finders = [ThresholdFinder, AdaptiveThresholdFinder, OtsuThresholdFinder]
    targeters = [CoreTargeter, RimTargeter, MomentsTargeter, SimpleBlobTargeter]
    generators = [ChromiumGenerator, GeoStarGenerator]

    finder = None
    targeter = None
    generator = None

    global_finder_settings = {
        'area': {
            'enabled': True,
            'min': 100,
            'max': 1e6
        },
        'circularity': {
            'enabled': True,
            'min': 0,
            'max': 2
        },
        'convexity': {
            'enabled': False,
            'min': 0,
            'max': 2
        },
        'match': {
            'enabled': True,
            'min': 0,
            'max': 2
        }
    }

    def set_source(self, source):
        print('source=%s' % source)

        # Find complement
        source_dir = os.path.dirname(source)
        image_file = ""
        align_file = ""
        print('source_dir=%s' % source_dir)

        if source.lower().endswith(".align"):
            print("given an align file")
            align_file = os.path.basename(source)
            file_root = align_file.split(".")[0]

            for file in os.listdir(source_dir):
                print(file)
                if file == align_file:
                    continue
                elif file.startswith(file_root) and file.split(".")[1] in ['bmp', 'jpg', 'png', 'tiff']:
                    image_file = file
                    break
        else:
            print("given an image file")
            image_file = os.path.basename(source)
            file_root = image_file.split(".")[0]

            for file in os.listdir(source_dir):
                print(file)
                if file == image_file:
                    continue
                elif file.startswith(file_root) and file.split(".")[1].lower() == "align":
                    align_file = file
                    break


        print('align file=%s'%align_file)
        print('image file=%s'%image_file)

        if not align_file or not image_file:
            print('Could not find the image/align pair... abort!')
            self._source_image = None
            return

        print('loading image file:%s'%os.path.join(source_dir, image_file))
        self._source_image = cv2.imread(os.path.join(source_dir, image_file))
        #self._source_image = noisy('gauss', self._source_image).astype(np.uint8)

        print('processing align file:%s'%os.path.join(source_dir, align_file))
        align_xml = ET.parse(os.path.join(source_dir, align_file))
        align_root = align_xml.getroot()
        align = align_root.find('Alignment')
        self.align_rotation = float(align.find('Rotation').text)
        self.align_center = [float(x) for x in align.find('Center').text.split(',')]
        self.align_size = [float(x) for x in align.find('Size').text.split(',')]

        print('microns per pixel = %f'%(self.microns_per_pixel()))

        xc, yc = self.align_center[0], self.align_center[1]
        xmin, ymin = xc - self.align_size[0]/2.0, yc - self.align_size[1]/2.0
        xmax, ymax = xc + self.align_size[0]/2.0, yc + self.align_size[1]/2.0
        r = -self.align_rotation

        def rot(x, y):
            xp = xc + (x - xc)*cos(radians(r)) - (y-yc)*sin(radians(r))
            yp = yc + (x - xc)*sin(radians(r)) - (y-yc)*cos(radians(r))
            return xp, yp
        
        src = np.float32([ 
            [0, 0],
            [0, self._source_image.shape[0]],
            [self._source_image.shape[1], self._source_image.shape[0]]
        ])
        dst = np.float32([
            rot(xmin, ymin),
            rot(xmin, ymax),
            rot(xmax, ymax)
        ])
        self.transform = cv2.getAffineTransform(src, dst)


    def microns_per_pixel(self):
        return np.array( [self.align_size[0]/self._source_image.shape[1], self.align_size[1]/self._source_image.shape[0] ]).mean()


    def coords_in_image_to_cellspace(self, coords):
        x, y = coords

        points = np.array([ coords ])
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])        

        cs = self.transform.dot(points_ones.T).T
        return cs[0]


    def source_image(self):
        return self._source_image


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 20
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
