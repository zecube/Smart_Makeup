import cv2
import numpy as np
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color
from imutils import face_utils
import numpy as np
import dlib
from skimage import io
from itertools import compress
from sklearn.cluster import KMeans
import os


class DetectFace:
    def __init__(self, image):
        # initialize dlib's face detector (HOG-based)
        # and then create the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        #face detection part
        self.img = image
        #if self.img.shape[0]>500:
        #    self.img = cv2.resize(self.img, dsize=(0,0), fx=0.8, fy=0.8)

        # init face parts
        self.right_eyebrow = []
        self.left_eyebrow = []
        self.right_eye = []
        self.left_eye = []
        self.left_cheek = []
        self.right_cheek = []

        # detect the face parts and set the variables
        self.detect_face_part()


    # return type : np.array
    def detect_face_part(self):
        face_parts = [[],[],[],[],[],[],[],[]]
        # detect faces in the grayscale image
        rect = self.detector(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 1)[0]

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), rect)
        shape = face_utils.shape_to_np(shape)
        # print(shape)
        idx = 0
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            face_parts[idx] = shape[i:j]
            idx += 1

        self.left_cheek = self.img[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]
        self.right_cheek = self.img[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.IMAGE = img.reshape((img.shape[0] * img.shape[1], 3))

        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(self.IMAGE)

        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # Return a list in order of color that appeared most often.
    def getHistogram(self):
        numLabels = np.arange(0, self.CLUSTERS+1)
        #create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        colors = self.COLORS
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]
        for i in range(self.CLUSTERS):
            colors[i] = colors[i].astype(int)
        # Blue mask 제거
        fil = [colors[i][2] < 250 and colors[i][0] > 10 for i in range(self.CLUSTERS)]
        colors = list(compress(colors, fil))
        return colors, hist


class Skin_ton:
    def __init__(self, lab_b, hsv_s):
        self.lab_b = lab_b
        self.hsv_s = hsv_s

    def is_warm(self):
        '''
        파라미터 lab_b = skin_b
        질의색상 lab_b값에서 warm의 lab_b, cool의 lab_b값 간의 거리를
        각각 계산하여 warm이 가까우면 1, 반대 경우 0 리턴
        '''
        # skin lab_b 기준치
        warm_b_std = 11.6518
        cool_b_std = 4.64255

        # 기준치와의 거리
        warm_dist = abs(self.lab_b - warm_b_std)

        cool_dist = abs(self.lab_b - cool_b_std)

        if(warm_dist <= cool_dist):
            return 1 #warm
        else:
            return 0 #cool

    def is_spr(self):
        '''
        파라미터 hsv_s = skin_s
        질의색상 hsv_s값에서 spring의 hsv_s, fall의 hsv_s값 간의 거리를
        각각 계산하여 spring이 가까우면 1, 반대 경우 0 리턴
        '''
        #skin hsv_s 기준치
        spr_s_std = 18.59296
        fal_s_std = 27.13987

        #기준치와의 거리 계산
        spr_dist = abs(self.hsv_s - spr_s_std)
        fal_dist = abs(self.hsv_s - fal_s_std)

        if(spr_dist <= fal_dist):
            return 1 #spring
        else:
            return 0 #fall

    def is_smr(self):
        '''
        파라미터 hsv_s = skin_s
        질의색상 hsv_s값에서 summer의 hsv_s, winter의 hsv_s값 간의 거리를
        각각 계산하여 summer가 가까우면 1, 반대 경우 0 리턴
        '''
        #skin hsv_s 기준치
        smr_s_std = 12.5
        wnt_s_std = 16.73913

        #기준치와의 거리 계산
        smr_dist = abs(self.hsv_s - smr_s_std)
        wnt_dist = abs(self.hsv_s - wnt_s_std)

        if(smr_dist <= wnt_dist):
            return 1 #summer
        else:
            return 0 #winter


def analysis(imgpath):
    #######################################
    #           Face detection            #
    #######################################
    
    df = DetectFace(imgpath)
    face = [df.left_cheek, df.right_cheek]

    #######################################
    #         Get Dominant Colors         #
    #######################################
    temp = []
    clusters = 4
    for f in face:
        dc = DominantColors(f, clusters)
        face_part_color, _ = dc.getHistogram()
        temp.append(np.array(face_part_color[0]))
    cheek = np.mean([temp[0], temp[1]], axis=0)

    color = cheek

    bbc = open("summer_bbc.txt", 'a')


    sc = open("summer_sc.txt", 'a')
    vc = open("summer_vc.txt", 'a')

    
    rgb = sRGBColor(color[0], color[1], color[2], is_upscaled=True)
    lab = convert_color(rgb, LabColor, through_rgb_type=sRGBColor)
    hsv = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor)
    Lab_b = float(format(lab.lab_b,".2f"))
    hsv_s = float(format(hsv.hsv_s,".2f"))*100

    bbc.write(str((float(format(lab.lab_b,".4f"))))+', ')
    sc.write(str(float(format(hsv.hsv_s,".7f"))*100)+', ')
    vc.write(str(float(format(hsv.hsv_v,".7f"))*100)+', ')

    bbc.close()
    sc.close()
    vc.close()

import os
dirpath = 'summer'
imgdir = os.listdir(dirpath)

for imgpath in imgdir:
    print(os.path.join(dirpath,imgpath))
    img = cv2.imread(os.path.join(dirpath,imgpath))
    analysis(img)