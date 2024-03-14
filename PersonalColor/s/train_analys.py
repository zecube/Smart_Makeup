import cv2
import numpy as np
import skin_tone
import face_detect
from color_extract import DominantColors
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color
from matplotlib import pyplot
import dlib
from imutils import face_utils


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
        face_parts = face_parts[1:5]
        # set the variables
        # Caution: this coordinates fits on the RESIZED image.
        self.right_eyebrow = self.extract_face_part(face_parts[2])
        #cv2.imshow("right_eyebrow", self.right_eyebrow)
        #cv2.waitKey(0)
        self.left_eyebrow = self.extract_face_part(face_parts[1])
        # Cheeks are detected by relative position to the face landmarks
        self.left_cheek = self.img[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]
        self.right_cheek = self.img[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]

        cv2.imshow('right_eyebrow',self.right_eyebrow)
        cv2.imshow('left_eyebrow',self.left_eyebrow)
        cv2.waitKey(0)

    # parameter example : self.right_eye
    # return type : image
    def extract_face_part(self, face_part_points):
        (x, y, w, h) = cv2.boundingRect(face_part_points)
        crop = self.img[y:y+h, x:x+w]
        adj_points = np.array([np.array([p[0]-x, p[1]-y]) for p in face_part_points])

        # Create an mask
        mask = np.zeros((crop.shape[0], crop.shape[1]))
        cv2.fillConvexPoly(mask, adj_points, 1)
        mask = mask.astype(np.bool_)
        crop[np.logical_not(mask)] = [255, 0, 0]

        return crop


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
        #dc.plotHistogram()
        temp.append(np.array(face_part_color[0]))
    cheek = np.mean([temp[0], temp[1]], axis=0)
    # eyebrow = np.mean([temp[2], temp[3]], axis=0)
    # eye = np.mean([temp[4], temp[5]], axis=0)

    rgb, lab, hsv = [[],[],[]], [[],[],[]], [[],[],[]]
    '''
    rgb = [[r값 : cheek, eyebrow, eye], [g값: c, e, e], [b값 : c, e, e]]
    '''

    rc = open("winter_rc.txt", 'a')
    reb = open("winter_reb.txt", 'a')
    re = open("winter_re.txt", 'a')

    gc = open("winter_gc.txt", 'a')
    geb = open("winter_geb.txt", 'a')
    ge = open("winter_ge.txt", 'a')

    bc = open("winter_bc.txt", 'a')
    beb = open("winter_beb.txt", 'a')
    be = open("winter_be.txt", 'a')

    llc = open("winter_llc.txt", 'a')
    lleb = open("winter_lleb.txt", 'a')
    lle = open("winter_lle.txt", 'a')

    aac = open("winter_aac.txt", 'a')
    aaeb = open("winter_aaeb.txt", 'a')
    aae = open("winter_aae.txt", 'a')

    bbc = open("winter_bbc.txt", 'a')
    bbeb = open("winter_bbeb.txt", 'a')
    bbe = open("winter_bbe.txt", 'a')

    hc = open("winter_hc.txt", 'a')
    heb = open("winter_heb.txt", 'a')
    he = open("winter_he.txt", 'a')

    sc = open("winter_sc.txt", 'a')
    seb = open("winter_seb.txt", 'a')
    se = open("winter_se.txt", 'a')

    vc = open("winter_vc.txt", 'a')
    veb = open("winter_veb.txt", 'a')
    ve = open("winter_ve.txt", 'a')

    color = [cheek]

    rc.write(str(color[0][0])+', ')
    reb.write(str(color[1][0])+', ')
    re.write(str(color[2][0])+', ')

    gc.write(str(color[0][1])+', ')
    geb.write(str(color[1][1])+', ')
    ge.write(str(color[2][1])+', ')

    bc.write(str(color[0][2])+', ')
    beb.write(str(color[1][2])+', ')
    be.write(str(color[2][2])+', ')

    rgb_c = sRGBColor(color[0], color[1], color[2], is_upscaled=True)
    # rgb_eb = sRGBColor(color[1][0], color[1][1], color[1][2], is_upscaled=True)
    # rgb_e = sRGBColor(color[2][0], color[2][1], color[2][2], is_upscaled=True)

    lab = [convert_color(rgb_c, LabColor, through_rgb_type=sRGBColor)]
    hsv = [convert_color(rgb_c, HSVColor, through_rgb_type=sRGBColor)]

    llc.write(str((float(format(lab[0].lab_l,".4f"))))+', ')
    # lleb.write(str((float(format(lab[1].lab_l,".4f"))))+', ')
    # lle.write(str((float(format(lab[2].lab_l,".4f"))))+', ')

    aac.write(str((float(format(lab[0].lab_a,".4f"))))+', ')
    # aaeb.write(str((float(format(lab[1].lab_a,".4f"))))+', ')
    # aae.write(str((float(format(lab[2].lab_a,".4f"))))+', ')

    bbc.write(str((float(format(lab[0].lab_b,".4f"))))+', ')
    # bbeb.write(str((float(format(lab[1].lab_b,".4f"))))+', ')
    # bbe.write(str((float(format(lab[2].lab_b,".4f"))))+', ')

    hc.write(str((float(format(hsv[0].hsv_h,".4f"))))+', ')
    # heb.write(str((float(format(hsv[1].hsv_h,".4f"))))+', ')
    # he.write(str((float(format(hsv[2].hsv_h,".4f"))))+', ')

    sc.write(str(float(format(hsv[0].hsv_s,".7f"))*100)+', ')
    # seb.write(str(float(format(hsv[1].hsv_s,".7f"))*100)+', ')
    # se.write(str(float(format(hsv[2].hsv_s,".7f"))*100)+', ')

    vc.write(str(float(format(hsv[0].hsv_v,".7f"))*100)+', ')
    # veb.write(str(float(format(hsv[1].hsv_v,".7f"))*100)+', ')
    # ve.write(str(float(format(hsv[2].hsv_v,".7f"))*100)+', ')


    rc.close()
    reb.close()
    re.close()

    gc.close()
    geb.close()
    ge.close()

    bc.close()
    beb.close()
    be.close()

    llc.close()
    lleb.close()
    lle.close()

    aac.close()
    aaeb.close()
    aae.close()

    bbc.close()
    bbeb.close()
    bbe.close()

    hc.close()
    heb.close()
    he.close()

    sc.close()
    seb.close()
    se.close()

    vc.close()
    veb.close()
    ve.close()

    #######################################
    #      Plot rgb, lab, hsv values      #
    #######################################


import os
dirpath = '../res/train/spring'
imgdir = os.listdir(dirpath)

for imgpath in imgdir:
    print(os.path.join(dirpath,imgpath))
    analysis(os.path.join(dirpath,imgpath))