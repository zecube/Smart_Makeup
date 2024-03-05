#pip install -r requirements.txt
import sys
import typing
from PyQt5.QtCore import *
from PyQt5.QtCore import QObject
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtTest import *
from animated_toggle import AnimatedToggle
from threading import Thread
import threading
from pynput import keyboard

import datetime
from pyowm import OWM #pip install pyowm  [Weather api] [https://github.com/csparpa/pyowm/]
import googlemaps, requests #pip install -U googlemaps      [Geocoding API,Geolocation API] [Google Cloud]
import json
from skimage import exposure
from skimage.exposure import match_histograms
import pandas as pd
import cv2
import numpy as np
import matplotlib.pylab as plt
import personal_color
import mediapipe as mp

import utils
#==========머리 마스킹 ========
from HairSegmentation import HairSegmentation
from PIL import Image
from keras.models import load_model
#=========헬스 케어 ============
import HealthCare_stress_and_melanoma as health
#========파이어 베이스==========
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import include_wishlist as wish
import Aircleaner_power as airpower



QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)     #고해상도 설정
df = pd.read_excel("savedata.xlsx")

#Firebase database 인증 및 앱 초기화
cred = credentials.Certificate('FireBase/ex2305-firebase-adminsdk-c7ei4-8835fc6f84.json') ## 바꿀수도 있는 줄, 파이어베이스.json키 위치
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://ex2305-default-rtdb.firebaseio.com/' ##  ## 바꿀수도 있는 줄, 파이어베이스 리얼타임 데이터 베이스 url
})

global cap
cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 세로  
lock = threading.Lock()


############################################################################################################
##################################################  AOD  ###################################################
############################################################################################################
class Display(QThread):
    change_start_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        # 가중치 파일 경로
        cascade_filename = 'models/haarcascade_frontalface_alt.xml'
        # 모델 불러오기
        self.cascade = cv2.CascadeClassifier(cascade_filename)
        self.power =True
        self.w = 0
        self.h = 0
        self.pre_detect_face = 0
        self.detect_face = 0
    def run(self):
        detect_face = False
        while self.power :
            lock.acquire()
            # cascade 얼굴 탐지 알고리즘 
            _, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = self.cascade.detectMultiScale(gray,            # 입력 이미지
                                            scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                            minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                            minSize=(20,20)  # 탐지 객체 최소 크기
                                            )        
                                            
            # 결과값 = 탐지된 객체의 경계상자 list                                                                           
            for box in results:
                # 좌표 추출       
                _, _, w, h = box
                # print(w,h)
                # 영상 출력        
                # self.change_start_signal.emit(detect_face, self.w, self.h)
                if w >= 400 and h>= 400:
                    self.detect_face = 1
                    if self.pre_detect_face == self.detect_face:
                        self.change_start_signal.emit(self.pre_detect_face)
                    else:
                        self.pre_detect_face = self.detect_face
                else:
                    self.pre_detect_face = 0
                    self.change_start_signal.emit(self.pre_detect_face)
            self.wait(5000)
            lock.release()
    def stop(self):
        self.power = False
        self.quit()
        self.wait(500)
############################################################################################################
##################################################  Keyboard  ###################################################
############################################################################################################
# class Mykeyboard(QThread):
#     cnt=0
#     running= False

#     def __init__(self):
#         super().__init__()  
  
#     def on_press(self,key):
#         try:
#             print(f'알파벳 \'{key.char}\' 눌림')
#         except AttributeError:
#             print(f'특수키 {key} 눌림')
 
#     def on_release(self,key):
#         print(f'키보드 {key} 풀림')
#         if key == keyboard.Key.esc:
#             # esc 키에서 풀림
#             return False

#     def run(self):
#         with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
#          listener.join()
############################################################################################################
########################################  MakeUp Thread  ###################################################
############################################################################################################
class MakeUpThread(QThread):
    # change_pixmap_signal = pyqtSignal(QImage)
    change_pixmap_signal = pyqtSignal(QPixmap)

    def __init__(self, skin_value = False, skin_ref_img = None, all_skin_ref_img = None, s_mix=0,
                        mouth_value = False, mouth_ref_img = None, all_mouth_ref_img = None,m_mix=0,
                        eyebrow_value = False, left_eyevrow = None, right_eyebrow = None, eb_mix = 0, e_posx =0, e_posy=0, e_length=0,
                        cheek_value = False, left_cheek = None, right_cheek = None, c_mix=0,c_posx=0, c_posy = 0, c_width = 0, c_height=0):
        super().__init__()
        self.skin_value = skin_value
        self.skin_ref_img = skin_ref_img
        self.all_skin_ref_img = all_skin_ref_img
        self.s_mix = s_mix
        self.mouth_value = mouth_value
        self.mouth_ref_img = mouth_ref_img
        self.all_mouth_ref_img = all_mouth_ref_img
        self.m_mix = m_mix
        self.eyebrow_value = eyebrow_value
        self.left_eyebrow = left_eyevrow
        self.right_eyebrow = right_eyebrow
        self.eb_mix = eb_mix
        self.e_posx = e_posx
        self.e_posy = e_posy
        self.e_length = e_length
        self.cheek_value = cheek_value
        self.left_cheek = left_cheek
        self.right_cheek = right_cheek
        self.c_mix = c_mix
        self.c_posx = c_posx
        self.c_posy = c_posy
        self.c_width = c_width
        self.c_height = c_height
        self.pause = True
        self.power = True

    def run(self):
        FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 
            323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 
            127, 162, 21, 54, 103,67, 109]

        LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
            185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 
            415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ] # 아랫입술+윗입술
        LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]   # 윗입술
        UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]    # 아래입술

        CLOSE_LIP=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185] # 닫힌입술 (입술+입술안 포함)

        LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263,   466, 388, 387, 386, 385,384, 398 ]
        LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
        LEFT_CLOWN = [357,350,349,348,347,346,340,372, 345,352,411,425,266,371,355]

        RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133,    173, 157, 158, 159, 160, 161 , 246 ]  
        RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
        RIGHT_CLOWN=[143,111,117,118,119,120,121,128, 126,142,36,205,187,123,116]

        LIPS1_up=[ 61, 96, 89, 179, 86, 316, 403, 319, 325, 291, 
                308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
                185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 
                415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
        LIPS1_down=[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                    325, 319, 403, 316, 86, 179, 89, 96]

        RIGHT_EYE_up1 = [247,30,29,27,28,56,190,  133, 173, 157, 158, 159, 160, 161 , 246, 33]
        RIGHT_EYE_up2 = [225,224,223,222,221,   133, 173, 157, 158, 159, 160, 161 , 246, 33]
        LEFT_EYE_up1 = [414,286,258,257,259,260,467,    263, 466, 388, 387, 386, 385,384, 398, 362 ]
        LEFT_EYE_up2 = [441,442,443,444,445,    263, 466, 388, 387, 386, 385,384, 398, 362 ]

        DARKCIRCLE_RIGHT= [33, 7, 163, 144, 145, 153, 154, 155, 133,  112,232,231,230,110]
        DARKCIRCLE_LEFT= [362, 382, 381, 380, 374, 373, 390, 249, 263, 339,450,451,452,341]
        
        def fillPolyTrans(img, points, color, opacity):
            """
            @param img: (mat) input image, where shape is drawn.
            @param points: list [tuples(int, int) these are the points custom shape,FillPoly
            @param color: (tuples (int, int, int)
            @param opacity:  it is transparency of image.
            @return: img(mat) image with rectangle draw.

            """
            list_to_np_array = np.array(points, dtype=np.int32)
            overlay = img.copy()  # coping the image
            cv2.polylines(overlay,[list_to_np_array],True,(0,0,0),thickness=1) # 다각형 경계
            cv2.fillPoly(overlay,[list_to_np_array], color )
            new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
            # print(points_list)
            img = new_img
            # cv2.polylines(img, [list_to_np_array], True, color,1, cv2.LINE_AA)
            return img
        # landmark detection function 
        def landmarksDetection(img, results, draw=False):
            img_height, img_width= img.shape[:2]
            # list[(x,y), (x,y)....]
            mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
            if draw :
                [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

            # returning the list of tuples for each landmarks 
            return mesh_coord

        map_face_mesh = mp.solutions.face_mesh
        face_mesh = map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5)    
        while self.power == True:
            lock.acquire()
            if self.pause:
                _, frame = cap.read()
                frame = cv2.flip(frame,1)   #반전
                frame = frame[40:1040, 460:1460]
                h,w,ch = frame.shape
                copy = frame.copy()
                mask = np.zeros_like(frame)
                results = face_mesh.process(frame)
                mouth_result = None
                skin_result = None
                    
                def draw_FACE_OVAL(mask, colors, ref_img):      #피부만 눌렸을때
                    # Inialize hair segmentation model / 머리카락 세분화 모델을 초기화합니다.
                    hair_segmentation = HairSegmentation()

                    hair_mask = hair_segmentation(frame)

                    # Get dyed frame. / 염료 처리된 프레임을 가져옵니다.

                    dyed_image = mask.copy()
                    dyed_image[:] = 255,255,255
                    
                    # Mask our dyed frame (pixels out of mask are black). / 염료 처리된 프레임에서 마스크를 적용합니다 (마스크 밖의 픽셀은 검은색).
                    dyed_hair = cv2.bitwise_or(frame, dyed_image, mask=hair_mask) ################### 이거로 #########################
                    dyed_hair =~dyed_hair

                    if results.multi_face_landmarks:
                        mesh_coords = landmarksDetection(frame, results, False)
                        #====== 머리 추가 =========================================================
                        addHair = mesh_coords[18][1] - mesh_coords[152][1]    # 18번매쉬.y좌표 - 0번매쉬.y좌표 / 18번을 [0, 14, 17, 18, 200, 199, 175] 중 1택 가능
                        
                        for i in [127, 162, 21, 54, 103,67, 109, 10, 338, 297, 332, 284, 251, 389, 356]:
                            mesh_coords[i] = (mesh_coords[i][0], mesh_coords[i][1] + addHair)
                        #==========================================================================

                        mask =fillPolyTrans(mask, [mesh_coords[p] for p in FACE_OVAL], colors, opacity=1 )

                        # 입술 뺄 시 아래 코드 사용
                        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in CLOSE_LIP], (0,0,0), opacity=1 )   # (입술+입술 내부) 포함해서 뺌
                        # 눈 뺄 시
                        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYE], (0,0,0), opacity=1 )
                        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYE], (0,0,0), opacity=1 )
                        # # 눈썹 뺄 시
                        # mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYEBROW], (0,0,0), opacity=1 )
                        # mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYEBROW], (0,0,0), opacity=1 )
                        # # 머리
                        mask = cv2.bitwise_and(mask, dyed_hair)

                        mask_copy = mask.copy
                        mask_copy =~mask
                        # #  마스킹 하기
                        masked = cv2.bitwise_or(copy, mask_copy)
                        matched = match_histograms(masked,ref_img, channel_axis= -1)

                        s_weighted_img = cv2.addWeighted(matched, self.s_mix/100, masked, 1-(self.s_mix/100), 0) # 두개의 이미지를 가중치에 따라서 다르게 보여줍니다.
                        
                        return s_weighted_img, mask
                
                def draw_LIPS(mask, colors, ref_img):
                    if results.multi_face_landmarks:
                        mesh_coords = landmarksDetection(frame, results, False)
                        mask = utils.fillPolyTrans(mask, [mesh_coords[p] for p in LIPS], colors, opacity=1)
                        mask_copy = mask.copy
                        mask_copy =~mask
                        ##마스킹 하기
                        masked = cv2.bitwise_or(copy, mask_copy)
                        # ref_img = makeup_ref_img.draw_LIPS(ref_img,utils.WHITE)
                        matched = match_histograms(masked, ref_img, channel_axis=-1)

                        m_weighted_img = cv2.addWeighted(matched, self.m_mix/100, masked, 1-(self.m_mix/100), 0) # 두개의 이미지를 가중치에 따라서 다르게 보여줍니다.
                        return m_weighted_img, mask
                
                def overlay(image, x, y, w, h, overlay_image): #대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지(4채널)
                    alpha = overlay_image[:, :, 3] #BRGA
                    mask_image = alpha / 255 # 0~ 255 -> 255로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0:완전)
                    copy = image.copy()

                    for c in range(0,3): #channel BGR
                        copy[y:y+h, x:x+w, c] = (overlay_image[:,:,c]*mask_image) + (copy[y:y+h, x:x+w, c] * (1 - mask_image))
                    return image, copy
                
                def Blur(cam):
                    #샤프닝
                    '''kernel9 = [-1, -1, -1,
                               -1, 9, -1,
                               -1, -1, -1]
                    kernel5 = [0, -1, -0,
                                -1, 5, -1,
                                0, -1, 0]'''
                    sigmax, sigmay = 10,10
                    GaussianBlur2 = cv2.GaussianBlur(cam, (1,1), sigmax, sigmay)
                    return GaussianBlur2

                #==========================================================================================================================
                #기본상태
                if (self.skin_value == False) and (self.mouth_value == False) and (self.eyebrow_value == False) and (self.cheek_value == False): 
                    output = frame.copy()
                #화장 내의 버튼이 눌린 상태
                else:
                    skin_mask = mask.copy()
                    mouth_mask = mask.copy()
                    final_mask = mask.copy()  
                    face_mask = np.zeros_like(frame)   

                    if self.mouth_value:    #입술 버튼 눌렸을 때
                        if results.multi_face_landmarks:
                            mouth_result, mask_mouth = draw_LIPS(mouth_mask, utils.WHITE, self.mouth_ref_img)
                            final_mask = cv2.bitwise_or(mask_mouth, final_mask)
                            mouth_result = cv2.bitwise_and(mouth_result, mask_mouth)
                    # else:
                    #     mouth_result = None
                    #     continue

                    if self.skin_value: #피부 버튼 눌렸을때
                        if results.multi_face_landmarks:
                            skin_result, mask_skin = draw_FACE_OVAL(skin_mask, utils.WHITE, self.skin_ref_img)
                            final_mask = cv2.bitwise_or(mask_skin, final_mask)
                            skin_result = cv2.bitwise_and(skin_result, mask_skin)
                    # else:
                    #     skin_result = None
                    #     continue
                    
                    if self.eyebrow_value:  #눈썹 버튼이 눌렸을 때
                        if results.multi_face_landmarks:
                            mesh_coords = landmarksDetection(frame, results, False)
                            
                            #-----------left
                            mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYEBROW], utils.WHITE, opacity=1 )
                            x_LEFT = [mesh_coords[p][0] for p in LEFT_EYEBROW]
                            w_LEFT = int(max(x_LEFT) - min(x_LEFT))
                            y_LEFT = [mesh_coords[p][1] for p in LEFT_EYEBROW]
                            h_LEFT = int(max(y_LEFT) - min(y_LEFT))

                            x_mid_pos_left = max(x_LEFT) - w_LEFT + self.e_posx 
                            y_mid_pos_left = max(y_LEFT) - h_LEFT + self.e_posy
                            pos_LEFT = (int(x_mid_pos_left), int(y_mid_pos_left))#(round(max(x_LEFT)-w_LEFT/2), round(max(y_LEFT)-h_LEFT/2))
                            
                            left_eyebrow = self.left_eyebrow.copy()
                            left_eyebrow = cv2.resize(left_eyebrow, (w_LEFT + self.e_length, h_LEFT))        
                            
                            #----------right
                            mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYEBROW], utils.WHITE, opacity=1 )
                            x_RIGHT = [mesh_coords[p][0] for p in RIGHT_EYEBROW]
                            w_RIGHT = int(max(x_RIGHT) - min(x_RIGHT))
                            y_RIGHT = [mesh_coords[p][1] for p in RIGHT_EYEBROW]
                            h_RIGHT = int(max(y_RIGHT) - min(y_RIGHT))

                            x_mid_pos_RIGHT= max(x_RIGHT) - w_RIGHT - self.e_posx - self.e_length 
                            y_mid_pos_RIGHT = max(y_RIGHT) - h_RIGHT + self.e_posy
                            pos_RIGHT = (int(x_mid_pos_RIGHT), int(y_mid_pos_RIGHT))

                            right_eyebrow = self.right_eyebrow.copy()
                            right_eyebrow = cv2.resize(right_eyebrow, (w_RIGHT + self.e_length, h_RIGHT))
                            #-----------------------------------------------
                            # masked = cv2.bitwise_and(copy, mask)
                            # out = cv2.bitwise_and(frame,masked)
                            
                            # eyebrow_result = cv2.bitwise_and(out, masked)

                    if self.cheek_value:    #볼터치 버튼을 눌렀을때
                        if results.multi_face_landmarks:
                            mesh_coords = landmarksDetection(frame, results, False)
                            
                            #-----------left
                            mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_CLOWN], utils.WHITE, opacity=1 )
                            c_x_LEFT = [mesh_coords[p][0] for p in LEFT_CLOWN]
                            c_w_LEFT = int(max(c_x_LEFT) - min(c_x_LEFT))
                            c_y_LEFT = [mesh_coords[p][1] for p in LEFT_CLOWN]
                            c_h_LEFT = int(max(c_y_LEFT) - min(c_y_LEFT))

                            c_x_mid_pos_left = max(c_x_LEFT) - c_w_LEFT + self.c_posx 
                            c_y_mid_pos_left = max(c_y_LEFT) - c_h_LEFT + self.c_posy
                            c_pos_LEFT = (int(c_x_mid_pos_left), int(c_y_mid_pos_left))#(round(max(x_LEFT)-w_LEFT/2), round(max(y_LEFT)-h_LEFT/2))
                            
                            left_cheek = self.left_cheek
                            left_cheek = cv2.resize(left_cheek, (c_w_LEFT + self.c_width , c_h_LEFT + self.c_height))        
                            
                            #----------right
                            mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_CLOWN], utils.WHITE, opacity=1 )
                            c_x_RIGHT = [mesh_coords[p][0] for p in RIGHT_CLOWN]
                            c_w_RIGHT = int(max(c_x_RIGHT) - min(c_x_RIGHT))
                            c_y_RIGHT = [mesh_coords[p][1] for p in RIGHT_CLOWN]
                            c_h_RIGHT = int(max(c_y_RIGHT) - min(c_y_RIGHT))

                            c_x_mid_pos_RIGHT= max(c_x_RIGHT) - c_w_RIGHT - self.c_posx - self.c_width 
                            c_y_mid_pos_RIGHT = max(c_y_RIGHT) - c_h_RIGHT + self.c_posy
                            c_pos_RIGHT = (int(c_x_mid_pos_RIGHT), int(c_y_mid_pos_RIGHT))

                            right_cheek = self.right_cheek
                            right_cheek = cv2.resize(right_cheek, (c_w_RIGHT + self.c_width , c_h_RIGHT  + self.c_height))
                            #-----------------------------------------------
                            # masked = cv2.bitwise_and(copy, mask)
                            # out = cv2.bitwise_and(frame,masked)
                            
                            # cheek_result = cv2.bitwise_and(out, masked)

                    final_mask = ~final_mask
                    final_mask = cv2.bitwise_and(frame, final_mask)
                    output = final_mask.copy()
                    
                    if mouth_result is not None:
                        output = cv2.bitwise_or(mouth_result, output)
                    if skin_result is not None:
                        output = cv2.bitwise_or(skin_result,output)

                    if self.eyebrow_value:
                        try:
                            image, copy = overlay(output, pos_LEFT[0],pos_LEFT[1], int(w_LEFT + self.e_length),int(h_LEFT), left_eyebrow)
                            output = cv2.addWeighted(copy, self.eb_mix/100, image, 1-(self.eb_mix/100), 0)
                        except:
                            continue
                        try:
                            image, copy = overlay(output, pos_RIGHT[0], pos_RIGHT[1], int(w_RIGHT + self.e_length),int(h_RIGHT), right_eyebrow)
                            output = cv2.addWeighted(copy, self.eb_mix/100, image, 1-(self.eb_mix/100), 0)
                        except:
                            continue
                    if self.cheek_value:
                        try:
                            image, copy = overlay(output, c_pos_LEFT[0],c_pos_LEFT[1], int(c_w_LEFT + self.c_width),int(c_h_LEFT + self.c_height), left_cheek)
                            output = cv2.addWeighted(copy, self.c_mix/100, image, 1-(self.c_mix/100), 0)
                        except:
                            continue

                        try:
                            image, copy = overlay(output, c_pos_RIGHT[0], c_pos_RIGHT[1], int(c_w_RIGHT + self.c_width),int(c_h_RIGHT + self.c_height), right_cheek)
                            output = cv2.addWeighted(copy, self.c_mix/100, image, 1-(self.c_mix/100), 0)
                        except:
                            continue
                
                        
                # https://stackoverflow.com/a/55468544/6622587
                #opencv는 bgr, pyqt는 rgb
                rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                p = QPixmap.fromImage(convert_to_qt_format)
                self.change_pixmap_signal.emit(p)
            else:
                continue
            lock.release()
    def stop(self):
        self.power = False
        self.quit()
        self.wait(500)
############################################################################################################
###############################  Personal Color Detect Thread  #############################################
############################################################################################################
class DetectPersonalColor(QThread):
    call_tone_value = pyqtSignal(str)
    def __init__(self):
        super().__init__()
    def run(self):
        lock.acquire()
        _, img = cap.read()
        #피부 tone 반환, spring summer fall winter 4종류
        tone = personal_color.analysis(img)

        ref = db.reference('test') #db 위치 지정, 기본 가장 상단을 가르킴
        ref.update({'skin tone' : tone}) #해당 변수가 없다면 생성한다. 혹은 해당 변수를 변경

        self.call_tone_value.emit(tone)
        lock.release()
############################################################################################################
########################################  Air Cleaner Thread  ###################################################
############################################################################################################
class AirCleanerTread(QThread):
    call_AirCleaner_value = pyqtSignal(int, int, int)
    def __init__(self):
        super().__init__()
        self.power = True
    def run(self):
        temp_ref = db.reference('test/temperature')
        humidity_ref = db.reference('test/humidity')
        dust_ref = db.reference('test/dust')
        
        while self.power:
            room_temp = temp_ref.get()              #방 안 온도
            room_humidity = humidity_ref.get()      #방 안 습도
            room_dust = dust_ref.get()              #방 안 먼지
            # 값 반환
            self.call_AirCleaner_value.emit(room_temp, room_humidity, room_dust)
    def stop(self):
        self.power = False
        self.quit()
        self.wait(500)
############################################################################################################
########################################  Set Time  ########################################################
############################################################################################################
class SetTime(QThread):
    call_Time_value = pyqtSignal(int, int, int, str, int, int)  
    def __init__(self):
        super().__init__()
    def run(self):
        EvenOrAfter = "A.M."
        while True:
            now=datetime.datetime.now() #현재 시각을 시스템에서 가져옴
            hour=now.hour

            if(now.hour>=12):
                EvenOrAfter="P.M."
                hour=now.hour%12

                if(now.hour==12):
                    hour=12
            else:
                EvenOrAfter="A.M."

            year = now.year
            month = now.month
            day = now.day
            minute = now.minute

            self.call_Time_value.emit(year, month, day, EvenOrAfter, hour, minute)
            self.wait(1000) #렉 걸려서 어쩔 수 없이 딜레이 줌
############################################################################################################
########################################  Set Weater  ######################################################
############################################################################################################
class SetWeater(QThread):
    call_Weater_value = pyqtSignal(float, QPixmap)
    def __init__(self):
        super().__init__()
        lat, lon = self.get_location()
        self.lat = lat
        self.lon = lon 
        self.power = True   
        
    def run(self):
        API_key = '1296f2181c5e3e22acb4fc333db31dd4'        #Openweathermap API 인증키 [https://openweathermap.org/]
        while self.power == True:
            owm = OWM(API_key)
            mgr = owm.weather_manager()
            #위치
            obs = mgr.weather_at_coords(self.lat, self.lon)      # lat/lon

            #날씨 정보 불러오기
            weather = obs.weather
            temp_dict_kelvin = weather.temperature('celsius')   # 섭씨 온도

            if 'Clear' in weather.status:
                weathericon = QPixmap("data/weather_icon/sun.png")
            elif "Cloudy" in weather.status:
                weathericon = QPixmap("data/weather_icon/clouds.png")
            elif "Rain" in weather.status:
                weathericon = QPixmap("data/weather_icon/drop.png")
            elif "Drizzle" in weather.status:
                weathericon = QPixmap("data/weather_icon/drop.png")
            elif "Snow" in weather.status:
                weathericon = QPixmap("data/weather_icon/snowflake.png")
            elif "Mist" in weather.status:
                weathericon = QPixmap("data/weather_icon/sun.png")
            elif "Mist" in weather.status:
                weathericon = QPixmap("data/weather_icon/mist(2).png")
            elif "Haze" in weather.status:
                weathericon = QPixmap("data/weather_icon/mist(2).png")
            elif "Dust" in weather.status:
                weathericon = QPixmap("data/weather_icon/mist(2).png")
            elif "Fog" in weather.status:
                weathericon = QPixmap("data/weather_icon/mist(2).png")
            elif "Thunderstorm" in weather.status:
                weathericon = QPixmap("data/weather_icon/bolt.png")
            else:
                weathericon = QPixmap("data/weather_icon/sun.png") 

            temp = round(temp_dict_kelvin['temp'], 1)

            self.call_Weater_value.emit(temp, weathericon)
            self.wait(1801000)  #렉 걸려서 어쩔 수 없이 딜레이 줌
    #IP 기반 위도와 경도 찾기
    def get_location(self):
        GOOGLE_API_KEY = 'AIzaSyD_qckRmOOWxcrSop9uWlKaqkec4ccwQOs'
        url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}'
        data = {'considerIp': True, }   # 현 IP로 데이터 추출

        result = requests.post(url, data) # 해당 API에 요청을 보내며 데이터를 추출한다.

        result2 = json.loads(result.text)

        lat = result2["location"]["lat"] # 현재 위치의 위도 추출
        lng = result2["location"]["lng"] # 현재 위치의 경도 추출
        
        return lat,lng 
    def stop(self):
        self.power = False
        self.quit()
        self.wait(500)
############################################################################################################
########################################  Health Care  #####################################################
############################################################################################################
class DetectMelanomaThread(QThread):
    call_Melanoma_value = pyqtSignal(float)
    def __init__(self):
        super().__init__()
    def run(self):
        lock.acquire()
        _, img = cap.read()
        comp = health.melanoma(img)
        self.call_Melanoma_value.emit(comp)
        lock.release()

class DetectStressThread(QThread):
    call_Stress_value = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.power = True
    def run(self):
        # cap = cv2.VideoCapture(1)  
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        while self.power:
            lock.acquire()
            _, frame_stress = cap.read()
            gray_img = cv2.cvtColor(frame_stress, cv2.COLOR_BGR2GRAY)
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                with mp_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
                    # 추출한 얼굴 영역에 FaceDetection 모듈을 적용합니다. / FaceDetection은 results 사용
                    results = face_detection.process(cv2.cvtColor(frame_stress, cv2.COLOR_BGR2RGB))
                    # 추출한 얼굴 영역에 FaceMesh 모듈을 적용합니다. / 23.5.7 이전 faceMash 사용시 if results.detections: 의 results를 face_results로 변환
                    try:
                        bad_count = health.emotion_song(results, frame_stress, gray_img)
                        self.call_Stress_value.emit(bad_count)
                    except:
                        pass
            lock.release()
    def stop(self):
        self.power = False
        self.quit()
        self.wait(500)
############################################################################################################
########################################  MAIN WINDOW   ####################################################
############################################################################################################
class Ui_MainWindow(QWidget, object):
    def __init__(self, MainWindow):
        super().__init__()
        self.MainWindow = MainWindow
        MainWindow.setObjectName("MainWindow")
        
        palette = QPalette()

        brush = QBrush(QColor(255, 255, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)

        brush = QBrush(QColor(0, 0, 0))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Window, brush)

        MainWindow.setPalette(palette)
        MainWindow.showFullScreen()

        
        self.detect_face_value = 0
        self.count = 0
        self.setupUiOff(self.MainWindow)
    def setupUiOff(self, MainWindow):
        #메인 윈도우에 들어갈 self.centralwidget 위젯 정의
        self.centralwidget = QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #메인 윈도우에 하나의 위젯만 넣도록 제한하기
        self.MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(self.MainWindow)
        #시간  ==================================================================================
        #시간 label [(오전/오후)시/분]
        self.time = QLabel(self.centralwidget)
        self.time.setGeometry(QRect(1050,770,500,60))    #(x,y, 가로, 높이)
        self.time.setObjectName("time")
        #setFont(QtGui.QFont("Font_name",Font_size))
        self.time.setFont(QFont("맑은 고딕",50)) 

        #날짜 label [년/월/일]
        self.date = QLabel(self.centralwidget)
        self.date.setGeometry(QRect(1180, 660, 300, 50))
        self.date.setObjectName("date")
        self.date.setFont(QFont("맑은 고딕",20))

        #시간 쓰래드
        self.set_time = SetTime()
        self.set_time.start()
        self.set_time.call_Time_value.connect(self.SetTimeFunc)

        displayon = QIcon("data/button_icon/turnoff2.png")
        self.DisplayOn = QPushButton(self.centralwidget)
        self.DisplayOn.setGeometry(QRect(2480,740,60,60))
        self.DisplayOn.setIcon(displayon)
        self.DisplayOn.setIconSize(QSize(45,45))
        self.DisplayOn.setStyleSheet("QPushButton { background-color: black;}")
        self.DisplayOn.clicked.connect(self.diplayon_button_func)
        self.DisplayOn.setShortcut("Esc")

        #시작 쓰래드
        self.DisplayOn =Display()
        self.DisplayOn.start()
        # self.DisplayOn.wait(1)
        self.DisplayOn.change_start_signal.connect(self.DisplayOnFunc)
        
    def setupUiOn(self, MainWindow):
        #self.centralwidget 위젯 재정의 하기
        self.centralwidget = QWidget(self.MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        #재정의 한 self.centralwidget 위젯을 메인 윈도우에 다시 넣기
        self.MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(self.MainWindow)


        self.count = 0
        #시간  ==================================================================================
        # 시간 label [(오전/오후)시/분]
        self.time = QLabel(self.centralwidget)
        self.time.setGeometry(QRect(30,1150,800,60))    #(x,y, 가로, 높이)
        self.time.setObjectName("time")
        #setFont(QtGui.QFont("Font_name",Font_size))
        self.time.setFont(QFont("맑은 고딕",50)) 

        #날짜 label [년/월/일]
        self.date = QLabel(self.centralwidget)
        self.date.setGeometry(QRect(250, 1100, 300, 50))
        self.date.setObjectName("date")
        self.date.setFont(QFont("맑은 고딕",20))

        #시간 쓰래드
        self.set_time = SetTime()
        self.set_time.start()
        self.set_time.call_Time_value.connect(self.SetTimeFunc)

        #날씨 이모티콘 ====================================================================
        self.weather = QLabel(self.centralwidget)
        self.weather.setGeometry(QRect(120, 50, 150,150))
        self.weather.setObjectName("weather")

        #온도 label [온도 출력]
        self.temperature = QLabel(self.centralwidget)
        self.temperature.setGeometry(QRect(80, 160, 300,130))
        self.temperature.setObjectName("temperature")
        self.temperature.setFont(QFont("맑은 고딕",20))

        #날씨 쓰래드
        self.set_weather = SetWeater()
        self.set_weather.start()
        self.set_weather.call_Weater_value.connect(self.SetWeatherFunc)
        #버튼 아이콘 ===========================================================================
        left_arrow = QIcon("data/button_icon/left_arrow.png")
        right_arrow = QIcon("data/button_icon/right_arrow.png")
        cheek1 = QIcon("data/button_icon/cheek1.png")
        cheek2 = QIcon("data/button_icon/cheek2.png")
        cheek3 = QIcon("data/button_icon/cheek3.png")
        cheek4 = QIcon("data/button_icon/cheek4.png")
        eyebrow1 = QIcon("data/button_icon/eyebrow1.png")
        eyebrow2 = QIcon("data/button_icon/eyebrow2(right).png")
        eyebrow3 = QIcon("data/button_icon/eyebrow3.png")
        eyebrow4 = QIcon("data/button_icon/eyebrow4.png")
        eyebrow5 = QIcon("data/button_icon/eyebrow5.png")
        makeup = QIcon("data/button_icon/makeup.png")
        healthcare = QIcon("data/button_icon/healthcare.png")
        aircleaner = QIcon("data/button_icon/aircleaner.png")
        displayoff = QIcon("data/button_icon/turnoff2.png")

        
        #=====================================================================================
        #화장 버튼 만들기======================================================================
        self.btnMakeUp = QPushButton()
        # self.btnMakeUp.setGeometry(QRect(2450,1000,50,50))
        self.btnMakeUp.clicked.connect(self.makeupshow)
        self.btnMakeUp.setCheckable(True)

        self.toggWidget = QWidget(self.centralwidget)
        self.toggleLayout = QFormLayout()
        self.toggWidget.setGeometry(QRect(2430,1100,100,80))
        self.showhideToggle = AnimatedToggle()
        self.showhideToggle.setFixedSize(self.showhideToggle.sizeHint())
        self.toggleLayout.addWidget(QLabel("Hide / Show"))
        self.toggleLayout.addWidget(self.showhideToggle)
        self.toggWidget.setLayout(self.toggleLayout)
        self.showhideToggle.stateChanged.connect(self.facehide)
        
        #얼굴 인식 칸 ====================================================================
        self.face = QLabel(self.centralwidget)
        self.face.setGeometry(QRect(800, 200, 800,1000))
        self.face.setStyleSheet('border-style: solid;''border-width: 3px;''border-color: #00FF00')
        self.face.setObjectName("face")
        
        self.face_make = QLabel(self.centralwidget)
        self.face_make.setGeometry(QRect(803, 203, 794,994))
        self.face_make.setObjectName("face_make")

        # 메이크업 탭 위젯 생성
        self.face_makeTab = QTabWidget(self.centralwidget) # = QWidget(self.addMouthWidjet)
        self.face_maketabbaseWidth = 0
        self.face_maketabextendedWidth = 735
        self.face_makeTab.setGeometry(QRect(1650, 300, self.face_maketabbaseWidth, 800))
        self.face_makeTab.setStyleSheet("QTabWidget::pane { background-color: #FF6666; border: 3px; solid orange; }"
                                "QTabBar::tab { background-color: #FF9999; border: 3px; solid orange; color: #FFFFFF; width: 100px; height: 30px;}"
                                "QTabBar::tab:selected { background-color: #FF6666; color: #FFFFFF; width: 100px; height: 30px;}")

        self.addSkinWidjet = QWidget()
        self.addMouthWidjet = QWidget()
        self.addCeekWidjet = QWidget()
        self.addEyeBrowWidjet = QWidget()

        self.face_makeTab.addTab(self.addSkinWidjet, 'Skin')
        self.face_makeTab.addTab(self.addMouthWidjet, 'Mouth')
        self.face_makeTab.addTab(self.addCeekWidjet, 'Cheek')
        self.face_makeTab.addTab(self.addEyeBrowWidjet, 'EyeBrow')
        self.face_maketabAni = QPropertyAnimation(self.face_makeTab, b'geometry')
        
        # 피부 =================================================================================
        self.SkinLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.addSkinWidjet.setLayout(self.SkinLayout)

        self.skin_gb_1 = QGroupBox()
        self.skin_gb_1.setTitle("Color")
        self.skin_gb_1.setStyleSheet('color: #FFFFFF;')
        self.skin_gb_1.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.SkinLayout.addWidget(self.skin_gb_1)
        self.skin_color_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.skin_gb_1.setLayout(self.skin_color_box)
        self.skin_gb_1.setFixedHeight(500)

        self.SkinColor1 = QPushButton()
        self.SkinColor1.setMaximumHeight(300)
        self.SkinColor1.setStyleSheet("QPushButton { background-color: rgb(220,175,145);}")
        self.SkinColor1.clicked.connect(self.s_color1_start)
        self.SkinColor1.setCheckable(True)

        self.SkinColor2 = QPushButton()
        self.SkinColor2.setMaximumHeight(300)
        self.SkinColor2.setStyleSheet("QPushButton { background-color: rgb(247,211,177);}")
        self.SkinColor2.clicked.connect(self.s_color2_start)
        self.SkinColor2.setCheckable(True)

        self.SkinColor3 = QPushButton()
        self.SkinColor3.setMaximumHeight(300)
        self.SkinColor3.setStyleSheet("QPushButton { background-color: rgb(194,145,90);}")
        self.SkinColor3.clicked.connect(self.s_color3_start)
        self.SkinColor3.setCheckable(True)

        self.SkinColor4 = QPushButton()
        self.SkinColor4.setMaximumHeight(300)
        self.SkinColor4.setStyleSheet("QPushButton { background-color: rgb(224,166,105);}")
        self.SkinColor4.clicked.connect(self.s_color4_start)
        self.SkinColor4.setCheckable(True)
        
        self.SkinColor5 = QPushButton()
        self.SkinColor5.setMaximumHeight(300)
        self.SkinColor5.setStyleSheet("QPushButton { background-color: rgb(207,155,102);}")
        self.SkinColor5.clicked.connect(self.s_color5_start)
        self.SkinColor5.setCheckable(True)
            #블렌딩
        self.skin_gb_2 = QGroupBox()
        self.skin_gb_2.setTitle("Blending")
        self.skin_gb_2.setStyleSheet('color: #FFFFFF;')
        self.skin_gb_2.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.SkinLayout.addWidget(self.skin_gb_2)
        self.skin_blending_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.skin_gb_2.setLayout(self.skin_blending_box)

        self.s_DOWN = QPushButton()
        self.s_DOWN.clicked.connect(self.s_DownVal)
        self.s_DOWN.setMaximumHeight(50)
        self.s_DOWN.setMaximumWidth(50)
        
        self.s_DOWN.setIcon(left_arrow)
        self.s_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.s_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px; }")

        self.s_Blending = QSlider(Qt.Horizontal)
        self.s_Blending.setMaximumHeight(50)
        self.s_Blending.setMaximumWidth(350)
        self.s_Blending.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")
        self.s_Blending.valueChanged.connect(self.s_Blendingval)

        self.s_UP = QPushButton()
        self.s_UP.setMaximumHeight(50)
        self.s_UP.setMaximumWidth(50)
        self.s_UP.clicked.connect(self.s_UpVal)
        self.s_UP.setIcon(right_arrow)
        self.s_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.s_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.s_Blending.setValue(0)
        self.s_mix = 0

        # self.skin_result = None
        # self.skin_value = False
        self.skin_color_box.addWidget(self.SkinColor1)
        self.skin_color_box.addWidget(self.SkinColor2)
        self.skin_color_box.addWidget(self.SkinColor3)
        self.skin_color_box.addWidget(self.SkinColor4)
        self.skin_color_box.addWidget(self.SkinColor5)
        self.skin_blending_box.addWidget(self.s_DOWN)
        self.skin_blending_box.addWidget(self.s_Blending)
        self.skin_blending_box.addWidget(self.s_UP)
        
        #입술 ==============================================================================================
        self.MouthLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.addMouthWidjet.setLayout(self.MouthLayout)

        self.mouth_gb_1 = QGroupBox()
        self.mouth_gb_1.setTitle("Color")
        self.mouth_gb_1.setStyleSheet('color: #FFFFFF;')
        self.mouth_gb_1.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.MouthLayout.addWidget(self.mouth_gb_1)
        self.mouth_color_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.mouth_gb_1.setLayout(self.mouth_color_box)
        self.mouth_gb_1.setFixedHeight(500)

        self.MouthColor1 = QPushButton()
        self.MouthColor1.setMaximumHeight(300)
        self.MouthColor1.setStyleSheet("QPushButton { background-color: rgb(229,106,75);}")
        self.MouthColor1.clicked.connect(self.m_color1_start)
        self.MouthColor1.setCheckable(True)

        self.MouthColor2 = QPushButton()
        self.MouthColor2.setMaximumHeight(300)
        self.MouthColor2.setStyleSheet("QPushButton { background-color: rgb(236,64,126);}")
        self.MouthColor2.clicked.connect(self.m_color2_start)
        self.MouthColor2.setCheckable(True)

        self.MouthColor3 = QPushButton()
        self.MouthColor3.setMaximumHeight(300)
        self.MouthColor3.setStyleSheet("QPushButton { background-color: rgb(156,59,43);}")
        self.MouthColor3.clicked.connect(self.m_color3_start)
        self.MouthColor3.setCheckable(True)
    
        self.MouthColor4 = QPushButton()
        self.MouthColor4.setMaximumHeight(300)
        self.MouthColor4.setStyleSheet("QPushButton { background-color: rgb(167,61,87);}")
        self.MouthColor4.clicked.connect(self.m_color4_start)
        self.MouthColor4.setCheckable(True)

        self.MouthColor5 = QPushButton()
        self.MouthColor5.setMaximumHeight(300)
        self.MouthColor5.setStyleSheet("QPushButton { background-color: rgb(218,128,117);}")
        self.MouthColor5.clicked.connect(self.m_color5_start)
        self.MouthColor5.setCheckable(True)

            #블렌딩
        self.mouth_gb_2 = QGroupBox()
        self.mouth_gb_2.setTitle("Blending")
        self.mouth_gb_2.setStyleSheet('color: #FFFFFF;')
        self.mouth_gb_2.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.MouthLayout.addWidget(self.mouth_gb_2)
        self.mouth_blending_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.mouth_gb_2.setLayout(self.mouth_blending_box)

        self.m_DOWN = QPushButton()
        self.m_DOWN.clicked.connect(self.m_DownVal)
        self.m_DOWN.setMaximumHeight(50)
        self.m_DOWN.setMaximumWidth(50)
        # self.m_DOWN.setText("DOWN")
        self.m_DOWN.setIcon(left_arrow)
        self.m_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.m_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")
        

        self.m_Blending = QSlider(Qt.Horizontal,self.addMouthWidjet)
        self.m_Blending.setMaximumHeight(50)
        self.m_Blending.setMaximumWidth(350)
        self.m_Blending.valueChanged.connect(self.m_Blendingval)
        self.m_Blending.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.m_UP = QPushButton(self.addMouthWidjet)
        self.m_UP.setMaximumHeight(50)
        self.m_UP.setMaximumWidth(50)
        self.m_UP.clicked.connect(self.m_UpVal)
        # self.m_UP.setText("UP")
        self.m_UP.setIcon(right_arrow)
        self.m_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.m_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.m_Blending.setValue(0)
        self.m_mix = 0

        self.mouth_color_box.addWidget(self.MouthColor1)
        self.mouth_color_box.addWidget(self.MouthColor2)
        self.mouth_color_box.addWidget(self.MouthColor3)
        self.mouth_color_box.addWidget(self.MouthColor4)
        self.mouth_color_box.addWidget(self.MouthColor5)

        self.mouth_blending_box.addWidget(self.m_DOWN)
        self.mouth_blending_box.addWidget(self.m_Blending)
        self.mouth_blending_box.addWidget(self.m_UP)
        
        # self.mouth_value = False
        # self.mouth_result = None
        #눈썹 ==============================================================================================
        self.EyeBrowLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.addEyeBrowWidjet.setLayout(self.EyeBrowLayout)

        self.eye_gb_1 = QGroupBox()
        self.eye_gb_1.setTitle("Shape")
        self.eye_gb_1.setStyleSheet('color: #FFFFFF;')
        self.eye_gb_1.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.EyeBrowLayout.addWidget(self.eye_gb_1)
        self.eye_blow_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.eye_gb_1.setLayout(self.eye_blow_box)
        self.eye_gb_1.setFixedHeight(250)
        
        self.EyeColor1 = QPushButton()
        self.EyeColor1.clicked.connect(self.e_color1_start)
        self.EyeColor1.setCheckable(True)
        self.EyeColor1.setMaximumHeight(200)
        self.EyeColor1.setIcon(eyebrow1)
        self.EyeColor1.setIconSize(QSize(100,100))

        self.EyeColor2 = QPushButton()
        self.EyeColor2.clicked.connect(self.e_color2_start)
        self.EyeColor2.setCheckable(True)
        self.EyeColor2.setMaximumHeight(200)
        self.EyeColor2.setIcon(eyebrow2)
        self.EyeColor2.setIconSize(QSize(100,100))

        self.EyeColor3 = QPushButton()
        self.EyeColor3.clicked.connect(self.e_color3_start)
        self.EyeColor3.setCheckable(True)
        self.EyeColor3.setMaximumHeight(200)
        self.EyeColor3.setIcon(eyebrow3)
        self.EyeColor3.setIconSize(QSize(100,100))

        self.EyeColor4 = QPushButton()
        self.EyeColor4.clicked.connect(self.e_color4_start)
        self.EyeColor4.setCheckable(True)
        self.EyeColor4.setMaximumHeight(200)
        self.EyeColor4.setIcon(eyebrow4)
        self.EyeColor4.setIconSize(QSize(100,100))

        self.EyeColor5 = QPushButton()
        self.EyeColor5.clicked.connect(self.e_color5_start)
        self.EyeColor5.setCheckable(True)
        self.EyeColor5.setMaximumHeight(200)
        self.EyeColor5.setIcon(eyebrow5)
        self.EyeColor5.setIconSize(QSize(100,100))

            #블렌딩
        self.eye_gb_2 = QGroupBox()
        self.eye_gb_2.setTitle("Blending")
        self.eye_gb_2.setStyleSheet('color: #FFFFFF;')
        self.eye_gb_2.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.EyeBrowLayout.addWidget(self.eye_gb_2)
        self.eye_blow_blending_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.eye_gb_2.setLayout(self.eye_blow_blending_box)

        self.e_DOWN = QPushButton()
        self.e_DOWN.clicked.connect(self.e_DownVal)
        self.e_DOWN.setMaximumHeight(50)
        self.e_DOWN.setMaximumWidth(50)
        self.e_DOWN.setIcon(left_arrow)
        self.e_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.e_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.e_Blending = QSlider(Qt.Horizontal)
        self.e_Blending.valueChanged.connect(self.e_Blendingval)
        self.e_Blending.setMaximumHeight(50)
        self.e_Blending.setMaximumWidth(350)
        self.e_Blending.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.e_UP = QPushButton()
        self.e_UP.clicked.connect(self.e_UpVal)
        self.e_UP.setMaximumHeight(50)
        self.e_UP.setMaximumWidth(50)
        self.e_UP.setIcon(right_arrow)
        self.e_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.e_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.eb_mix = 0

            #눈썹 위치==============================================================================================
        self.eye_gb_3 = QGroupBox()
        self.eye_gb_3.setTitle("Pos x")
        self.eye_gb_3.setStyleSheet('color: #FFFFFF;')
        self.eye_gb_3.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.EyeBrowLayout.addWidget(self.eye_gb_3)
        self.eye_blow_posx_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.eye_gb_3.setLayout(self.eye_blow_posx_box)

        self.xpos_DOWN = QPushButton()
        self.xpos_DOWN.clicked.connect(self.xpos_DOWNVal)
        self.xpos_DOWN.setMaximumHeight(50)
        self.xpos_DOWN.setMaximumWidth(50)
        self.posx = 0
        self.xpos_DOWN.setIcon(left_arrow)
        self.xpos_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.xpos_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.dyxSlider = QSlider(Qt.Horizontal)
        self.dyxSlider.setValue(50)
        self.dyxSlider.valueChanged.connect(self.dyxVal)
        self.dyxSlider.setMaximumHeight(50)
        self.dyxSlider.setMaximumWidth(350)
        self.dyxSlider.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.xpos_UP = QPushButton()
        self.xpos_UP.clicked.connect(self.xpos_UPVal)
        self.xpos_UP.setMaximumHeight(50)
        self.xpos_UP.setMaximumWidth(50)
        self.xpos_UP.setIcon(right_arrow)
        self.xpos_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.xpos_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.eye_gb_4 = QGroupBox()
        self.eye_gb_4.setTitle("Pos y")
        self.eye_gb_4.setStyleSheet('color: #FFFFFF;')
        self.eye_gb_4.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.EyeBrowLayout.addWidget(self.eye_gb_4)
        self.eye_blow_posy_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.eye_gb_4.setLayout(self.eye_blow_posy_box)

        self.ypos_DOWN = QPushButton()
        self.ypos_DOWN.clicked.connect(self.ypos_DOWNVal)
        self.ypos_DOWN.setMaximumHeight(50)
        self.ypos_DOWN.setMaximumWidth(50)
        self.ypos_DOWN.setIcon(left_arrow)
        self.ypos_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.ypos_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.dyySlider = QSlider(Qt.Horizontal)
        self.dyySlider.setValue(50)
        self.dyySlider.valueChanged.connect(self.dyyVal)
        self.dyySlider.setMaximumHeight(50)
        self.dyySlider.setMaximumWidth(350)
        self.dyySlider.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.ypos_UP = QPushButton()
        self.ypos_UP.clicked.connect(self.ypos_UPVal)
        self.ypos_UP.setMaximumHeight(50)
        self.ypos_UP.setMaximumWidth(50)
        self.ypos_UP.setIcon(right_arrow)
        self.ypos_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.ypos_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.posy = 0

            #눈썹 길이
        self.eye_gb_5 = QGroupBox()
        self.eye_gb_5.setTitle("Length")
        self.eye_gb_5.setStyleSheet('color: #FFFFFF;')
        self.eye_gb_5.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.EyeBrowLayout.addWidget(self.eye_gb_5)
        self.eye_blow_length_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.eye_gb_5.setLayout(self.eye_blow_length_box)

        self.length_DOWN = QPushButton()
        self.length_DOWN.clicked.connect(self.length_DOWNVal)
        self.length_DOWN.setMaximumHeight(50)
        self.length_DOWN.setMaximumWidth(50)
        self.length_DOWN.setIcon(left_arrow)
        self.length_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.length_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.lengthSlider = QSlider(Qt.Horizontal)
        self.lengthSlider.valueChanged.connect(self.lengthVal)
        self.lengthSlider.setMaximumHeight(50)
        self.lengthSlider.setMaximumWidth(350)
        self.lengthSlider.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.length_UP = QPushButton()
        self.length_UP.clicked.connect(self.length_UPVal)
        self.length_UP.setMaximumHeight(50)
        self.length_UP.setMaximumWidth(50)
        self.length_UP.setIcon(right_arrow)
        self.length_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.length_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.length = 0

        self.eye_blow_box.addWidget(self.EyeColor1)
        self.eye_blow_box.addWidget(self.EyeColor2)
        self.eye_blow_box.addWidget(self.EyeColor3)
        self.eye_blow_box.addWidget(self.EyeColor4)
        self.eye_blow_box.addWidget(self.EyeColor5)
        self.eye_blow_blending_box.addWidget(self.e_DOWN)
        self.eye_blow_blending_box.addWidget(self.e_Blending)
        self.eye_blow_blending_box.addWidget(self.e_UP)
        self.eye_blow_posx_box.addWidget(self.xpos_DOWN)
        self.eye_blow_posx_box.addWidget(self.dyxSlider)
        self.eye_blow_posx_box.addWidget(self.xpos_UP)
        self.eye_blow_posy_box.addWidget(self.ypos_DOWN)
        self.eye_blow_posy_box.addWidget(self.dyySlider)
        self.eye_blow_posy_box.addWidget(self.ypos_UP)
        self.eye_blow_length_box.addWidget(self.length_DOWN)
        self.eye_blow_length_box.addWidget(self.lengthSlider)
        self.eye_blow_length_box.addWidget(self.length_UP)

        # self.eyebrow_value = False
        # self.eyebrow_result = None
        #볼 ==============================================================================================
        self.CheekLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.addCeekWidjet.setLayout(self.CheekLayout)

        self.cheek_gb_1 = QGroupBox()
        self.cheek_gb_1.setTitle("Shape")
        self.cheek_gb_1.setStyleSheet('color: #FFFFFF;')
        self.cheek_gb_1.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.CheekLayout.addWidget(self.cheek_gb_1)
        self.cheek_shape_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.cheek_gb_1.setLayout(self.cheek_shape_box)
        self.cheek_gb_1.setFixedHeight(230)

        self.CheekColor1 = QPushButton()
        self.CheekColor1.clicked.connect(self.c_color1_start)
        self.CheekColor1.setCheckable(True)
        self.CheekColor1.setMaximumHeight(200)
        self.CheekColor1.setMaximumWidth(133)
        self.CheekColor1.setIcon(cheek1)
        self.CheekColor1.setIconSize(QSize(100,100))

        self.CheekColor2 = QPushButton()
        self.CheekColor2.clicked.connect(self.c_color2_start)
        self.CheekColor2.setCheckable(True)
        self.CheekColor2.setMaximumHeight(200)
        self.CheekColor2.setMaximumWidth(133)
        self.CheekColor2.setIcon(cheek2)
        self.CheekColor2.setIconSize(QSize(100,100))

        self.CheekColor3 = QPushButton()
        self.CheekColor3.clicked.connect(self.c_color3_start)
        self.CheekColor3.setCheckable(True)
        self.CheekColor3.setMaximumHeight(200)
        self.CheekColor3.setMaximumWidth(133)
        self.CheekColor3.setIcon(cheek3)
        self.CheekColor3.setIconSize(QSize(100,100))

        self.CheekColor4 = QPushButton()
        self.CheekColor4.clicked.connect(self.c_color4_start)
        self.CheekColor4.setCheckable(True)
        self.CheekColor4.setMaximumHeight(200)
        self.CheekColor4.setMaximumWidth(133)
        self.CheekColor4.setIcon(cheek4)
        self.CheekColor4.setIconSize(QSize(100,100))

        # self.cheek_value = False
        # self.cheek_result = None
            #블렌딩
        self.cheek_gb_2 = QGroupBox()
        self.cheek_gb_2.setTitle("Blending")
        self.cheek_gb_2.setStyleSheet('color: #FFFFFF;')
        self.cheek_gb_2.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.CheekLayout.addWidget(self.cheek_gb_2)
        self.cheek_blending_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.cheek_gb_2.setLayout(self.cheek_blending_box)

        self.c_DOWN = QPushButton()
        self.c_DOWN.clicked.connect(self.c_DownVal)
        self.c_DOWN.setMaximumHeight(50)
        self.c_DOWN.setMaximumWidth(50)
        self.c_DOWN.setIcon(left_arrow)
        self.c_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_Blending = QSlider(Qt.Horizontal)
        self.c_Blending.valueChanged.connect(self.c_Blendingval)
        self.c_Blending.setMaximumHeight(50)
        self.c_Blending.setMaximumWidth(350)
        self.c_Blending.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.c_UP = QPushButton()
        self.c_UP.clicked.connect(self.c_UpVal)
        self.c_UP.setMaximumHeight(50)
        self.c_UP.setMaximumWidth(50)
        self.c_UP.setIcon(right_arrow)
        self.c_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")
        
        self.c_mix = 0
    
            #볼 위치==============================================================================================
        self.cheek_gb_3 = QGroupBox()
        self.cheek_gb_3.setTitle("Pos x")
        self.cheek_gb_3.setStyleSheet('color: #FFFFFF;')
        self.cheek_gb_3.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.CheekLayout.addWidget(self.cheek_gb_3)
        self.cheek_posx_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.cheek_gb_3.setLayout(self.cheek_posx_box)

        self.c_xpos_DOWN = QPushButton()
        self.c_xpos_DOWN.clicked.connect(self.c_xpos_DOWNVal)
        self.c_xpos_DOWN.setMaximumHeight(50)
        self.c_xpos_DOWN.setMaximumWidth(50)
        self.c_xpos_DOWN.setIcon(left_arrow)
        self.c_xpos_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_xpos_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_dyxSlider = QSlider(Qt.Horizontal)
        self.c_dyxSlider.setValue(50)
        self.c_dyxSlider.valueChanged.connect(self.c_dyxVal)
        self.c_dyxSlider.setMaximumHeight(50)
        self.c_dyxSlider.setMaximumWidth(350)
        self.c_dyxSlider.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.c_xpos_UP = QPushButton()
        self.c_xpos_UP.clicked.connect(self.c_xpos_UPVal)
        self.c_xpos_UP.setMaximumHeight(50)
        self.c_xpos_UP.setMaximumWidth(50)
        self.c_xpos_UP.setIcon(right_arrow)
        self.c_xpos_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_xpos_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_posx = 0
        
        self.cheek_gb_4 = QGroupBox()
        self.cheek_gb_4.setTitle("Pos y")
        self.cheek_gb_4.setStyleSheet('color: #FFFFFF;')
        self.cheek_gb_4.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.CheekLayout.addWidget(self.cheek_gb_4)
        self.cheek_posy_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.cheek_gb_4.setLayout(self.cheek_posy_box)

        self.c_ypos_DOWN = QPushButton()
        self.c_ypos_DOWN.clicked.connect(self.c_ypos_DOWNVal)
        self.c_ypos_DOWN.setMaximumHeight(50)
        self.c_ypos_DOWN.setMaximumWidth(50)
        self.c_ypos_DOWN.setIcon(left_arrow)
        self.c_ypos_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_ypos_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_dyySlider = QSlider(Qt.Horizontal)
        self.c_dyySlider.setValue(50)
        self.c_dyySlider.valueChanged.connect(self.c_dyyVal)
        self.c_dyySlider.setMaximumHeight(50)
        self.c_dyySlider.setMaximumWidth(350)
        self.c_dyySlider.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.c_ypos_UP = QPushButton()
        self.c_ypos_UP.clicked.connect(self.c_ypos_UPVal)
        self.c_ypos_UP.setMaximumHeight(50)
        self.c_ypos_UP.setMaximumWidth(50)
        self.c_ypos_UP.setIcon(right_arrow)
        self.c_ypos_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_ypos_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_posy = 0

        self.cheek_gb_5 = QGroupBox()
        self.cheek_gb_5.setTitle("Width")
        self.cheek_gb_5.setStyleSheet('color: #FFFFFF;')
        self.cheek_gb_5.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.CheekLayout.addWidget(self.cheek_gb_5)
        self.cheek_width_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.cheek_gb_5.setLayout(self.cheek_width_box)

        self.c_width_DOWN = QPushButton()
        self.c_width_DOWN.clicked.connect(self.c_width_DOWNVal)
        self.c_width_DOWN.setMaximumHeight(50)
        self.c_width_DOWN.setMaximumWidth(50)
        self.c_width_DOWN.setIcon(left_arrow)
        self.c_width_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_width_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_widthSlider = QSlider(Qt.Horizontal)
        self.c_widthSlider.valueChanged.connect(self.c_widthVal)
        self.c_widthSlider.setMaximumHeight(50)
        self.c_widthSlider.setMaximumWidth(350)
        self.c_widthSlider.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.c_width_UP = QPushButton()
        self.c_width_UP.clicked.connect(self.c_width_UPVal)
        self.c_width_UP.setMaximumHeight(50)
        self.c_width_UP.setMaximumWidth(50)
        self.c_width_UP.setIcon(right_arrow)
        self.c_width_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_width_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_width = 0

        self.cheek_gb_6 = QGroupBox()
        self.cheek_gb_6.setTitle("Height")
        self.cheek_gb_6.setStyleSheet('color: #FFFFFF;')
        self.cheek_gb_6.setFont(QFont("맑은 고딕",15,QFont.Bold))
        self.CheekLayout.addWidget(self.cheek_gb_6)
        self.cheek_height_box = QBoxLayout(QBoxLayout.LeftToRight)
        self.cheek_gb_6.setLayout(self.cheek_height_box)

        self.c_height_DOWN = QPushButton()
        self.c_height_DOWN.clicked.connect(self.c_height_DOWNVal)
        self.c_height_DOWN.setMaximumHeight(50)
        self.c_height_DOWN.setMaximumWidth(50)
        self.c_height_DOWN.setIcon(left_arrow)
        self.c_height_DOWN.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_height_DOWN.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")

        self.c_heightSlider = QSlider(Qt.Horizontal)
        self.c_heightSlider.valueChanged.connect(self.c_heightVal)
        self.c_heightSlider.setMaximumHeight(50)
        self.c_heightSlider.setMaximumWidth(350)
        self.c_heightSlider.setStyleSheet("QSlider::groove:horizontal { border: 2px solid grey; background: #FF9999; height: 30px; margin: 2px 0;}"
                                    "QSlider::handle:horizontal { background: white; width: 20px; margin: -2px 0; border-radius: 3px;}")

        self.c_height_UP = QPushButton()
        self.c_height_UP.clicked.connect(self.c_height_UPVal)
        self.c_height_UP.setMaximumHeight(50)
        self.c_height_UP.setMaximumWidth(50)
        self.c_height_UP.setIcon(right_arrow)
        self.c_height_UP.setIconSize(QSize(30,30)) # 아이콘 크기 설정
        self.c_height_UP.setStyleSheet("QPushButton { background-color: #FFFFFF; border-radius: 25px;}")
        
        self.c_height = 0

        self.cheek_shape_box.addWidget(self.CheekColor1)
        self.cheek_shape_box.addWidget(self.CheekColor2)
        self.cheek_shape_box.addWidget(self.CheekColor3)
        self.cheek_shape_box.addWidget(self.CheekColor4)
        self.cheek_blending_box.addWidget(self.c_DOWN)
        self.cheek_blending_box.addWidget(self.c_Blending)
        self.cheek_blending_box.addWidget(self.c_UP)
        self.cheek_posx_box.addWidget(self.c_xpos_DOWN)
        self.cheek_posx_box.addWidget(self.c_dyxSlider)
        self.cheek_posx_box.addWidget(self.c_xpos_UP)
        self.cheek_posy_box.addWidget(self.c_ypos_DOWN)
        self.cheek_posy_box.addWidget(self.c_dyySlider)
        self.cheek_posy_box.addWidget(self.c_ypos_UP)
        self.cheek_width_box.addWidget(self.c_width_DOWN)
        self.cheek_width_box.addWidget(self.c_widthSlider)
        self.cheek_width_box.addWidget(self.c_width_UP)
        self.cheek_height_box.addWidget(self.c_height_DOWN)
        self.cheek_height_box.addWidget(self.c_heightSlider)
        self.cheek_height_box.addWidget(self.c_height_UP)
        # ==============================================================================================================
        # 세이브 로드====================================================================================================
        self.savebtn = QPushButton(self.centralwidget)
        self.savebtn.setGeometry(QRect(1650,1110,100,50))
        self.savebtn.setText("SAVE")
        self.savebtn.clicked.connect(self.openSaveWindow)
        self.savebtn.setCheckable(True)
        self.savebtn.hide()

        self.saveloadbaseWidth = 0
        self.saveloadextendedWidth = 300

        self.saveWidget = QWidget(self.centralwidget)
        self.saveWidget.setGeometry(QRect(1750, 1100, self.saveloadbaseWidth, 180))
        
        self.saveForm = QBoxLayout(QBoxLayout.TopToBottom)
        self.saveWidget.setLayout(self.saveForm)
        self.save_gb = QGroupBox()
        self.save_gb.setStyleSheet('color: #FFFFFF')
        self.save_gb.setTitle("SAVE")
        self.saveForm.addWidget(self.save_gb)
        self.saveLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.save_gb.setLayout(self.saveLayout)
        
        self.savelabel = QLabel("데이터를 저장할 위치를 선택하세요.")
        self.saveLayout.addWidget(self.savelabel)
        self.saverbtn1 = QRadioButton("save data1")
        self.saverbtn1.clicked.connect(self.saverbtn1_func)
        self.saveLayout.addWidget(self.saverbtn1)
        self.saverbtn2 = QRadioButton("save data2")
        self.saverbtn2.clicked.connect(self.saverbtn2_func)
        self.saveLayout.addWidget(self.saverbtn2)
        self.saverbtn3 = QRadioButton("save data3")
        self.saverbtn3.clicked.connect(self.saverbtn3_func)
        self.saveLayout.addWidget(self.saverbtn3)
        self.saveButton = QPushButton("SAVE")
        self.saveButton.clicked.connect(self.save_func)
        self.saveLayout.addWidget(self.saveButton)        

        self.loadbtn = QPushButton(self.centralwidget)
        self.loadbtn.setGeometry(QRect(1650,1165,100,50))
        self.loadbtn.setText("LOAD")
        self.loadbtn.clicked.connect(self.openLoadWindow)
        self.loadbtn.setCheckable(True)
        self.loadbtn.hide()

        self.loadWidget = QWidget(self.centralwidget)
        self.loadWidget.setGeometry(QRect(1750, 1100, self.saveloadbaseWidth, 180))
        self.loadWidget.hide()
        self.loadForm = QBoxLayout(QBoxLayout.TopToBottom)
        self.loadWidget.setLayout(self.loadForm)
        self.load_gb = QGroupBox()
        self.load_gb.setStyleSheet('color: #FFFFFF')
        self.load_gb.setTitle("LOAD")
        self.loadForm.addWidget(self.load_gb)
        self.loadLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.load_gb.setLayout(self.loadLayout)
        
        self.loadlabel = QLabel("불러올 데이터를 선택하세요.")
        self.loadLayout.addWidget(self.loadlabel)
        self.loadrbtn1 = QRadioButton("save data1")
        self.loadrbtn1.clicked.connect(self.loadrbtn1_func)
        self.loadLayout.addWidget(self.loadrbtn1)
        self.loadrbtn2 = QRadioButton("save data2")
        self.loadrbtn2.clicked.connect(self.loadrbtn2_func)
        self.loadLayout.addWidget(self.loadrbtn2)
        self.loadrbtn3 = QRadioButton("save data3")
        self.loadrbtn3.clicked.connect(self.loadrbtn3_func)
        self.loadLayout.addWidget(self.loadrbtn3)

        self.loadButton = QPushButton("LOAD")
        self.loadButton.clicked.connect(self.load_func)
        self.loadLayout.addWidget(self.loadButton)

        self.saveAni = QPropertyAnimation(self.saveWidget, b'geometry')
        self.loadAin = QPropertyAnimation(self.loadWidget, b'geometry')

        # ==============================================================================================================
        # Personal Color Recommend =====================================================================================
        self.aibtn = QPushButton(self.centralwidget)
        self.aibtn.setGeometry(QRect(2050,1110,100,50))
        self.aibtn.setText("AI추천")
        self.aibtn.clicked.connect(self.AIfunc)
        self.aibtn.hide()  
        # ==============================================================================================================
        # wishlist                 =====================================================================================
        self.wishlist = QPushButton(self.centralwidget)
        self.wishlist.setGeometry(QRect(2050,1165,100,50))
        self.wishlist.setText("장바구니")
        self.wishlist.clicked.connect(self.Wishlist_Func)
        self.wishlist.hide()  
        # ===============================================================================================================
        #메이크업 스레드 =================================================================================================
        '''MakeUpThread(self, skin_value = False, skin_ref_img = None, s_mix=0,
                                mouth_value = False, mouth_ref_img = None, m_mix=0,
                                eyebrow_value = False, left_eyevrow = None, right_eyebrow = None, eb_mix = 0, e_posx =0, e_posy=0, e_length=0,
                                cheek_value = False, left_cheek = None, right_cheek = None, c_posx=0, c_posy = 0, c_width = 0, c_height=0):'''
        self.face_make_thread = MakeUpThread(s_mix = self.s_mix, 
                                            m_mix= self.m_mix,
                                            eb_mix= self.eb_mix, e_posx= self.posx, e_posy= self.posy, e_length= self.length,
                                            c_mix = self.c_mix,c_posx= self.c_posx, c_posy = self.c_posy, c_width = self.c_width, c_height = self.c_height)
        # ==============================================================================================================
        # Health Care ==================================================================================================
        self.btnHealth = QPushButton()
        self.btnHealth.clicked.connect(self.healthcare)
        self.btnHealth.setCheckable(True)

        #create widget for Health Care function
        self.healthcareWidget = QWidget(self.centralwidget)
        self.healthcareWidget.setStyleSheet('background-color: #009900; border-style: solid; border-color: white; border-width: 3px; border-radius: 10px;')
        self.healthcarebaseWidth = 0
        self.healthcareextendedWidth = 520
        self.healthcareWidget.setGeometry(QRect(1700, 300, self.healthcarebaseWidth, 500))

        self.healthcareForm = QBoxLayout(QBoxLayout.TopToBottom)
        self.healthcareWidget.setLayout(self.healthcareForm)
        self.healthcareAni = QPropertyAnimation(self.healthcareWidget, b'geometry')
        
        self.healthcare_gb1 = QGroupBox()
        self.healthcare_gb1.setTitle("Stress")
        self.healthcare_gb1.setStyleSheet("QGroupBox { font: bold; font-size: 20px; margin-top: 6px; color: white; }"
                                        "QGroupBox::title { subcontrol-origin:  margin; subcontrol-position: top left; padding: 0 10px 0 10px;}")
        
        self.healthcareForm.addWidget(self.healthcare_gb1)
        self.stressLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.healthcare_gb1.setLayout(self.stressLayout)
        self.healthcare_gb1.setMaximumHeight(150)
        self.stress = QLabel()
        self.stress.setStyleSheet('color: white;'
                                'border-style: solid;'
                                'border-width: 5px;'
                                'border-color: white;')
        self.stress.setFont(QFont("맑은 고딕",20,QFont.Bold))
        self.stress.setMaximumHeight(100)
        self.stress.setAlignment(Qt.AlignCenter)
        self.stressLayout.addWidget(self.stress)
        

        self.healthcare_gb2 = QGroupBox()
        self.healthcare_gb2.setTitle("Melanoma")
        self.healthcare_gb2.setStyleSheet("QGroupBox { font: bold; font-size: 20px; margin-top: 6px; color: white; }"
                                        "QGroupBox::title { subcontrol-origin:  margin; subcontrol-position: top left; padding: 0 10px 0 10px;}")

        self.healthcareForm.addWidget(self.healthcare_gb2)
        self.melanoaLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.healthcare_gb2.setLayout(self.melanoaLayout)
        self.healthcare_gb2.setMaximumHeight(300)

        #흑색종
        self.btnMelanoma = QPushButton()
        self.btnMelanoma.clicked.connect(self.DetectMelanoma)
        self.btnMelanoma.setFont(QFont("맑은 고딕",20,QFont.Bold))
        self.btnMelanoma.setText("흑색종 탐색")
        self.btnMelanoma.setStyleSheet("QPushButton { color: white; background-color: #66FF66; border-radius: 30px;}")
        self.btnMelanoma.setMaximumHeight(100)
        self.melanoaLayout.addWidget(self.btnMelanoma)
        self.melanoma = QLabel()
        self.melanoma.setStyleSheet('color : white;'
                                    'border-style: solid;'
                                'border-width: 5px;'
                                'border-color: white;' 
                                'border-radius: 50px;')
        self.melanoma.setFont(QFont("맑은 고딕",20,QFont.Bold))
        self.melanoma.setMaximumHeight(100)
        self.melanoma.setAlignment(Qt.AlignCenter)
        self.melanoaLayout.addWidget(self.melanoma)

        self.StressDetectThread = DetectStressThread()
        self.StressDetectThread.call_Stress_value.connect(self.update_Stress)
        # ==============================================================================================================
        # Air Cleaner ==================================================================================================
        self.btnAircleanerShow = QPushButton()
        self.btnAircleanerShow.setCheckable(True)
        self.btnAircleanerShow.clicked.connect(self.AirCleanerFunc)
        self.aircleanerbaseWidth = 0
        self.airclenerextendedWidth = 500

        #Aircleaner ON/OFF
        self.aircleanerstate_Widget = QWidget(self.centralwidget)
        self.aircleanerstate_Widget
        self.aircleanerstate_Widget.setGeometry(QRect(1700, 300, self.aircleanerbaseWidth,170))
        self.aircleanerstate_Widget.setStyleSheet('background-color: #006666;'
                                            'border-style: solid;'
                                            'border-width: 3px;'
                                            'border-radius: 10px;'
                                            'border-color: white;')
        self.aircleanerstate_Layout = QFormLayout(self.centralwidget)

        self.aircleanerstate_ch = QLabel()
        self.aircleanerstate_ch.setAlignment(Qt.AlignCenter)
        self.aircleanerstate_ch.setStyleSheet('color : white;') 
        self.aircleanerstate_ch.setFont(QFont("맑은 고딕",25,QFont.Bold))

        self.aircleanerOn_btn = QPushButton()
        self.aircleanerOn_btn.setText("ON")
        self.aircleanerOn_btn.setStyleSheet("QPushButton { background-color: white; border-radius: 10px; color: #006666;}")
        self.aircleanerOn_btn.setFont(QFont("맑은 고딕",25,QFont.Bold))
        self.aircleanerOn_btn.setMinimumSize(240,25)
        self.aircleanerOn_btn.clicked.connect(self.aircleanerOn_Func)


        self.aircleanerOff_btn = QPushButton('OFF')
        self.aircleanerOff_btn.setStyleSheet("QPushButton { background-color: white; border-radius: 10px; color: #006666;}")
        self.aircleanerOff_btn.setFont(QFont("맑은 고딕",25,QFont.Bold))
        self.aircleanerOff_btn.setMaximumWidth(240)
        self.aircleanerOff_btn.clicked.connect(self.aircleanerOff_Func)

        self.aircleanerstate_Widget.setLayout(self.aircleanerstate_Layout)
        self.aircleanerstate_Layout.addRow(self.aircleanerstate_ch)
        self.aircleanerstate_Layout.addRow(self.aircleanerOn_btn, self.aircleanerOff_btn)
        self.aircleanerstate_Ani = QPropertyAnimation(self.aircleanerstate_Widget, b'geometry')

        #Aircleaner info
        self.aircleanerWidget = QWidget(self.centralwidget)
        self.aircleanerWidget.setGeometry(QRect(1700, 480, self.aircleanerbaseWidth,170))
        self.aircleanerWidget.setStyleSheet('background-color: #006666;'
                                            'border-style: solid;'
                                            'border-width: 3px;'
                                            'border-radius: 10px;'
                                            'border-color: white;')
        self.aircleanerLayout = QFormLayout(self.centralwidget)

        
        self.roomtemplabelName = QLabel()
        self.roomtemplabelName.setAlignment(Qt.AlignCenter)
        self.roomtemplabelName.setText('방       온도: ')
        self.roomtemplabelName.setStyleSheet('color : white;') 
        self.roomtemplabelName.setFont(QFont("맑은 고딕",25,QFont.Bold))
        self.roomtemplabel = QLabel()
        self.roomtemplabel.setAlignment(Qt.AlignVCenter)
        self.roomtemplabel.setStyleSheet('color : white;')
        self.roomtemplabel.setFont(QFont("맑은 고딕",25,QFont.Bold))
        
        self.roomhumiditylabelName = QLabel()
        self.roomhumiditylabelName.setAlignment(Qt.AlignCenter)
        self.roomhumiditylabelName.setText('방       습도: ')
        self.roomhumiditylabelName.setStyleSheet('color : white;') 
        self.roomhumiditylabelName.setFont(QFont("맑은 고딕",25,QFont.Bold))

        self.roomhumiditylabel = QLabel(self.centralwidget)
        self.roomhumiditylabel.setAlignment(Qt.AlignVCenter)
        self.roomhumiditylabel.setStyleSheet('color : white;')
        self.roomhumiditylabel.setFont(QFont("맑은 고딕",25,QFont.Bold))
        
        self.roomdustlabelName = QLabel()
        self.roomdustlabelName.setAlignment(Qt.AlignCenter)
        self.roomdustlabelName.setText('방 미세먼지: ')
        self.roomdustlabelName.setStyleSheet('color : white;') 
        self.roomdustlabelName.setFont(QFont("맑은 고딕",26,QFont.Bold))

        self.roomdustlabel = QLabel()
        self.roomdustlabel.setAlignment(Qt.AlignVCenter)
        self.roomdustlabel.setStyleSheet('color : white;')
        self.roomdustlabel.setFont(QFont("맑은 고딕",25,QFont.Bold))

        self.aircleanerLayout.addRow(self.roomtemplabelName, self.roomtemplabel)
        self.aircleanerLayout.addRow(self.roomhumiditylabelName, self.roomhumiditylabel)
        self.aircleanerLayout.addRow(self.roomdustlabelName, self.roomdustlabel)
        self.aircleanerWidget.setLayout(self.aircleanerLayout)
        self.aircleanerAni = QPropertyAnimation(self.aircleanerWidget, b'geometry')

        self.AirCleanerTread = AirCleanerTread()
        self.AirCleanerTread.call_AirCleaner_value.connect(self.update_AirCleaner)
        # ==============================================================================================================
        #DisplayOff button====================================================================================================
        self.DisplayOff =QPushButton(self.centralwidget)
        self.DisplayOff.clicked.connect(self.DisplayOffFunc)

        # ==============================================================================================================
        # button widget====================================================================================================
        self.mainbuttonWidget = QWidget(self.centralwidget)
        self.mainbuttonLayout = QBoxLayout(QBoxLayout.TopToBottom)
        self.mainbuttonWidget.setLayout(self.mainbuttonLayout)
        self.mainbuttonWidget.setGeometry(QRect(2450,325,80,0))
        self.mainbuttonLayout.addWidget(self.btnAircleanerShow)
        self.mainbuttonLayout.addWidget(self.btnHealth)
        self.mainbuttonLayout.addWidget(self.btnMakeUp)
        self.mainbuttonLayout.addWidget(self.DisplayOff)
        
        self.btnAircleanerShow.setMaximumHeight(60)
        self.btnAircleanerShow.setMaximumWidth(60)
        self.btnHealth.setMaximumHeight(60)
        self.btnHealth.setMaximumWidth(60)
        self.btnMakeUp.setMaximumHeight(60)
        self.btnMakeUp.setMaximumWidth(60)
        self.DisplayOff.setMaximumHeight(60)
        self.DisplayOff.setMaximumWidth(60)

        self.btnAircleanerShow.setIcon(aircleaner)
        self.btnAircleanerShow.setStyleSheet("QPushButton { background-color: #99CCFF; border-radius: 30px;}")
        self.btnAircleanerShow.setIconSize(QSize(45,45))
        self.btnHealth.setIcon(healthcare)
        self.btnHealth.setIconSize(QSize(43,43))
        self.btnHealth.setStyleSheet("QPushButton { background-color: #66FF66; border-radius: 30px;}")
        self.btnMakeUp.setIcon(makeup)
        self.btnMakeUp.setIconSize(QSize(45,45))
        self.btnMakeUp.setStyleSheet("QPushButton { background-color: #FF9999; border-radius: 30px;}")
        self.DisplayOff.setIcon(displayoff)
        self.DisplayOff.setIconSize(QSize(45,45))
        self.DisplayOff.setStyleSheet("QPushButton { background-color: black;}")

        self.mainbuttinAni = QPropertyAnimation(self.mainbuttonWidget, b"geometry")
        self.mainbuttinAni.setDuration(800)
        self.mainbuttinAni.setStartValue(QRect(QRect(2450,325,80,0)))
        self.mainbuttinAni.setEndValue(QRect(QRect(2450,325,80,550)))
        self.mainbuttinAni.start()

        self.btnAircleanerShow.setShortcut("Q")
        self.aircleanerOn_btn.setShortcut("W")
        self.aircleanerOff_btn.setShortcut("E")

        self.btnHealth.setShortcut("A")
        self.btnMelanoma.setShortcut("S")

        self.btnMakeUp.setShortcut("Z")
        self.changetab = QShortcut(QKeySequence("X"), self.face_makeTab, self.on_change_tab)
        self.changeshortcut = QShortcut(QKeySequence("C"), self.face_makeTab, self.on_change_shortcut)
        self.SkinColor1.setShortcut("1")
        self.MouthColor1.setShortcut("2")
        self.CheekColor1.setShortcut("3")
        self.EyeColor1.setShortcut("4")
        
        self.savebtn.setShortcut("/")
        self.loadbtn.setShortcut("*")
        self.aibtn.setShortcut("-")
        self.wishlist.setShortcut("+")

        
        self.DisplayOff.setShortcut("Esc")


    #-----------------------------------------------------------------------------------------
    # 이벤트
    # EVENT
    #-----------------------------------------------------------------------------------------
    def on_change_tab(self):
        if self.face_makeTab.currentIndex() == 3:
            self.face_makeTab.setCurrentIndex(0)
        else:
            self.face_makeTab.setCurrentIndex(self.face_makeTab.currentIndex()+1)
    def on_change_shortcut(self):
        if self.face_makeTab.currentIndex() == 0:
            self.s_DOWN.setShortcut("F1")
            self.s_UP.setShortcut("F2")
        elif self.face_makeTab.currentIndex() == 1:
            self.m_DOWN.setShortcut("F1")
            self.m_UP.setShortcut("F2")
        elif self.face_makeTab.currentIndex() == 2:
            self.c_DOWN.setShortcut("F1")
            self.c_UP.setShortcut("F2")
            self.c_xpos_DOWN.setShortcut("F3")
            self.c_xpos_UP.setShortcut("F4")
            self.c_ypos_DOWN.setShortcut("F5")
            self.c_ypos_UP.setShortcut("F6")
            self.c_width_DOWN.setShortcut("F7")
            self.c_width_UP.setShortcut("F8")
            self.c_height_DOWN.setShortcut("F9")
            self.c_height_UP.setShortcut("F10")
        elif self.face_makeTab.currentIndex() == 3:
            self.e_DOWN.setShortcut("F1")
            self.e_UP.setShortcut("F2")
            self.xpos_DOWN.setShortcut("F3")
            self.xpos_UP.setShortcut("F4")
            self.ypos_DOWN.setShortcut("F5")
            self.ypos_UP.setShortcut("F6")
            self.length_DOWN.setShortcut("F7")
            self.length_UP.setShortcut("F8")


    def sleep(self, n):
        QTest.qWait(n)


    # def keyPressEvent(self, event):
    #     super().keyPressEvent(event)
        
    #     if event.key() == Qt.Key_Escape:
    #         self.close()
    #     elif event.key() == Qt.Key_F:
    #         # self.DisplayOn.click()
    #         self.AirCleanerFunc()
    #     elif event.key() == Qt.Key_N:
    #         self.showNormal()
    #-------------------------------------------------------------------------------------------
    #DinsplayOn
    #-------------------------------------------------------------------------------------------
    def DisplayOnFunc(self, detect_face_value):
        self.detect_face_value = detect_face_value   
        #시작하기
        if self.detect_face_value ==1:
            self.DisplayOn.stop()
            self.sleep(600)
            self.setupUiOn(self.MainWindow)
    def diplayon_button_func(self):
        self.DisplayOn.stop()
        self.sleep(600)
        self.setupUiOn(self.MainWindow)

    def DisplayOffFunc(self):
        if self.StressDetectThread.isRunning():
            self.StressDetectThread.stop()
        if self.AirCleanerTread.isRunning():
            self.AirCleanerTread.stop()
        if self.face_make_thread.isRunning():
            self.face_make_thread.stop()
        self.set_weather.stop()
        self.sleep(600)
        self.setupUiOff(self.MainWindow)
        
        
    #-------------------------------------------------------------------------------------------
    #health Care Func
    #-------------------------------------------------------------------------------------------
    def healthcare(self, MainWindow):
        if self.btnHealth.isChecked():
            if self.btnAircleanerShow.isChecked():
                self.aircleanerWidget.hide()
                self.aircleanerstate_Widget.hide()
                self.AirCleanerTread.stop()
                self.btnAircleanerShow.setChecked(False)
                self.btnAircleanerShow.setStyleSheet("QPushButton { background-color: #99CCFF; border-radius: 30px;}")
            if self.btnMakeUp.isChecked():
                self.btnMakeUp.setChecked(False)
                self.btnMakeUp.setStyleSheet("QPushButton { background-color: #FF9999; border-radius: 30px;}")
                self.face_makeTab.hide()
                self.savebtn.hide()
                self.aibtn.hide()
                if self.savebtn.isChecked():
                    self.savebtn.setChecked(False)
                    self.saveWidget.hide()
                self.loadbtn.hide()
                if self.loadbtn.isChecked():
                    self.loadbtn.setChecked(False)
                    self.loadWidget.hide()
            self.healthcareAni.setDuration(800)
            self.healthcareAni.setStartValue(QRect(1700, 300, self.aircleanerbaseWidth,500))
            self.healthcareAni.setEndValue(QRect(1700, 300, self.airclenerextendedWidth,500))
            self.healthcareAni.start()
            self.btnHealth.setStyleSheet("QPushButton { background-color: #009900; border-radius: 30px;}")

            self.healthcareWidget.show()
            self.StressDetectThread.start()

        else:
            self.healthcareAni.setDuration(800)
            self.healthcareAni.setStartValue(QRect(1700, 300, self.airclenerextendedWidth,500))
            self.healthcareAni.setEndValue(QRect(1700, 300, self.aircleanerbaseWidth,500))
            self.healthcareAni.start()
            self.btnHealth.setStyleSheet("QPushButton { background-color: #66FF66; border-radius: 30px;}")
            # self.healthcareWidget.hide()
            self.StressDetectThread.stop()
        
    def update_Stress(self, bad_count):
        if bad_count == 0:
            self.stress.setText("스트레스지수 보통")
        elif bad_count == 1:
            self.stress.setText("스트레스지수 높음")
        else:
            self.stress.setText("스트레스지수 매우 높음")
        
    def DetectMelanoma(self, MainWindow):
        self.MelanomaDetectThread = DetectMelanomaThread()
        self.MelanomaDetectThread.call_Melanoma_value.connect(self.update_Melanoma)
        self.MelanomaDetectThread.start()
        
    def update_Melanoma(self, comp):
        if comp < 0.099:
            self.melanoma.setText("점이 비대칭, 흑색종 의심")
        else:
            self.melanoma.setText("점이 대칭, 흑색종 안심")


    #-------------------------------------------------------------------------------------------
    #Air Cleaner Func
    #-------------------------------------------------------------------------------------------
    def AirCleanerFunc(self, MainWindow):
        if self.btnAircleanerShow.isChecked():
            #헬스케어 버튼 비활성화 및 숨기기
            if self.btnHealth.isChecked():
                self.btnHealth.setStyleSheet("QPushButton { background-color: #66FF66; border-radius: 30px;}")
                self.btnHealth.setChecked(False)
                self.healthcareWidget.hide()
                self.StressDetectThread.stop()
            if self.btnMakeUp.isChecked():
                self.btnMakeUp.setChecked(False)
                self.face_makeTab.hide()
                self.savebtn.hide()
                self.btnMakeUp.setStyleSheet("QPushButton { background-color: #FF9999; border-radius: 30px;}")
                self.aibtn.hide()
                if self.savebtn.isChecked():
                    self.savebtn.setChecked(False)
                    self.saveWidget.hide()
                self.loadbtn.hide()
                if self.loadbtn.isChecked():
                    self.loadbtn.setChecked(False)
                    self.loadWidget.hide()
            self.aircleanerWidget.show()
            self.aircleanerstate_Widget.show()
            self.aircleanerAni.setDuration(800)
            self.aircleanerstate_Ani.setDuration(800)
            self.aircleanerAni.setStartValue(QRect(1700, 470, self.aircleanerbaseWidth,170))
            self.aircleanerstate_Ani.setStartValue(QRect(1700, 300, self.aircleanerbaseWidth,130))
            self.aircleanerAni.setEndValue(QRect(1700, 470, self.airclenerextendedWidth,170))
            self.aircleanerstate_Ani.setEndValue(QRect(1700, 300, self.airclenerextendedWidth,130))
            self.aircleanerAni.start()
            self.aircleanerstate_Ani.start()

            self.btnAircleanerShow.setStyleSheet("QPushButton { background-color: #99FFFF; border-radius: 30px}")

            fan_ref = db.reference('test/fan')
            fan_01 = fan_ref.get()
            if fan_01 == '0':
                self.aircleanerstate_ch.setText('공기 청정기 OFF')
            elif fan_01 == '1':
                self.aircleanerstate_ch.setText('공기 청정기 ON')

            
            self.AirCleanerTread.start()   
        else:
            self.btnAircleanerShow.setStyleSheet("QPushButton { background-color: #99CCFF; border-radius: 30px}")
            self.aircleanerAni.setDuration(800)
            self.aircleanerstate_Ani.setDuration(800)
            self.aircleanerAni.setStartValue(QRect(1700, 470, self.airclenerextendedWidth,170))
            self.aircleanerstate_Ani.setStartValue(QRect(1700, 300, self.airclenerextendedWidth,130))
            self.aircleanerAni.setEndValue(QRect(1700, 470, self.aircleanerbaseWidth,170))
            self.aircleanerstate_Ani.setEndValue(QRect(1700, 300, self.aircleanerbaseWidth,130))
            self.aircleanerAni.start()
            self.aircleanerstate_Ani.start()

            # self.aircleanerWidget.hide()
            self.AirCleanerTread.stop()
    
    def update_AirCleaner(self, temp, humidity, dust):
        self.roomtemplabel.setText(str(temp))
        self.roomhumiditylabel.setText(str(humidity))
        self.roomdustlabel.setText(str(dust))

    def aircleanerOn_Func(self):
        airpower.button_FBfan_ON()
        self.aircleanerstate_ch.setText('공기 청정기 ON')

    def aircleanerOff_Func(self):
        airpower.button_FBfan_OFF()
        self.aircleanerstate_ch.setText('공기 청정기 OFF')
    #-------------------------------------------------------------------------------------------
    #Set Time Func
    #-------------------------------------------------------------------------------------------
    def SetTimeFunc(self, year, month, day, EvenOrAfter, hour, minute):
        self.date.setText("%s년 %s월 %s일"%(str(year), str(month), str(day)))
        self.time.setText(" %s %s시 %s분" %(EvenOrAfter,str(hour),str(minute)))
    #-------------------------------------------------------------------------------------------
    #Set Weather Func
    #-------------------------------------------------------------------------------------------
    def SetWeatherFunc(self, temp, weathericon):
        self.temperature.setText("현재 온도 %s ℃ " %(str(temp)))
        self.weather.setPixmap(weathericon)
   
   #화장결과 숨기기 / 화장결과 보이기
    def facehide(self,MainWindow):
        if self.count % 2 == 0:
            self.face_make.show()
            self.count = self.count +1
        else:
            self.face_make.hide()
            self.count = self.count +1

    #-------------------------------------------------------------------------------------------
    #화장 기능
    #-------------------------------------------------------------------------------------------   
    #화장기능 활성화
    def makeupshow(self,MainWindow):
        if self.btnAircleanerShow.isChecked():
            self.aircleanerWidget.hide()
            self.aircleanerstate_Widget.hide()
            self.AirCleanerTread.stop()
            self.btnAircleanerShow.setChecked(False)
            self.btnAircleanerShow.setStyleSheet("QPushButton { background-color: #99CCFF; border-radius: 30px;}")

        if self.btnMakeUp.isChecked():
            #헬스케어 버튼 비활성화 및 숨기기
            if self.btnHealth.isChecked():
                self.btnHealth.setStyleSheet("QPushButton { background-color: #66FF66; border-radius: 30px;}")
                self.btnHealth.setChecked(False)
                self.healthcareWidget.hide()
                self.StressDetectThread.stop()
            self.face_makeTab.show()
            self.face_maketabAni.setDuration(800)
            self.face_maketabAni.setStartValue(QRect(1650, 300, self.face_maketabbaseWidth,800))
            self.face_maketabAni.setEndValue(QRect(1650, 300, self.face_maketabextendedWidth,800))
            self.face_maketabAni.start()
            
            self.btnMakeUp.setStyleSheet("QPushButton { background-color: #FF6666; border-radius: 30px;}")

            self.savebtn.show()
            self.loadbtn.show()
            self.aibtn.show()
            self.wishlist.show()
            
            #connect its signal to the update_makeup slot
            self.face_make_thread.change_pixmap_signal.connect(self.update_image)
            self.face_make_thread.start()
            self.showhideToggle.setChecked(True)
            
        else:
            self.face_maketabAni.setDuration(800)
            self.face_maketabAni.setStartValue(QRect(1650, 300, self.face_maketabextendedWidth,800))
            self.face_maketabAni.setEndValue(QRect(1650, 300, self.face_maketabbaseWidth,800))
            self.face_maketabAni.start()
            self.btnMakeUp.setStyleSheet("QPushButton { background-color: #FF9999; border-radius: 30px;}")
            # self.face_makeTab.hide()
            self.savebtn.hide()
            if self.savebtn.isChecked():
                self.savebtn.setChecked(False)
                self.saveWidget.hide()
            self.loadbtn.hide()
            if self.loadbtn.isChecked():
                self.loadbtn.setChecked(False)
                self.loadWidget.hide()
            self.aibtn.hide()
            
    def update_image(self, pixmap):
        self.face_make.setPixmap(pixmap)
    #AI추천 기능
    def AIfunc(self):
        self.personalcolorThread = DetectPersonalColor()
        self.personalcolorThread.call_tone_value.connect(self.update_tone)
        self.personalcolorThread.start()

    def update_tone(self, tone):
        if tone == 'spring':
            self.SkinColor1.setChecked(True)
            self.s_color1_start(MainWindow)
            self.MouthColor1.setChecked(True)
            self.m_color1_start(MainWindow)
        elif tone == 'summer':
            self.SkinColor2.setChecked(True)
            self.s_color2_start(MainWindow)
            self.MouthColor2.setChecked(True)
            self.m_color2_start(MainWindow)
        elif tone == 'fall':
            self.SkinColor3.setChecked(True)
            self.s_color3_start(MainWindow)
            self.MouthColor3.setChecked(True)
            self.m_color3_start(MainWindow)
        elif tone == 'winter':
            self.SkinColor4.setChecked(True)
            self.s_color4_start(MainWindow)
            self.MouthColor4.setChecked(True)
            self.m_color4_start(MainWindow)
    
    #wishlist
    def Wishlist_Func(self):
        if self.SkinColor1.isChecked():
            wish.button1(8)
        elif self.SkinColor2.isChecked():
            wish.button1(6) 
        elif self.SkinColor3.isChecked():
            wish.button1(10)
        elif self.SkinColor4.isChecked():
            wish.button1(7)
        elif self.SkinColor5.isChecked():
            wish.button1(9)

        if self.MouthColor1.isChecked():
            wish.button1(1)
        elif self.MouthColor2.isChecked():
            wish.button1(2)
        elif self.MouthColor3.isChecked():
            wish.button1(3)
        elif self.MouthColor4.isChecked():
            wish.button1(4)
        elif self.MouthColor5.isChecked():
            wish.button1(5)


    def s_Blendingval(self):                      #슬라이더를 투명도 값에 연동
        self.s_mix = self.s_Blending.value()
        # self.face_make_thread.pause = False
        self.face_make_thread.s_mix = self.s_mix
        # self.face_make_thread.pause = True


    def s_UpVal(self):                            #UP버튼 누르면 슬라이더 값 5씩 증가
        y = self.s_Blending.value()
        y = y + 5
        self.s_Blending.setValue(y)
    def s_DownVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.s_Blending.value()
        y = y - 5
        self.s_Blending.setValue(y)

    def s_color1(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/skin/spring.jpg') #임시
        self.face_make_thread.skin_ref_img = img
        self.face_make_thread.skin_value = True
        self.face_make_thread.pause = True


    def s_color2(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/skin/summer.jpg')
        self.face_make_thread.skin_ref_img = img
        self.face_make_thread.skin_value = True
        self.face_make_thread.pause = True   
   

    def s_color3(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/skin/fall.jpg')
        self.face_make_thread.skin_ref_img = img
        self.face_make_thread.skin_value = True
        self.face_make_thread.pause = True  


    def s_color4(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/skin/winter.jpg')
        self.face_make_thread.skin_ref_img = img
        self.face_make_thread.skin_value = True
        self.face_make_thread.pause = True



    def s_color5(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/skin/extra.jpg')
        self.face_make_thread.skin_ref_img = img
        self.face_make_thread.skin_value = True
        self.face_make_thread.pause = True

           
    def m_Blendingval(self):                      #슬라이더를 투명도 값에 연동
        self.m_mix = self.m_Blending.value()
        # self.face_make_thread.pause = False
        self.face_make_thread.m_mix = self.m_mix
        # self.face_make_thread.pause = True
    def m_UpVal(self):                            #UP버튼 누르면 슬라이더 값 5씩 증가
        y = self.m_Blending.value()
        y = y + 5
        self.m_Blending.setValue(y)
    def m_DownVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.m_Blending.value()
        y = y - 5
        self.m_Blending.setValue(y)

    def m_color1(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/lip/spring.jpg', cv2.IMREAD_UNCHANGED)
        self.face_make_thread.mouth_ref_img =img
        self.face_make_thread.mouth_value = True
        self.face_make_thread.pause = True


    def m_color2(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/lip/summer.jpg', cv2.IMREAD_UNCHANGED)
        self.face_make_thread.mouth_ref_img =img
        self.face_make_thread.mouth_value = True
        self.face_make_thread.pause = True


    def m_color3(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/lip/fall.jpg', cv2.IMREAD_UNCHANGED)
        self.face_make_thread.mouth_ref_img =img
        self.face_make_thread.mouth_value = True
        self.face_make_thread.pause = True

     
    def m_color4(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/lip/winter.jpg', cv2.IMREAD_UNCHANGED)
        self.face_make_thread.mouth_ref_img =img
        self.face_make_thread.mouth_value = True
        self.face_make_thread.pause = True

    
    def m_color5(self):
        self.face_make_thread.pause = False
        img = cv2.imread('makeup_img/lip/extra.jpg', cv2.IMREAD_UNCHANGED)
        self.face_make_thread.mouth_ref_img =img
        self.face_make_thread.mouth_value = True
        self.face_make_thread.pause = True


    def e_Blendingval(self):                      #슬라이더를 투명도 값에 연동
        self.eb_mix = self.e_Blending.value()
        self.face_make_thread.eb_mix = self.eb_mix
    def e_UpVal(self):                            #UP버튼 누르면 슬라이더 값 5씩 증가
        y = self.e_Blending.value()
        y = y + 5
        self.e_Blending.setValue(y)
    def e_DownVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.e_Blending.value()
        y = y - 5
        self.e_Blending.setValue(y)

    #눈썹 x위치
    def dyxVal(self):                      #눈썹의 가로 위치 값에 연동
        self.posx = self.dyxSlider.value() - 50
        self.face_make_thread.e_posx = self.posx
    def xpos_UPVal(self):
        x = self.dyxSlider.value()
        x = x + 5
        self.dyxSlider.setValue(x)
    def xpos_DOWNVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        x = self.dyxSlider.value()
        x = x - 5
        self.dyxSlider.setValue(x)

    # 눈썹 y위치
    def dyyVal(self):                      #눈썹의 세로 위치 값에 연동
        self.posy = self.dyySlider.value() - 50
        self.face_make_thread.e_posy = self.posy
    def ypos_UPVal(self):
        y = self.dyySlider.value()
        y = y + 5
        self.dyySlider.setValue(y)
    def ypos_DOWNVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.dyySlider.value()
        y = y - 5
        self.dyySlider.setValue(y)
    #눈썹의 길이
    def lengthVal(self):                      #눈썹의 길이 값에 연동
        self.length = self.lengthSlider.value()
        self.face_make_thread.e_length = self.length
    def length_UPVal(self):
        y = self.lengthSlider.value()
        y = y + 5
        self.lengthSlider.setValue(y)
    def length_DOWNVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.lengthSlider.value()
        y = y - 5
        self.lengthSlider.setValue(y)

    
    def e_color1(self):
        self.face_make_thread.pause = False
        img =  cv2.imread("makeup_img/eyebrown/eyebrow1.png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.left_eyebrow = cv2.flip(img,1)   #반전
        self.face_make_thread.right_eyebrow = img
        self.face_make_thread.eyebrow_value = True
        self.face_make_thread.pause = True

        

    def e_color2(self):
        self.face_make_thread.pause = False
        self.face_make_thread.left_eyebrow = cv2.imread("makeup_img/eyebrown/eyebrow2(left).png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.right_eyebrow = cv2.imread("makeup_img/eyebrown/eyebrow2(right).png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.eyebrow_value = True
        self.face_make_thread.pause = True

    def e_color3(self):
        self.face_make_thread.pause = False
        img =  cv2.imread("makeup_img/eyebrown/eyebrow3.png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.left_eyebrow = cv2.flip(img,1)   #반전
        self.face_make_thread.right_eyebrow = img
        self.face_make_thread.eyebrow_value = True
        self.face_make_thread.pause = True

    def e_color4(self):
        self.face_make_thread.pause = False
        img =  cv2.imread("makeup_img/eyebrown/eyebrow4.png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.left_eyebrow = cv2.flip(img,1)   #반전
        self.face_make_thread.right_eyebrow = img
        self.face_make_thread.eyebrow_value = True
        self.face_make_thread.pause = True        

    def e_color5(self):
        self.face_make_thread.pause = False
        img =  cv2.imread("makeup_img/eyebrown/eyebrow5.png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.left_eyebrow = cv2.flip(img,1)   #반전
        self.face_make_thread.right_eyebrow = img
        self.face_make_thread.eyebrow_value = True
        self.face_make_thread.pause = True
            
    def c_Blendingval(self):                      #슬라이더를 투명도 값에 연동
        self.c_mix = self.c_Blending.value()
        self.face_make_thread.c_mix = self.c_mix
    def c_UpVal(self):                            #UP버튼 누르면 슬라이더 값 5씩 증가
        y = self.c_Blending.value()
        y = y + 5
        self.c_Blending.setValue(y)
    def c_DownVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.c_Blending.value()
        y = y - 5
        self.c_Blending.setValue(y)

    #볼터치 x위치
    def c_dyxVal(self):                      #눈썹의 가로 위치 값에 연동
        self.c_posx = self.c_dyxSlider.value() - 50
        self.face_make_thread.c_posx = self.c_posx
    def c_xpos_UPVal(self):
        x = self.c_dyxSlider.value()
        x = x + 5
        self.c_dyxSlider.setValue(x)
    def c_xpos_DOWNVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        x = self.c_dyxSlider.value()
        x = x - 5
        self.c_dyxSlider.setValue(x)

    # 볼터치 y위치
    def c_dyyVal(self):                      #볼터치의 세로 위치 값에 연동
        self.c_posy = self.c_dyySlider.value() - 50
        self.face_make_thread.c_posy = self.c_posy
    def c_ypos_UPVal(self):
        y = self.c_dyySlider.value()
        y = y + 5
        self.c_dyySlider.setValue(y)
    def c_ypos_DOWNVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.c_dyySlider.value()
        y = y - 5
        self.c_dyySlider.setValue(y)

    #볼터치의 가로
    def c_widthVal(self):                      #볼터치의 크기 값에 연동
        self.c_width = self.c_widthSlider.value()
        self.face_make_thread.c_width = self.c_width
    def c_width_UPVal(self):
        y = self.c_widthSlider.value()
        y = y + 5
        self.c_widthSlider.setValue(y)
    def c_width_DOWNVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.c_widthSlider.value()
        y = y - 5
        self.c_widthSlider.setValue(y)
    #볼터치 세로
    def c_heightVal(self):                      #볼터치의 크기 값에 연동
        self.c_height = self.c_heightSlider.value()
        self.face_make_thread.c_width = self.c_height
    def c_height_UPVal(self):
        y = self.c_heightSlider.value()
        y = y + 5
        self.c_heightSlider.setValue(y)
    def c_height_DOWNVal(self):                          #DOWN버튼 누르면 슬라이더값 5씩 감소
        y = self.c_heightSlider.value()
        y = y - 5
        self.c_heightSlider.setValue(y)

    def c_color1(self):
        self.face_make_thread.pause = False
        self.face_make_thread.left_cheek = cv2.imread("makeup_img/cheek/circle_80alp.png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.right_cheek = cv2.imread("makeup_img/cheek/circle_80alp.png",cv2.IMREAD_UNCHANGED)
        self.face_make_thread.cheek_value = True
        self.face_make_thread.pause = True

    def c_color2(self):
        # self.cheek_value = True
        self.left_cheek = cv2.imread("makeup_img/cheek/eyeunder.png", cv2.IMREAD_UNCHANGED)
        self.right_cheek = cv2.flip(self.left_cheek, 1)         #좌우반전
        self.face_make_thread.pause = False
        self.face_make_thread.left_cheek = self.left_cheek
        self.face_make_thread.right_cheek = self.right_cheek
        self.face_make_thread.cheek_value = True
        self.face_make_thread.pause = True

    def c_color3(self):
        # self.cheek_value = True
        self.left_cheek = cv2.imread("makeup_img/cheek/long.png", cv2.IMREAD_UNCHANGED)
        self.right_cheek = cv2.flip(self.left_cheek, 1)         #좌우반전
        self.face_make_thread.pause = False
        self.face_make_thread.left_cheek = self.left_cheek
        self.face_make_thread.right_cheek = self.right_cheek
        self.face_make_thread.cheek_value = True
        self.face_make_thread.pause = True

    def c_color4(self):
        # self.cheek_value = True
        self.left_cheek = cv2.imread("makeup_img/cheek/short.png", cv2.IMREAD_UNCHANGED)
        self.right_cheek = cv2.flip(self.left_cheek, 1)         #좌우반전
        self.face_make_thread.pause = False
        self.face_make_thread.left_cheek = self.left_cheek
        self.face_make_thread.right_cheek = self.right_cheek
        self.face_make_thread.cheek_value = True
        self.face_make_thread.pause = True
        
    def openSaveWindow(self):
        if self.loadbtn.isChecked():
            self.loadWidget.hide()
            self.loadbtn.setChecked(False)
        skin = self.SkincolorCheck()
        mouth = self.MouthcolorCheck()
        eye = self.EyecolorCheck()
        global savelst
        savelst = [skin, mouth, eye]
        self.saveWidget.show()
        self.saveAni.setDuration(800)
        self.saveAni.setStartValue(QRect(1750, 1100, self.saveloadbaseWidth, 180))
        self.saveAni.setEndValue(QRect(1750, 1100, self.saveloadextendedWidth, 180))
        self.saveAni.start()
        if self.savebtn.isChecked() == False:
            self.saveAni.setDuration(800)
            self.saveAni.setStartValue(QRect(1750, 1100, self.saveloadextendedWidth, 180))
            self.saveAni.setEndValue(QRect(1750, 1100, self.saveloadbaseWidth, 180))
            self.saveAni.start()
        
    def saverbtn1_func(self):
        self.savedata = 'savedata1'
        return self.savedata
    def saverbtn2_func(self):
        self.savedata = 'savedata2'
        return self.savedata
    def saverbtn3_func(self):
        self.savedata = 'savedata3'
        return self.savedata
    
    def save_func(self):
        df[self.savedata] = savelst
        df.to_excel("savedata.xlsx", index=False)
        self.saveWidget.hide()
        self.savebtn.setChecked(False)

    def loadrbtn1_func(self):
        self.loaddata = 'savedata1'
        return self.loaddata
    def loadrbtn2_func(self):
        self.loaddata = 'savedata2'
        return self.loaddata
    def loadrbtn3_func(self):
        self.loaddata = 'savedata3'
        return self.loaddata
    
    def load_func(self):
        global loadlst
        loadlst = list(df[self.loaddata].head(3))

        a = loadlst[0]
        b = loadlst[1]
        c = loadlst[2]
        # print(a,b,c)
        if a == 'self.SkinColor1':
            self.SkinColor1.setChecked(True)
            self.s_color1_start(MainWindow)
        elif a == 'self.SkinColor2':
            self.SkinColor2.setChecked(True)
            self.s_color2_start(MainWindow)
        elif a == 'self.SkinColor3':
            self.SkinColor3.setChecked(True)
            self.s_color3_start(MainWindow)
        elif a == 'self.SkinColor4':
            self.SkinColor4.setChecked(True)
            self.s_color4_start(MainWindow)
        elif a == 'self.SkinColor5':
            self.SkinColor5.setChecked(True)
            self.s_color5_start(MainWindow)

        if b == 'self.MouthColor1':
            self.SkinColor1.setChecked(True)
            self.m_color1_start(MainWindow)
        elif b == 'self.MouthColor2':
            self.MouthColor2.setChecked(True)
            self.m_color2_start(MainWindow)
        elif b == 'self.MouthColor3':
            self.SkinColor3.setChecked(True)
            self.m_color3_start(MainWindow)
        elif b == 'self.MouthColor4':
            self.MouthColor4.setChecked(True)
            self.m_color4_start(MainWindow)
        elif b == 'self.MouthColor5':
            self.MouthColor5.setChecked(True)
            self.m_color5_start(MainWindow)

        if c == 'self.EyeColor1':
            self.EyeColor1.setChecked(True)
            self.e_color1_start(MainWindow)
        elif c == 'self.EyeColor2':
            self.e_color2_start(MainWindow)
            self.EyeColor2.setChecked(True)
        elif c == 'self.EyeColor3':
            self.EyeColor3.setChecked(True)
            self.e_color3_start(MainWindow)
        elif c == 'self.EyeColor4':
            self.EyeColor4.setChecked(True)
            self.e_color2_start(MainWindow)
        elif c == 'self.EyeColor5':
            self.EyeColor5.setChecked(True)
            self.e_color3_start(MainWindow)
        
        self.loadWidget.hide()
        self.loadbtn.setChecked(False)

    
    def openLoadWindow(self):
        if self.savebtn.isChecked():
            self.saveWidget.hide()
            self.savebtn.setChecked(False)

        self.loadWidget.show()
        self.loadAin.setDuration(800)
        self.loadAin.setStartValue(QRect(1750, 1100, self.saveloadbaseWidth, 180))
        self.loadAin.setEndValue(QRect(1750, 1100, self.saveloadextendedWidth, 180))
        self.loadAin.start()
        if self.loadbtn.isChecked() == False:
            self.loadAin.setDuration(800)
            self.loadAin.setStartValue(QRect(1750, 1100, self.saveloadextendedWidth, 180))
            self.loadAin.setEndValue(QRect(1750, 1100, self.saveloadbaseWidth, 180))
            self.loadAin.start()

        
        
    def SkincolorCheck(self):
        if self.SkinColor1.isChecked():
            skin = "self.SkinColor1"
            return skin
        elif self.SkinColor2.isChecked():
            skin = "self.SkinColor2"
            return skin
        elif self.SkinColor3.isChecked():
            skin = "self.SkinColor3"
            return skin
        elif self.SkinColor4.isChecked():
            skin = "self.SkinColor4"
            return skin
        elif self.SkinColor5.isChecked():
            skin = "self.SkinColor5"
            return skin
        else:
            skin = "self.SkinColor1"
            return skin

    def MouthcolorCheck(self):
        if self.MouthColor1.isChecked():
            mouth= "self.MouthColor1"
            return mouth
        elif self.MouthColor2.isChecked():
            mouth = "self.MouthColor2"
            return mouth
        elif self.MouthColor3.isChecked():
            mouth= "self.MouthColor3"
            return mouth
        elif self.MouthColor4.isChecked():
            mouth= "self.MouthColor4"
            return mouth
        elif self.MouthColor5.isChecked():
            mouth= "self.MouthColor5"
            return mouth
        else:
            mouth= "self.MouthColor1"
            return mouth

    def EyecolorCheck(self):
        if self.EyeColor1.isChecked():
            eye= "self.EyeColor1"
            return eye
        elif self.EyeColor2.isChecked():
            eye= "self.EyeColor2"
            return eye
        elif self.EyeColor3.isChecked():
            eye= "self.EyeColor3"
            return eye
        elif self.EyeColor4.isChecked():
            eye= "self.EyeColor4"
            return eye
        elif self.EyeColor5.isChecked():
            eye= "self.EyeColor5"
            return eye
        else:
            eye= "self.EyeColor1"
            return eye
        
    
    
    #피부 색깔 변경 기능들을 스레드로 사용--------------------------------------------------------------------
    def s_color1_start(self,MainWindow):
        self.SkinColor2.setChecked(False)
        self.SkinColor5.setChecked(False)
        self.SkinColor3.setChecked(False)
        self.SkinColor4.setChecked(False)
        self.SkinColor5.setChecked(False)
        # print(self.SkinColor5.isChecked())
        thread=Thread(target=self.s_color1)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def s_color2_start(self,MainWindow):
        self.SkinColor1.setChecked(False)
        self.SkinColor5.setChecked(False)
        self.SkinColor3.setChecked(False)
        self.SkinColor4.setChecked(False)
        thread=Thread(target=self.s_color2)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
        
    def s_color3_start(self,MainWindow):
        self.SkinColor1.setChecked(False)
        self.SkinColor2.setChecked(False)
        self.SkinColor5.setChecked(False)
        self.SkinColor4.setChecked(False)
        thread=Thread(target=self.s_color3)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def s_color4_start(self,MainWindow):
        self.SkinColor1.setChecked(False)
        self.SkinColor2.setChecked(False)
        self.SkinColor3.setChecked(False)
        self.SkinColor5.setChecked(False)
        thread=Thread(target=self.s_color4)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def s_color5_start(self,MainWindow):
        self.SkinColor1.setChecked(False)
        self.SkinColor2.setChecked(False)
        self.SkinColor3.setChecked(False)
        self.SkinColor4.setChecked(False)
        thread=Thread(target=self.s_color5)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
    
    #입술 색깔 변경 기능들을 스레드로 사용--------------------------------------------------------------------
    def m_color1_start(self,MainWindow):
        self.MouthColor2.setChecked(False)
        self.MouthColor3.setChecked(False)
        self.MouthColor4.setChecked(False)
        self.MouthColor5.setChecked(False)
        thread=Thread(target=self.m_color1)
        # thread.daemon=True #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def m_color2_start(self,MainWindow):
        self.MouthColor1.setChecked(False)
        self.MouthColor3.setChecked(False)
        self.MouthColor4.setChecked(False)
        self.MouthColor5.setChecked(False)
        thread=Thread(target=self.m_color2)
        # thread.daemon=True #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def m_color3_start(self,MainWindow):
        self.MouthColor1.setChecked(False)
        self.MouthColor2.setChecked(False)
        self.MouthColor4.setChecked(False)
        self.MouthColor5.setChecked(False)
        thread=Thread(target=self.m_color3)
        # thread.daemon=True #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def m_color4_start(self,MainWindow):
        self.MouthColor1.setChecked(False)
        self.MouthColor2.setChecked(False)
        self.MouthColor3.setChecked(False)
        self.MouthColor5.setChecked(False)
        thread=Thread(target=self.m_color4)
        # thread.daemon=True #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def m_color5_start(self,MainWindow):
        self.MouthColor1.setChecked(False)
        self.MouthColor2.setChecked(False)
        self.MouthColor3.setChecked(False)
        self.MouthColor4.setChecked(False)
        thread=Thread(target=self.m_color5)
        # thread.daemon=True #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
    
    #눈썹
    def e_color1_start(self, MainWindow):
        self.EyeColor2.setChecked(False)
        self.EyeColor3.setChecked(False)
        self.EyeColor4.setChecked(False)
        self.EyeColor5.setChecked(False)
        thread=Thread(target=self.e_color1)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def e_color2_start(self, MainWindow):
        self.EyeColor1.setChecked(False)
        self.EyeColor3.setChecked(False)
        self.EyeColor4.setChecked(False)
        self.EyeColor5.setChecked(False)
        thread=Thread(target=self.e_color2)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def e_color3_start(self, MainWindow):
        self.EyeColor1.setChecked(False)
        self.EyeColor2.setChecked(False)
        self.EyeColor4.setChecked(False)
        self.EyeColor5.setChecked(False)
        thread=Thread(target=self.e_color3)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
    
    def e_color4_start(self, MainWindow):
        self.EyeColor1.setChecked(False)
        self.EyeColor2.setChecked(False)
        self.EyeColor3.setChecked(False)
        self.EyeColor5.setChecked(False)
        thread=Thread(target=self.e_color4)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def e_color5_start(self, MainWindow):
        self.EyeColor1.setChecked(False)
        self.EyeColor2.setChecked(False)
        self.EyeColor3.setChecked(False)
        self.EyeColor4.setChecked(False)
        thread=Thread(target=self.e_color5)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def c_color1_start(self, MainWindow):
        self.CheekColor2.setChecked(False)
        self.CheekColor3.setChecked(False)
        self.CheekColor4.setChecked(False)
        thread=Thread(target=self.c_color1)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
    
    def c_color2_start(self, MainWindow):
        self.CheekColor1.setChecked(False)
        self.CheekColor3.setChecked(False)
        self.CheekColor4.setChecked(False)
        thread=Thread(target=self.c_color2)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

    def c_color3_start(self, MainWindow):
        self.CheekColor1.setChecked(False)
        self.CheekColor2.setChecked(False)
        self.CheekColor4.setChecked(False)
        thread=Thread(target=self.c_color3)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
        
    def c_color4_start(self, MainWindow):
        self.CheekColor1.setChecked(False)
        self.CheekColor2.setChecked(False)
        self.CheekColor3.setChecked(False)
        thread=Thread(target=self.c_color4)
        thread.daemon=False #프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()

#-------------메인---------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow(MainWindow)
    # ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())