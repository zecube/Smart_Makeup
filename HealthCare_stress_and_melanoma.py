## utils.py, HairSegmentation.py, emotion_model.hdf5, song_1.mp3, 점(흑색종) 이미지파일.jpg 필요

import cv2 as cv
import mediapipe as mp
import numpy as np
import utils

# pip install pygame
import pygame
# from time import sleep # 이거 사용해서 sleep() 써보니 화면이 그만큼 멈추더라/못쓸듯
from PIL import Image
from keras.models import load_model



def emotion_song(results, frame, gray_img):
    pre_bad_count = 0
    bad_count = 0

    emotion_classifier = load_model('models\emotion_model.hdf5', compile=False)    # 사용되는 데이터셋 emotion_model.hdf5
    EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]                
    
    if results.detections:
        detections = sorted(results.detections, key=lambda x: x.score[0], reverse=True)

        # Get the largest face / FaceDetection의 직사각형 사용
        face = detections[0].location_data.relative_bounding_box
        (im_height, im_width, _) = frame.shape
        (fX, fY, fW, fH) = (int(face.xmin * im_width), int(face.ymin * im_height),
                            int(face.width * im_width), int(face.height * im_height))

        # Resize the image to 48x48 for neural network
        roi = gray_img[fY:fY + fH, fX:fX + fW]
        roi = cv.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        pil_img = Image.fromarray(roi)
        roi = np.asarray(pil_img)
        roi = np.expand_dims(roi, axis=0)
        
        #  감정을 예측합니다. / Emotion predict
        preds = emotion_classifier.predict(roi)[0]
                            
        # 레이블을 출력합니다. / Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            
            # 감정 저장
            # 화날 때, bad_count 증가 
            if emotion == 'Angry':
                Angry_prob = prob # 출력값은 0.00~1.00 까지로 나옴 
                # print('Angry_prob',Angry_prob)
                if Angry_prob > 0.50:
                    bad_count = bad_count + 1
                    # print('angry',bad_count)
            # # if emotion == 'Disgusting':
            # #     Disgusting_prob = prob # 출력값은 0.00~1.00 까지로 나옴 
            # #     # print('Disgusting_prob',Disgusting_prob)
            # # if emotion == 'Fearful':
            # #     Fearful_prob = prob # 출력값은 0.00~1.00 까지로 나옴 
            # #     # print('Fearful_prob',Fearful_prob)
            # # if emotion == 'happy':
            # #     happy_prob = prob # 출력값은 0.00~1.00 까지로 나옴 
            # #     # print('happy_prob',happy_prob)
                
            # 슬플 때, bad_count 증가 / 
            if emotion == 'Sad':
                Sad_prob = prob # 출력값은 0.00~1.00 까지로 나옴 
                # print('Sad_prob',Sad_prob)
                if Sad_prob > 0.50:
                    bad_count = bad_count + 1
                    # print('sad',bad_count)
                    
            # # if emotion == 'Surpring':
            # #     Surpring_prob = prob # 출력값은 0.00~1.00 까지로 나옴 
            # #     # print('Surpring_prob',Surpring_prob)
            # # if emotion == 'Neutral':
            # #     Neutral_prob = prob # 출력값은 0.00~1.00 까지로 나옴 
            # #     # print('Neutral_prob',Neutral_prob)
            # #     # print('\n')

            # 노래 출력
            # bad_count가 일정값만큼 높아진 경우, bad_count = 0으로 바꾸고 노래 출력
            if bad_count == 2:
                # print(bad_count)
                pygame.mixer.init()                     # 파이게임믹서 초기화
                pygame.mixer.music.load("song_1.mp3")   # 음악로드
                pygame.mixer.music.play()               # 음악재생
                # sleep(10)                             # time만큼 멈춤 / time쓰면 imshow의 화면이 멈춤
                bad_count = 0
    return bad_count

def melanoma(img):
    # path = '1111.jpg'
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    ret,thresh = cv.threshold(blur,70,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # im2, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_cnt = max(contours, key=cv.contourArea)

    ellipse = cv.fitEllipse(max_cnt)
    ellipse_pnts = cv.ellipse2Poly( (int(ellipse[0][0]),int(ellipse[0][1]) ) ,( int(ellipse[1][0]),int(ellipse[1][1]) ),int(ellipse[2]),0,360,1)
    comp = cv.matchShapes(max_cnt,ellipse_pnts,1,0.0)

    return comp


# draw_FACE_OVAL_hairless(copy)

# bad_count = emotion_song(bad_count)

# melanoma('1111.jpg')  # 여러번출력댐
            

