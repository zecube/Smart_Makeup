from __future__ import print_function

import  dlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import timeit

detector = dlib.get_frontal_face_detector()
 
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture(0)    # 캠 저장

# range는 끝값이 포함안됨   
ALL = list(range(0, 68))                # 0~67
RIGHT_EYEBROW = list(range(17, 22))     # 오른쪽 눈섭   17~21
LEFT_EYEBROW = list(range(22, 27))      # 왼쪽 눈섭     22~26
RIGHT_EYE = list(range(36, 42))         # 오른눈        36~41
LEFT_EYE = list(range(42, 48))          # 왼눈          42~47
NOSE = list(range(27, 36))              # 코            27~35
MOUTH_OUTLINE = list(range(48, 60))     # 입 밖         48~60
MOUTH_INNER = list(range(60, 68))       # 입 안         61~67
JAWLINE = list(range(0, 17))            # 턱 라인       00~16

index = ALL

faceCount = 0

#==================================================
while True:
    ret, img_frist = cap.read()         # 캠 화면 read()
    if ret is True:
        start_t = timeit.default_timer()
        img_frame = cv.resize(img_frist,(1000,800))  # 캠 화면 키우기
        img_frame_copy = img_frame.copy()
        mask = np.zeros_like(img_frame)             # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        face_mask = np.zeros_like(img_frame)        # 캠 크기와 같은 검정색으로 이루어진 화면 (얼굴만 마스킹 사용시 사용됨)


        img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)        # 회색
        copy =img_frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
        copy1 = img_frame.copy()


        dets = detector(img_gray, 1)        # 회색
        for face in dets:
            # print('인식중\t')                   # 얼굴인식 될 시 출력
            faceCount = faceCount + 1

            shape = predictor(img_frame, face) #얼굴에서 68개 점 찾기

            list_points = []                # 아래의 for문으로 넘파이 상태로 좌표 저장(총 68개)
            for p in shape.parts():
                list_points.append([p.x, p.y])                  # x,y좌표 리스트에 추가

            list_points = np.array(list_points)                 # 넘파이 형태로 얼굴좌표 저장

            for i,pt in enumerate(list_points[index]):              # 좌표에 원 설정
                pt_pos = (pt[0], pt[1])                             # 원의 좌표 설정
                cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)    # 원 나오기

            # 얼굴인식한 박스 나오기
            # cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
            #     (0, 0, 255), 3)

        cv.imshow('img_frame', img_frame)

        if faceCount > 4:
            # 라인 도형

            # # # 얼굴라인
            lineAdd(JAWLINE)
            # #  눈섭 둘
            # RIGHT_EYEBROW_line = EYEBROW_lineAdd(RIGHT_EYEBROW)
            # LEFT_EYEBROW_line = EYEBROW_lineAdd(LEFT_EYEBROW)
            # 입술
            # Outline_line, Inner_line = MOUTH_lineAdd(MOUTH_OUTLINE, MOUTH_INNER)
            # MOUTH_lineAdd(MOUTH_OUTLINE, MOUTH_INNER)
            
            mask_copy = mask.copy
            mask_copy =~mask
            # #  마스킹 하기
            masked = cv.bitwise_or(copy, mask_copy)
             
            matched = match_histograms(masked,ref_img, channel_axis= -1)
            # masked = cv.bitwise_and(copy, matched)
            out = cv.bitwise_and(matched,mask)
            result = cv.bitwise_and(copy,mask_copy)
            result = cv.bitwise_or(out,result)
            cv.imshow('out', mask_copy)
            cv.imshow('result', result)
            # color = cv.bitwise_and(mouth_color1,masked)

        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t ))
        print(FPS)     
    
    
        
  


    # 기존에 있던 랜드마크에 중점 3개씩 추가
    def lineAdd(list):  # 얼굴 가능 (순서대로 그려지는 모양)
        line = np.array([ list_points[list[0]] ])

        for i in list:
            if i < list[-1] : # 말전까지만 for문 실행되도록
                from_point = list_points[i] # 0 지점
                to_point = list_points[i+1] # 1 지점
                middle_point = np.array( [(from_point[0]+to_point[0])//2,(from_point[1]+to_point[1])//2] ) # 0.5지점, from_point와 to_point의 중점

                middle_point1 = np.array( [(from_point[0]+middle_point[0])//2,(from_point[1]+middle_point[1])//2] ) # 0.25 지점
                middle_point2 = np.array( [(middle_point[0]+to_point[0])//2  ,  (middle_point[1]+to_point[1])//2] ) # 0.75 지점
                
                line = np.append(line,[middle_point1,middle_point,middle_point2,to_point], axis=0) # 0.25, 0.50, 0.75. 1.00 지점 저장(추가)
        
        cv.fillConvexPoly(mask,line,(255, 255, 255))    # 리스트 모양으로 다각형 형성

    def EYEBROW_lineAdd(list):  # 눈썹만 사용가능
        line1 = np.array([ list_points[list[0]] ]) # 눈섭 좌표 리스트 생성(처음 지점 추가)
        line2 = np.array([ list_points[list[0]] ])

        for i in list:
            if i < list[-1] : # 말전까지만 for문 실행되도록
                from_point = list_points[i] # 0 지점
                to_point = list_points[i+1] # 1 지점
                middle_point = np.array( [(from_point[0]+to_point[0])//2,(from_point[1]+to_point[1])//2] ) # 0.5 지점, from_point와 to_point의 중점, 
                # 눈섭 중점 높이값 구하기
                top_left_LEFT_EYBROW = np.min(list_points[LEFT_EYEBROW], axis=0)        # 눈섭 좌표 최대높이 구하기
                bottom_right_LEFT_EYBROW = np.max(list_points[LEFT_EYEBROW], axis=0)    # 눈섭 좌표 최소높이 구하기
                middle_LEFT_EYEBOW = (top_left_LEFT_EYBROW[1] - bottom_right_LEFT_EYBROW[1])//10     # 눈섭의 높이값, 눈섭의 최대 최소 높이 빼기(대강 10나눔)

                middle_point1 = np.array( [(from_point[0]+middle_point[0])//2,(from_point[1]+middle_point[1])//2] ) # 0.25 지점
                middle_point2 = np.array( [(middle_point[0]+to_point[0])//2  ,  (middle_point[1]+to_point[1])//2] ) # 0.75 지점
                # 눈섭의 선 윗부분
                up_middle_point1 = np.array( [ middle_point1[0], middle_point1[1]+middle_LEFT_EYEBOW ] )                        # 위 0.25 지점
                up_middle_point =  np.array( [ middle_point[0], middle_point[1]+middle_LEFT_EYEBOW ])                           # 위 0.5 지점
                up_middle_point2 = np.array( [ middle_point2[0], middle_point2[1]+middle_LEFT_EYEBOW ] )                        # 위 0.75 지점
                up_to_point =  np.array( [ to_point[0], to_point[1]+middle_LEFT_EYEBOW ] )                                      # 위 1 지점

                # 눈섭의 선 아랫부분
                down_middle_point1 = np.array( [ middle_point1[0], middle_point1[1]-middle_LEFT_EYEBOW ] )                      # 아래 0.25 지점
                down_middle_point =  np.array( [ middle_point[0], middle_point[1]-middle_LEFT_EYEBOW ])                         # 아래 0.5 지점
                down_middle_point2 = np.array( [ middle_point2[0], middle_point2[1]-middle_LEFT_EYEBOW ] )                      # 아래 0.75 지점
                down_to_point =  np.array( [ to_point[0], to_point[1]-middle_LEFT_EYEBOW ] )                               # 아래 1 지점


                line1 = np.append(line1,[up_middle_point1,up_middle_point,up_middle_point2,up_to_point], axis=0) # 0.25, 0.75. 1.00 지점 저장(추가)
                line2 = np.append(line2,[down_middle_point1,down_middle_point,down_middle_point2,down_to_point], axis=0) # 0.25, 0.75. 1.00 지점 저장(추가)
        line2 = line2[1:-1] # line1과 line2의 중복되는 부분을 뺀다
        line2 = np.flipud(line2) # line2를 반전
        line = np.concatenate((line1,line2),axis=0) # line1과 line2를 합친다.
        cv.fillConvexPoly(mask,line,(255, 255, 255))    # 눈섭 모양으로 다각형 형성
        # return line

    def MOUTH_lineAdd(Outline_list,Inner_list):    # 입술만 사용가능, Outline_list : 입술 밖, Inner_list : 입술 안
        Outline_line = np.array([ list_points[Outline_list[0]] ]) # 입술 밖 좌표 리스트 생성(처음 지점 추가)
        Inner_line = np.array([ list_points[Inner_list[0]] ]) # 입술 안 좌표 리스트 생성(처음 지점 추가)

        # 입술 밖 추가
        for i in Outline_list:
            if i < Outline_list[-1] : # 말전까지만 for문 실행되도록
                from_point = list_points[i] # 0 지점
                to_point = list_points[i+1] # 1 지점
                middle_point = np.array( [(from_point[0]+to_point[0])//2,(from_point[1]+to_point[1])//2] ) # from_point와 to_point의 중점, 0.5지점

                middle_point1 = np.array( [(from_point[0]+middle_point[0])//2,(from_point[1]+middle_point[1])//2] ) # 0.25 지점
                middle_point2 = np.array( [(middle_point[0]+to_point[0])//2  ,  (middle_point[1]+to_point[1])//2] ) # 0.75 지점
                
                Outline_line = np.append(Outline_line,[middle_point1,middle_point,middle_point2,to_point], axis=0) # Outline_line에 0.25, 0.75. 1.00 지점 저장(추가)
        # 입술 안 추가
        for i in Inner_list:
            if i < Inner_list[-1] : # 말전까지만 for문 실행되도록
                from_point = list_points[i] # 0 지점
                to_point = list_points[i+1] # 1 지점
                middle_point = np.array( [(from_point[0]+to_point[0])//2,(from_point[1]+to_point[1])//2] ) # from_point와 to_point의 중점, 0.5지점

                middle_point1 = np.array( [(from_point[0]+middle_point[0])//2,(from_point[1]+middle_point[1])//2] ) # 0.25 지점
                middle_point2 = np.array( [(middle_point[0]+to_point[0])//2  ,  (middle_point[1]+to_point[1])//2] ) # 0.75 지점
                
                Inner_line = np.append(Inner_line,[middle_point1,middle_point,middle_point2,to_point], axis=0) # Inner_line에 0.25, 0.75. 1.00 지점 저장(추가)

        
        cv.fillConvexPoly(mask,Outline_line,(255, 255, 255))    # 입 밖은 흰색으로 다각형 형성
        cv.fillConvexPoly(mask,Inner_line,(0, 0, 0))            # 입 내부는 검정으로 다각형 형성

    
    if cv.waitKey(1) == ord('q'):
        break