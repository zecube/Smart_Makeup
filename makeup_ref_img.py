from __future__ import print_function
import cv2 as cv
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
from HairSegmentation import HairSegmentation

import utils

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
    cv.fillPoly(overlay,[list_to_np_array], color )
    new_img = cv.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    # print(points_list)
    img = new_img
    cv.polylines(img, [list_to_np_array], True, color,1, cv.LINE_AA)
    return img

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

map_face_mesh = mp.solutions.face_mesh
face_mesh = map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5)

    

def draw_LIPS(img,colors):    # 입술
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LIPS], colors, opacity=1 )
        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_or(copy, mask_copy)
        return masked # , masked
    

def draw_FACE_OVAL(img,colors):   # 얼굴 라인
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    # Inialize hair segmentation model / 머리카락 세분화 모델을 초기화합니다.
    hair_segmentation = HairSegmentation()
    
    
    blackbox = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
    
    hair_mask = hair_segmentation(frame)

    # Get dyed frame. / 염료 처리된 프레임을 가져옵니다.

    dyed_image = np.zeros_like(frame)
    dyed_image[:] = 255, 255, 255
    dyed_frame = dyed_image

    # Mask our dyed frame (pixels out of mask are black). / 염료 처리된 프레임에서 마스크를 적용합니다 (마스크 밖의 픽셀은 검은색).
    dyed_hair = cv.bitwise_or(frame, dyed_frame, mask=hair_mask) ################### 이거로 #########################
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
        # mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LIPS], (0,0,0), opacity=1 )  # 입술만 뺌
        # mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in CLOSE_LIP], (0,0,0), opacity=1 )   # (입술+입술 내부) 포함해서 뺌
        # # 눈 뺄 시
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYE], (0,0,0), opacity=1 )
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYE], (0,0,0), opacity=1 )
        # # # 눈썹 뺄 시
        # mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYEBROW], (0,0,0), opacity=1 )
        # mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYEBROW], (0,0,0), opacity=1 )
        # # 머리
        mask = cv.bitwise_and(mask, dyed_hair)

        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_or(copy, mask_copy)
        # cv.imshow('f',masked)
        return masked # , masked
    
def draw_FACE_example(img,colors):   # 얼굴 라인
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    # Inialize hair segmentation model / 머리카락 세분화 모델을 초기화합니다.
    hair_segmentation = HairSegmentation()
    
    
    blackbox = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
    
    hair_mask = hair_segmentation(frame)

    # Get dyed frame. / 염료 처리된 프레임을 가져옵니다.

    dyed_image = np.zeros_like(frame)
    dyed_image[:] = 255, 255, 255
    dyed_frame = dyed_image

    # Mask our dyed frame (pixels out of mask are black). / 염료 처리된 프레임에서 마스크를 적용합니다 (마스크 밖의 픽셀은 검은색).
    dyed_hair = cv.bitwise_or(frame, dyed_frame, mask=hair_mask) ################### 이거로 #########################
    dyed_hair =~dyed_hair

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)
        #====== 머리 추가 =========================================================
        addHair = mesh_coords[18][1] - mesh_coords[152][1]    # 18번매쉬.y좌표 - 0번매쉬.y좌표 / 18번을 [0, 14, 17, 18, 200, 199, 175] 중 1택 가능
        
        for i in [127, 162, 21, 54, 103,67, 109, 10, 338, 297, 332, 284, 251, 389, 356]:
            mesh_coords[i] = (mesh_coords[i][0], mesh_coords[i][1] + addHair)
        #==========================================================================

        mask =fillPolyTrans(mask, [mesh_coords[p] for p in FACE_OVAL], colors, opacity=1 )
        
        # # 머리
        mask = cv.bitwise_and(mask, dyed_hair)

        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_or(copy, mask_copy)
        # cv.imshow('f',mask_copy)
        return masked # , masked
    
def draw_FACE_CLOWN(img,colors): # 얼굴 광대
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_CLOWN], colors, opacity=1 )
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_CLOWN], colors, opacity=1 )
        # cv.imshow('FACE_CLOWN mask', mask) # 자른 형상 출력

        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_or(copy, mask_copy)
        cv.imshow('masked', masked)
        return masked #, masked    
    
def draw_FACE_CLOWN_MIDDLE(img,colors): # 얼굴 볼 중앙 (타원형 긴얼굴형 볼터치)
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)

        x = [50,101, 330,280]
        ################## 왼쪽 #####################
        middle_left = ( ([mesh_coords[p] for p in x][0][0]+[mesh_coords[p] for p in x][1][0])//2, 
                        ([mesh_coords[p] for p in x][0][1]+[mesh_coords[p] for p in x][1][1])//2  )

        # long_right와 short_right 하나씩 고르기 (elipse 좌우길이, 상하길이)
        long_left = [mesh_coords[p] for p in [126]][0][0]-middle_left[0]    # 좌우길이 1번
        
        short_left = [mesh_coords[p] for p in [205]][0][1]-middle_left[1]    # 상하길이 3번, 가장 보기좋음


        cv.ellipse(mask, middle_left,(long_left,short_left), 0, 0,360, colors, -1)
        ##############################################


        ################## 오른쪽 #####################
        middle_right = ( ([mesh_coords[p] for p in x][2][0]+[mesh_coords[p] for p in x][3][0])//2, 
                            ([mesh_coords[p] for p in x][2][1]+[mesh_coords[p] for p in x][3][1])//2  )

        # long_right와 short_right 하나씩 고르기 (elipse 좌우길이, 상하길이)
        long_right = middle_right[0]-[mesh_coords[p] for p in [355]][0][0]    # 좌우길이 1번

        short_right = [mesh_coords[p] for p in [425]][0][1]-middle_right[1]     # 상하길이 3번, 가장 보기좋음
        
        cv.ellipse(mask, middle_right,(long_right,short_right), 0, 0,360, colors, -1)
        ##############################################




        # [cv.circle(mask, mesh_coords[p], 1, utils.GREEN , -1, cv.LINE_AA) for p in x]
        # [cv.ellipse(mask, mesh_coords[p],(10,5), 0, 0,360, (0,255,255), -1) for p in x]
        

        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_and(copy, mask_copy)
        cv.imshow('FACE_CLOWN1 mask', masked) # 자른 형상 출력
        masked = cv.bitwise_or(copy, mask_copy)
        cv.imshow('dd',masked)
        return masked #, masked    
        
def draw_FACE_CLOWN_TOP(img,colors): # 얼굴 눈 아래 볼 (타원형 긴얼굴형 볼터치)
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)


        # mask = utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_CLOWN], utils.WHITE, opacity=1 )
        # mask = utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_CLOWN], utils.WHITE, opacity=1 )
        # # print(middle)
        # print([mesh_coords[p] for p in LEFT_CLOWN][0][0],[mesh_coords[p] for p in LEFT_CLOWN][0][1])
        # print([mesh_coords[p] for p in LEFT_CLOWN][1][0],[mesh_coords[p] for p in LEFT_CLOWN][1][1],'\n')


        x = [50,101, 330,280]
        ################## 왼쪽 #####################
        middle_left = ( ([mesh_coords[p] for p in x][0][0]+[mesh_coords[p] for p in x][1][0])//2, 
                        ([mesh_coords[p] for p in x][0][1]+[mesh_coords[p] for p in x][1][1])//2 -20 )

        # long_right와 short_right 하나씩 고르기 (elipse 좌우길이, 상하길이)
        long_left = [mesh_coords[p] for p in [126]][0][0]-middle_left[0]    # 좌우길이 1번
        # long_left = [mesh_coords[p] for p in [142]][0][0]-middle_left[0]      # 좌우길이 2번

        # short_left = middle_left[1] -[mesh_coords[p] for p in [126]][0][1]  # 상하길이 1번
        # short_left = [mesh_coords[p] for p in [187]][0][1]-middle_left[1]   # 상하길이 2번
        short_left = [mesh_coords[p] for p in [205]][0][1]-middle_left[1] -20    # 상하길이 3번, 가장 보기좋음


        cv.ellipse(mask, middle_left,(long_left,short_left), 0, 0,360, colors, -1)
        ##############################################


        ################## 오른쪽 #####################
        middle_right = ( ([mesh_coords[p] for p in x][2][0]+[mesh_coords[p] for p in x][3][0])//2, 
                            ([mesh_coords[p] for p in x][2][1]+[mesh_coords[p] for p in x][3][1])//2 -20 )

        # long_right와 short_right 하나씩 고르기 (elipse 좌우길이, 상하길이)
        long_right = middle_right[0]-[mesh_coords[p] for p in [355]][0][0]    # 좌우길이 1번
        # long_right = middle_right[0]-[mesh_coords[p] for p in [371]][0][0]     # 좌우길이 2번

        # short_right = middle_right[1] -[mesh_coords[p] for p in [347]][0][1]  # 상하길이 1번
        # short_right = [mesh_coords[p] for p in [411]][0][1]-middle_right[1]   # 상하길이 2번
        short_right = [mesh_coords[p] for p in [425]][0][1]-middle_right[1] -20    # 상하길이 3번, 가장 보기좋음
        
        cv.ellipse(mask, middle_right,(long_right,short_right), 0, 0,360, colors, -1)
        ##############################################




        # [cv.circle(mask, mesh_coords[p], 1, utils.GREEN , -1, cv.LINE_AA) for p in x]
        # [cv.ellipse(mask, mesh_coords[p],(10,5), 0, 0,360, (0,255,255), -1) for p in x]

        # cv.imshow('FACE_CLOWN1 mask', mask) # 자른 형상 출력

        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_or(copy, mask_copy)

        return masked #, masked    
    
def draw_FACE_CLOWN_APPLE(img,colors): # 얼굴 볼 중앙 (타원형 긴얼굴형 볼터치)
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)

        x = [50,101, 330,280]
        ################## 왼쪽 #####################
        middle_left = ( ([mesh_coords[p] for p in x][0][0]+[mesh_coords[p] for p in x][1][0])//2 , 
                        ([mesh_coords[p] for p in x][0][1]+[mesh_coords[p] for p in x][1][1])//2  )

        # long_right와 short_right 하나씩 고르기 (elipse 좌우길이, 상하길이)
        long_left = [mesh_coords[p] for p in [126]][0][0]-middle_left[0] -15   # 좌우길이 1번
        
        short_left = [mesh_coords[p] for p in [205]][0][1]-middle_left[1]     # 상하길이 3번, 가장 보기좋음


        cv.ellipse(mask, middle_left,(long_left,short_left), 0, 0,360, colors, -1)
        ##############################################


        ################## 오른쪽 #####################
        middle_right = ( ([mesh_coords[p] for p in x][2][0]+[mesh_coords[p] for p in x][3][0])//2 , 
                            ([mesh_coords[p] for p in x][2][1]+[mesh_coords[p] for p in x][3][1])//2  )

        # long_right와 short_right 하나씩 고르기 (elipse 좌우길이, 상하길이)
        long_right = middle_right[0]-[mesh_coords[p] for p in [355]][0][0] -15  # 좌우길이 1번

        short_right = [mesh_coords[p] for p in [425]][0][1]-middle_right[1]      # 상하길이 3번, 가장 보기좋음
        
        cv.ellipse(mask, middle_right,(long_right,short_right), 0, 0,360, colors, -1)
        ##############################################




        # [cv.circle(mask, mesh_coords[p], 1, utils.GREEN , -1, cv.LINE_AA) for p in x]
        # [cv.ellipse(mask, mesh_coords[p],(10,5), 0, 0,360, (0,255,255), -1) for p in x]

        # cv.imshow('FACE_CLOWN1 mask', mask) # 자른 형상 출력

        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        # masked = cv.bitwise_and(copy, mask_copy)
        # cv.imshow('FACE_CLOWN1 mask', masked) # 자른 형상 출력
        masked = cv.bitwise_or(copy, mask_copy)
        # cv.imshow('dd',masked)
        return masked #, masked    
    
def draw_EYEBROW(img,colors): # 눈섭
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)

        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYEBROW], colors, opacity=1 )
        x_LEFT = [mesh_coords[p][0] for p in LEFT_EYEBROW]
        w_LEFT = int(max(x_LEFT) - min(x_LEFT))
        y_LEFT = [mesh_coords[p][1] for p in LEFT_EYEBROW]
        h_LEFT = int(max(y_LEFT) - min(y_LEFT))

        half_w_LEFT = w_LEFT/2
        half_h_LEFT = h_LEFT/2
        x_mid_pos_left = max(x_LEFT) - w_LEFT
        y_mid_pos_left = max(y_LEFT) - h_LEFT
        pos_LEFT = (int(x_mid_pos_left), int(y_mid_pos_left))#(round(max(x_LEFT)-w_LEFT/2), round(max(y_LEFT)-h_LEFT/2))
        
        img_fg = cv.imread("eyebrow2(left).png",cv.IMREAD_UNCHANGED)
        img_fg = cv.resize(img_fg, (w_LEFT, h_LEFT))
        
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYEBROW], colors, opacity=1 )
        
        x_RIGHT = [mesh_coords[p][0] for p in RIGHT_EYEBROW]
        w_RIGHT = int(max(x_RIGHT) - min(x_RIGHT))
        y_RIGHT = [mesh_coords[p][1] for p in RIGHT_EYEBROW]
        h_RIGHT = int(max(y_RIGHT) - min(y_RIGHT))

        half_w_RIGHT = w_RIGHT/2
        half_h_RIGHT = h_RIGHT/2
        x_mid_pos_RIGHT= max(x_RIGHT) - w_RIGHT
        y_mid_pos_RIGHT = max(y_RIGHT) - h_RIGHT
        pos_RIGHT = (int(x_mid_pos_RIGHT), int(y_mid_pos_RIGHT))

        img_ = cv.imread("eyebrow2(right).png",cv.IMREAD_UNCHANGED)
        img_ = cv.resize(img_, (w_RIGHT, h_RIGHT))


        masked = cv.bitwise_and(copy, mask)
        out = cv.bitwise_or(frame,masked)
        # cv.imshow('EYEBROW copy', copy)
        overlay(out, pos_LEFT[0],pos_LEFT[1], int(w_LEFT),int(h_LEFT), img_fg)
        overlay(out, pos_RIGHT[0], pos_RIGHT[1], int(w_RIGHT),int(h_RIGHT), img_)
        # out = cv.bitwise_or(frame,masked)
        # cv.imshow('EYEBROW mask', mask) # 자른 형상 출력
        cv.imshow('EYEBROW masked', masked) # 마스킹 되는 형상 출력
        cv.imshow('EYEBROW out', out)
        
        
        return out # , masked

            
def draw_EYE(img,colors): # 눈
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYE], colors, opacity=1 )
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYE], colors, opacity=1 )
        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_or(copy, mask_copy)

        
        return masked # , masked


def Blur(img,colors):   # 얼굴 라인
    frame = img
    copy =frame.copy()              # 카피, 자른 화면 보이게 할 시 사용
    mask = np.zeros_like(frame)          # 캠 크기와 같은 검정색으로 이루어진 화면 (마스킹 사용시 사용됨)
        
    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coords = landmarksDetection(frame, results, False)
        mask =fillPolyTrans(mask, [mesh_coords[p] for p in FACE_OVAL], colors, opacity=1 )
        

        # 눈 뺄 시
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYE], (0,0,0), opacity=1 )
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYE], (0,0,0), opacity=1 )
        # # 눈썹 뺄 시
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in LEFT_EYEBROW], (0,0,0), opacity=1 )
        mask =utils.fillPolyTrans(mask, [mesh_coords[p] for p in RIGHT_EYEBROW], (0,0,0), opacity=1 )
        masked = cv.bitwise_and(copy, mask)

        mask_copy = mask.copy
        mask_copy =~mask
        # #  마스킹 하기
        masked = cv.bitwise_or(copy, mask_copy)
        cv.imshow('masked',masked)
        cv.imshow('mask',mask)
        return masked # , masked
def overlay(image, x, y, w, h, overlay_image): # 대상 이미지 (3채널), x, y 좌표, width, height, 덮어씌울 이미지 (4채널)
    alpha = overlay_image[:, :, 3] # BGRA
    mask_image = alpha / 255 # 0 ~ 255 -> 255 로 나누면 0 ~ 1 사이의 값 (1: 불투명, 0: 완전)
    
    for c in range(0, 3): # channel BGR
        image[y:y+h, x:x+w, c] = (overlay_image[:,:, c] * mask_image) + (image[y:y+h, x:x+w, c] * (1 - mask_image))

'''cam = cv.VideoCapture(0)
while True:
    _,frame = cam.read()
    frame = cv.resize(frame,(1000,800)) 
    
    draw_EYEBROW(frame,utils.WHITE)
    key = cv.waitKey(1)
    if key==ord('q') or key ==ord('Q'):
        break'''

# img = cv.imread("makeup_img\skin\\fall_skin.jpg")
# # draw_EYEBROW(img,utils.WHITE)
# # Blur(img, utils.WHITE)
# w = draw_FACE_OVAL(img, utils.WHITE)
# cv.imshow('w',w)
# cv.waitKey(0)


