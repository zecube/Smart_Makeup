
## 기본설정 ##
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


def button1(num):
    # #Firebase database 인증 및 앱 초기화
    # cred = credentials.Certificate('FireBase/ex2305-firebase-adminsdk-c7ei4-8835fc6f84.json') ## 바꿀수도 있는 줄, 파이어베이스.json키 위치
    # firebase_admin.initialize_app(cred,{
    #     'databaseURL' : 'https://ex2305-default-rtdb.firebaseio.com/' ##  ## 바꿀수도 있는 줄, 파이어베이스 리얼타임 데이터 베이스 url
    # })

    limit_num = 4
    ref = db.reference('test/contain_list')
    list1 = ref.get()
    print(len(list1))

    if isinstance(list1, list): # 리스트 형태인 경우
        limit_num = 4
    ref = db.reference('test/contain_list')
    list0 = ref.get()
    print("불러운 값: ",list0)
    print("\n")
    print("\n")
    
    if isinstance(list0, list) == 1 : # 리스트 형태인 경우
        print("리스트 형태임")
        if len(list0) < 4:
            print("리스트 길이",len(list0)," < 4")
            
            print("\n 사용될 리스트", list0)
            list1 = list0
            list1.append(num) # 리스트에 값 추가
            print("list1 = list0.append(num)", list1)
            
            ref = db.reference('test') #db 위치 지정, 기본 가장 상단을 가르킴
            ref.update({'contain_list' : list1})            ############# 파베 보내기1
            print("저장된 리스트",list, type(list))
            
            list_size = 3 # 리스트 사이즈 저장
            ref = db.reference('test') #db 위치 지정, 기본 가장 상단을 가르킴
            ref.update({'contain_list_size' : list_size})   ############# 파베 보내기2
            print("저장된 리스트 사이즈 : ",list_size)

        else:
            print("리스트형태, limit_num 초과, 정지")
            
    elif isinstance(list0, str) == 1 : # 문자형 형태인 경우
        print("문자형 형태임")
        print(list0, len(list0), type(list0))
        
        # 리스트 형식으로 저장이 안되고 str형태로 저장되었을 경우 ex "[1,2,3]"으로 저장됨
        list1 = list0[1:-1]
        print("[1:-1]이후 ", list1)
        list2 = list1.split(",")
        print("split 이후: ", list2,"   길이: ", len(list2))
        
        if len(list2) <= limit_num :
            list2.append(num)
            print("값 추가 이후 :",list2)
            ref = db.reference('test')
            ref.update({'contain_list' : list2})            ############# 파베 보내기1
            print("저장된 리스트값 :",list2)
            list_size = len(list2)
            ref = db.reference('test')
            ref.update({'contain_list_size' : list_size})   ############# 파베 보내기2
            print("저장된 리스트 사이즈 :",list_size)
        else:
            print("문자형태, limit_num 초과, 정지")

    else:
        print("리스트, 문자형 형태 둘다 아님")


    