## 기본설정 ##
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

def button_FBfan_ON():
    ref = db.reference('test/fan')
    fan_01 = ref.get()
    print("불러운 값: ",fan_01, " 타입: ", type(fan_01))

    ref = db.reference('test') #db 위치 지정, 기본 가장 상단을 가르킴
    ref.update({'fan' : "1"})            ############# 파베 보내기1
    print("저장되는 fan값:1  데이터 타입:", type(fan_01))

def button_FBfan_OFF():
    ref = db.reference('test/fan')
    fan_01 = ref.get()
    print("불러운 값: ",fan_01, " 타입: ", type(fan_01))
    
    ref = db.reference('test') #db 위치 지정, 기본 가장 상단을 가르킴
    ref.update({'fan' : "0"})            ############# 파베 보내기1
    print("저장되는 fan값:0  데이터 타입:", type(fan_01))