from pyowm import OWM #pip install pyowm  [Weather api] [https://github.com/csparpa/pyowm/]
import googlemaps, requests #pip install -U googlemaps      [Geocoding API,Geolocation API] [Google Cloud]
import json
import UnityEngine as ue


GOOGLE_API_KEY = 'AIzaSyD_qckRmOOWxcrSop9uWlKaqkec4ccwQOs'
url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}'
data = {'considerIp': True, }   # 현 IP로 데이터 추출

result = requests.post(url, data) # 해당 API에 요청을 보내며 데이터를 추출한다.

result2 = json.loads(result.text)

lat = result2["location"]["lat"] # 현재 위치의 위도 추출
lng = result2["location"]["lng"] # 현재 위치의 경도 추출 

while True:
    API_key = '1296f2181c5e3e22acb4fc333db31dd4'        #Openweathermap API 인증키 [https://openweathermap.org/]
    owm = OWM(API_key)
    mgr = owm.weather_manager()
    

    #위치
    #obs = mgr.weather_at_place('Seoul')              # Toponym
    #obs = mgr.weather_at_id(2643741)                      # city ID
    obs = mgr.weather_at_coords(lat,lng)      # lat/lon

    #날씨 정보 불러오기
    weather = obs.weather #날씨
    temp_dict_kelvin = weather.temperature('celsius')   # 섭씨 온도
    temp = temp_dict_kelvin['temp']
    
    now_weather = weather.status