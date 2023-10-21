# -*- coding: utf-8 -*-

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import os
import requests
import time
from bing_image_downloader import downloader

# 음식 항목 리스트
food_items = [
    "갈비탕", "갈치구이", "고등어구이", "더덕구이", "장어구이", "조개구이", "조기구이", "황태구이", "훈제오리", "계란국",
    "떡국", "만두국", "무국", "미역국", "북엇국", "시래기국", "육개장", "콩나물국", "콩자반", "갓김치", "깍두기", "무생채",
    "배추김치", "백김치", "부추김치", "열무김치", "오이소박이", "총각김치", "파김치", "가지볶음", "고사리나물", "미역줄기볶음",
    "숙주나물", "시금치나물", "애호박볶음", "수제비", "열무국수", "잔치국수", "꽈리고추무침", "도라지무침", "도토리묵", "잡채",
    "콩나물무침", "김치볶음밥", "비빔밥", "새우볶음밥", "알밥", "감자채볶음", "건새우볶음", "고추장진미채볶음", "두부김치", "멸치볶음", "어묵볶음",
    "오징어채볶음", "주꾸미볶음", "깻잎장아찌", "감자전", "김치전", "동그랑땡", "생선전", "파전", "호박전", "갈치조림", "감자조림",
    "고등어조림", "꽁치조림", "두부조림", "땅콩조림", "연근조림", "우엉조림", "코다리조림", "전복죽", "호박죽", "닭계장",
    "동태찌개", "순두부찌개", "계란찜", "김치찜", "해물찜", "갈비탕", "감자탕", "매운탕", "삼계탕",
    "추어탕", "돈까스", "피자", "스파게티", "햄버거", "크림 파스타", "치즈 스틱", "로제 파스타", "치즈버거", "브런치", "베이컨 스크램블 에그",
    "미트볼 스파게티", "치즈 케이크", "샌드위치", "수프", "닭가슴살 샐러드", "스크램블 에그", "치킨버거", "짜장면", "짬뽕", "깐풍기",
    "탕수육", "유산슬", "막국수", "고로케", "만두", "오뎅탕", "계란밥", "꼬막비빔밥", "초밥", "라멘", "돈부리",
    "우동", "나베", "유부우동", "막걸리", "코다리조림", "소주", "맥주", "수육", "족발", "물회", "송편",
    "탕후루", "모듬 회", "송이버섯볶음", "주먹밥", "계란탕", "고추잡채", "기스면", "깐쇼새우", "누룽지탕", "라조기",
    "마파두부", "마라탕", "멘보샤", "부추잡채", "오향장육", "울면", "유린기", "유산슬", "자춘걸", "크림새우",
    "해파리냉채", "팔보채", "부리또", "오므라이스", "감바스", "필라프", "라자냐", "포케", "함박스테이크", "미트볼",
    "뇨끼", "리조또", "샐러드", "타코", "베이글", "라볶이", "떡꼬치", "핫도그", "토스트", "순대", "덴뿌라",
    "카레", "볶음 우동", "카츠산도", "사케동", "오코노미야끼", "규동", "냉모밀", "후토마끼", "텐동", "오야꼬동",
    "사케", "커피", "케밥", "도넛", "나초", "버팔로 윙", "어니언 링", "치즈", "애플 파이", "아이스크림", "랍스터",
    "칠면조", "오렌지 주스", "스무디", "티라미수", "브라우니", "케이크", "딤섬", "우육면", "타코야키", "닭가슴살",
    "새우튀김", "육전", "부추전", "샤브샤브", "낫토", "양갱", "소금빵", "앙버터", "짜장밥", "짬뽕밥", "잡채밥",
    "고추잡채밥", "마파두부밥", "호빵", "호떡", "붕어빵", "호두과자", "물만두", "고구마맛탕", "양꼬치", "샥스핀", "분짜",
    "쌀국수", "반미", "똠양꿍", "솜땀", "똠카까이", "깽댕", "팟타이", "나시고랭", "딹꼬치", "초콜릿", "숙주볶음",
    "피시 앤 칩스", "된장찌개", "부대찌개", "라면", "돼지국밥", "마카롱", "바나나", "사과", "오렌지", "딸기",
    "포도", "복숭아", "블루베리", "키위", "파인애플", "레몬", "수박", "자두", "아보카도", "밤", "감",
    "배", "토마토", "멜론", "귤", "참외", "곶감", "식혜", "카스테라", "푸딩", "머핀", "모찌",
    "크레페", "팬케이크", "민트 초콜릿", "젤라또", "와플", "말린 과일", "브라우니", "쿠키", "파르페", "시리얼", "크로와상",
    "갈비찜", "김밥", "유부초밥", "백설기", "장어덮밥", "체리", "건포도", "코코넛", "석류", "은행꼬치", "화채",
    "바지락술찜", "하몽 멜론", "와인", "마른 오징어", "우렁쌈밥", "치즈볼", "대창전골", "곱창전골", "대창구이", "곱창구이",
    "야채곱창", "양고기", "월남쌈", "소고기 타다끼", "쭈꾸미볶음", "순대전골", "곰탕", "설렁탕"
]

# 이미지를 저장할 기본 디렉토리 생성
base_directory = 'C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb1/image/'
os.makedirs(base_directory, exist_ok=True)

# Chrome 웹 드라이버 설정
chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=chrome_options)

for food_item in food_items:
    # Google 이미지 검색 페이지로 이동
    URL = 'https://www.google.co.kr/imghp'
    driver.get(URL)
    driver.implicitly_wait(10)

    # 검색어 입력
    keyElement = driver.find_element(By.XPATH, '//*[@id="APjFqb"]')  # 각자 XPATH 수정 필요
    keyElement.clear()
    keyElement.send_keys(food_item)
    keyElement.send_keys(Keys.RETURN)

    # 스크롤 다운하여 이미지 로딩
    bodyElement = driver.find_element(By.TAG_NAME, 'body')
    time.sleep(5)
    for i in range(10):
        bodyElement.send_keys(Keys.PAGE_DOWN)
        time.sleep(2)

    # 이미지 URL 수집
    image_elements = driver.find_elements(By.CSS_SELECTOR, 'img.rg_i')
    image_urls = []
    for image_element in image_elements:
        data_src = image_element.get_attribute('data-src')
        if data_src and data_src.startswith('http'):
            image_urls.append(data_src)

    # 음식별 디렉토리 생성
    food_directory = os.path.join(base_directory, food_item)
    os.makedirs(food_directory, exist_ok=True)

    # 이미지 다운로드 및 저장
    for seq, url in enumerate(image_urls):
        response = requests.get(url)
        if response.status_code == 200:
            filename = os.path.join(food_directory, f'{food_item}_{seq}.jpg')
            with open(filename, 'wb') as file:
                file.write(response.content)

# 웹 드라이버 종료
driver.quit()

# 각 음식 항목에 대해 이미지 다운로드
for food_item in food_items:
    query = food_item
    downloader.download(query, limit=10, output_dir='C:/Users/KITCOOP/kicpython/hansik/kfood_new/learning_rgb1/image')
