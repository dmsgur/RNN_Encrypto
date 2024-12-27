:::::::::::::::::::::::::::::::::::::::::::::::discribe:::::::::::::::::::::::::::::::::::::::::::::::  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

시계열데이터 회귀분석예측을 위해 가상화폐 API를 이용하여 향후 가격을 예측한다.
지속적인 훈련 업그레이드를 위해 저장된 모델로 계속되는 훈련이 진행될수 있어 모델의 성능은 시간이 갈수록 높아 질수 있다.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
기본라이브러리 : numpy, matplotlib.pyplot, tensoflow, random, os
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
1. 크롤링 작업    
 Crawling/Craw_image_running :: main run    
 Crawling/Craw_image_google :: import   
 Crawling/Craw_image_naver :: import       
    이미지 저장경로, 검색어, 영문검색어(공백없어야함)    
    ::: 실습환경 :::     
    크롬 드라이버 필요
    (https://googlechromelabs.github.io/chrome-for-testing/)  
    win64 v131.0.6788.108  
    크롬 웹브라우저 필요 
      버전 131.0.6778.140(공식 빌드) (64비트)   
    
2. 데이터 증강 및 배경 제거   
 Preprocessing/preprocessing_running :: main run  
 Preprocessing/remove_background :: import  
 Preprocessing/util :: import   
    이미지 배경 제거 및 회전, 밝기조정, 확대 기법으로 이미지 
    데이터 증강   
    ::: 실습환경 :::  
    tf.image.random_brightness  
    tf.keras.layers.RandomRotation  
    tf.keras.layers.RandomFlip  
    tf.keras.layers.RandomZoom  
    rembg.remove   

3. 데이터 전처리 및 훈련 실행, 평가   
 Trainning/train_fit :: main run   
 Trainning/construct_Model :: import   
    이미지 전처리 및 훈련실행과 훈련결과 평가 
    훈련 실행시 최적값으로 조기종료 콜백 등록됨   
    :: 실습환경 :::   
    Sequential,Input,Dense,Conv2D,Dropout,MaxPool2D,  
    Flatten, confusion_matrix,classfication_report,heatmap 

4. 실제 이미지 처리 샘플 확인
 SampleData_Predict/test_class :: main run   
    인터넷등에서 가져온 이미지 파일을 모델이 측정   
    ::: 준비물 :::   
    단일 이미지   
  
* 순서대로 실행시 생성되는 파일 리스트  
* Trainning/classification_image.keras (저장모델)  
* classication_image.history (훈련결과 손실수치)  
* config (영문 라벨 리스트)  
* error 처리는 수행하지 않음.    
* 원본이미지 60*10 여개 , 증강이미지 60*5 여개    
copy right - 2024.12.18 광주컴퓨터기술학원 (dmsgur) 
