//데이터 17KPamLcwRpk7yZVpjMqqaumfIBL-8R8k
data_sets = []

//메뉴 생성기 종료 E==============================
//데이터 아키텍처{sub_title:"",sub_content:"",sub_img:[],user_fill:""}
data_sets.push({sub_title:"가상화폐 사이트 API 파악",
  sub_content:"훈련에 필요한 정보 추출 주소와 요청 및 응답값 확인",
  sub_img:["1S0rv5tVKoKnAqKRLrW7huVwCqFct3AnY","1pTmtTsFhCBLjsOIIDIBtx2MonQiwPc6h",
    "1xjyNyS2sxtbov76bDhKaPzvucu-pGf58","1XXsGddarlNGJVYKwNl7c52eB_aup__9T",
    "16RHb3Jd-u_n_0gitYRJ5kMfrPK2ftFQf"
  ],
  user_fill:"opencv, selenium 라이브러리와 크롬 드라이버를 활용하여 데이터 수집 자동화 - 관련없는 데이터는 임의 삭제 ",
  asso_file:"관련파일 /Crawling/Craw_image_running 실행"
})
data_sets.push({sub_title:"이미지 증대 및 배경 제거",
  sub_content:"수집된 이미지의 밝기,회전등을 조정하여 이미지 수량을 증대시키고 학습의 효율을 높이기 위해 이미지의 배경을 제거",
  sub_img:["1WIn9gRPAWIOLxXbwba07WlKiWy3eveQq"],
  user_fill:"배경제거를 위해 rembg 라이브러리와 증대를 위해 텐서플로우의 밝기,회전 등의 조정 라이브러리를 사용",
  asso_file:"관련파일 /Preprocessing/preprocessing_running.readImageDirect 호출 실행"
})
data_sets.push({sub_title:"이미지 전처리",
  sub_content:"증대된 이미지의 사이즈 및 수치 표준화 및 정답파일의 원핫인코딩후 훈련데이터 셋 준비",
  sub_img:["1wIWJDsTxrIMYLpt30dbApsC8fDDijv5w","1FPB9F4Jo_F_ln8pqq332Vm7Uq7OdSTHm"],
  user_fill:`이미지의 사이즈를 64*64로 조정하였으며, 색상의 최대값 255를 이용하여 0 ~ 1 사이의 수치로 minmax scale을 실행하고
   정수로된 정답데이터를 원핫인코딩으로 이진데이터로 변경 수행`,
  asso_file:"관련파일 /Preprocessing/preprocessing_running Trainning/construct_model 일부 호출 실행"
})
data_sets.push({sub_title:"데이터 확인 및 모델 구성",
  sub_content:"전처리가 마무리되고 정답과 이미지의 순서별 일치를 확인후 모델 구성 훈련실행",
  sub_img:["1w_VpWt44pW84hxctJo4cHyxzmp0VpzQm","1bx2inTLlp0sEEB3zWp4nL2OEvLm0tUj6"],
  user_fill:`컨볼루션 레이어와 맥스풀 및 드랍아웃 레이어를 이용하여 특성 추출과 과적합 방지 및 일반화를 위해 드랍아웃 레이어 적용`,
  asso_file:"관련파일 /Trainning/construct_model 일부 호출 실행"
})
data_sets.push({sub_title:"훈련 시작과 종료",
  sub_content:"구성된 모델과 전처리된 데이터를 이용하여 모델의 훈련",
  sub_img:["1bx2inTLlp0sEEB3zWp4nL2OEvLm0tUj6","17KPamLcwRpk7yZVpjMqqaumfIBL-8R8k",
  "1Nr5irPNLdxPsr25h6NNLTKxZZxj1p7lr","1z5Y5JmcnqvGEubKRDR1b0bxfubKbc5Uh"],
  user_fill:`최적의 검증데이터 정확도를 찾기 위해 조기종료 콜백함수와 adam의 최적화 함수, categorical_crossentroy 손실함수를 이용하여 모델 컴파일 및 훈련 실행`,
  asso_file:"관련파일 /Trainning/construct_model 호출 실행"
})
data_sets.push({sub_title:"훈련된 모델의 평가 및 예측",
  sub_content:"훈련이 완료된 모델 결과의 시각화",
  sub_img:["1sU0p2CqZs1e_HLtbyWZT_SaQoApr4gcp",
    "1uR_MyZpX_Ij2f5rKtXWEK8HhdLShJnKD",
  "1OKuvpXVDtSefucYk6qPSZnMYSk6ze9-2",],
  user_fill:`이미지 다양성이 부족하여 과대적합 증상이 점점 나타나고 있어 다양하고 더 많은 이미지가 필요함`,
  asso_file:"관련파일 /Trainning/construct_model 호출 실행"
})
data_sets.push({sub_title:"모델의 혼동행렬 시각화 및 f1 score 요약",
  sub_content:"모델의 종합 성능평가와 개선을 위해 혼동행렬 표기",
  sub_img:["17OG2tT84AOlubrN4ss_CpXzwjrFNpNLE",
    "19_JfNcm_lB-EPiJrokWKlzGiDG_edu9Q",
  "1K3kEmnDJ08IBoutPanDidwf7HZ0BPkdw"],
  user_fill:`이미지 다양성이 부족하여 과대적합 증상이 점점 나타나고 있어 다양하고 더 많은 이미지가 필요하며 혼동행렬을 참고한바, 곰, 개구리, 코끼리, 타조의 이미지 추가가 필요 `,
  asso_file:"관련파일 /Trainning/construct_model 호출 실행"
})
data_sets.push({sub_title:"실제 데이터 예측 테스트",
  sub_content:"새로운 단일 이미지를 구비하여 자동 전처리와 모델에 예측 구현",
  sub_img:["1xYuUqpVOCnWGxf85SjJg93zx7AjPRWBO",
    "1k9hYVa9-Xb86TR4SVMEA4Z_C8FTuYG0v",
  "1OEsiyCcCmksME8-NSjfhsdQF5YvPoPnJ",
  "1L5RkCTiL71LTU7wzyGvZUQ46EXgd53hM",
  "1_7lzeuIG8DMsSjzIbITiv0JwC3LKHQIl",
  "1gwyu3cCUm-qyqjKfyXaJePDstV39dqZY",
  "1pQrKhvZvPh0gkoPjGyk4o1yT6XKIm75t",
  "1vykFxj9A6Q4WeEGQ0lSJl3sL3gsIJGIL",
  "1k-Qq1Qd3oi3vO_keDzsqi3qvAizBWGFY"],
  user_fill:`이미지 다양성의 부족으로 성능은 다소 떨어지지만 대부분의 이미를 예측함`,
  asso_file:"관련파일 /SampleData_Predict/test_class 호출 실행"
})