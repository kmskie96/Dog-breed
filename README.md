# dog-breed-project

서비스에 사용된 최종모델
[trained_weights_final.h5](https://drive.google.com/file/d/1LkjOsvRhTnHFQR4sx8UKieyX0D6D_tYl/view?usp=sharing)

사전학습된 yolov3모델
[yolov3.weights](https://drive.google.com/file/d/18VRY49A2zHZzb__vCuO0R5Rn1-I5H3Jq/view?usp=sharing)



Created: 2022년 3월 25일 오후 2:44
Last Edited Time: 2022년 3월 25일 오후 4:50
Participants: 익명, 익명, 익명, 익명, 익명

# 프로젝트 요약

## 제목 : 영상 객체탐지 기술을 기반한 견종 분류 및 실종견 탐색 서비스

### 선정배경

> 반려동물의 양육 가구수가 2018~2020 최근 3년 간 꾸준히 증가하고 있다.  유기,유실견수도 함께 증가 중이다. 유기견 관련 기존 웹서비스 탐색 후 기존 서비스가 유기견 분양에 집중되어 있고 실종신고 리스트가 여러 곳에 업로드 되어 있어 한 눈에 보기 불편하다는 것을 발견하였다. 실종견을 소유주에게 단기간에 인도하기 위한 방법으로  이 웹서비스를 기획하였다.
> 

![개요1](https://user-images.githubusercontent.com/95560893/161726881-479a8405-de0a-4e4e-91c9-27d20218091c.png)

![개요2](https://user-images.githubusercontent.com/95560893/161727012-302138ce-9bb6-4d70-9d68-f37d54e1af31.png)

### 데이터 수집 및 전처리

- 데이터 종류 :  10종 이미지, 견종 정보, 실종신고데이터

<aside>
💡 10종 : 비글, 웰시코기, 불독, 포메라니안, 진돗개, 시추, 말티즈, 치와와, 푸들, 골든 리트리버

</aside>

- 데이터 수집 및 저장 : Requests, Beautifulsoup, sqlite3
- 데이터 전처리 : Labelimg, tensorflow hub object detection API, Pandas, Numpy, Imgaug

### 모델 학습

- 모델 :  YOLOv3
- 사전훈련에 사용된 Dataset : COCO dataset
- 학습방법 : 전이학습, 배치학습
- 학습결과
    - 데이터 수 : 4580장
    - learning rate: 0.00001
    - epoch : 76
    - 한 epoch당 860초
    - loss : 12.6
    - Accuracy : 77.8%
    
    ![Untitled](https://user-images.githubusercontent.com/95560893/161727124-7fe1f4fc-bd8f-47f6-a84d-e5a5ed8f2915.png)
    

### 웹 서비스

- 사용 언어 : HTML5, CSS, Javascript
- 웹 프레임워크 : Bootstrap,Flask
- 웹페이지
    
    ![Untitled 1](https://user-images.githubusercontent.com/95560893/161727185-216611eb-1db8-4dc8-b2d1-6a2f15c22dc8.png)
    
    ![Untitled 2](https://user-images.githubusercontent.com/95560893/161727207-2ca8ca79-6508-49f4-a680-a3d1b5f8e232.png)
