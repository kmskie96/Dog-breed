# dog-breed-project

trained_weights_final.h5
yolov3.weights

위 두 파일은 용량 문제(>100MB)로 올리지 못하였음.

Created: 2022년 3월 25일 오후 2:44
Last Edited Time: 2022년 3월 25일 오후 4:50
Participants: 익명, 익명, 익명, 익명, 익명

# 프로젝트 요약

## 제목 : 영상 객체탐지 기술을 기반한 견종 분류 및 실종견 탐색 서비스

### 선정배경

> 반려동물의 양육 가구수가 2018~2020 최근 3년 간 꾸준히 증가하고 있다.  유기,유실견수도 함께 증가 중이다. 유기견 관련 기존 웹서비스 탐색 후 기존 서비스가 유기견 분양에 집중되어 있고 실종신고 리스트가 여러 곳에 업로드 되어 있어 한 눈에 보기 불편하다는 것을 발견하였다. 실종견을 소유주에게 단기간에 인도하기 위한 방법으로  이 웹서비스를 기획하였다.
> 

![개요1.png](https://user-images.githubusercontent.com/102518623/160393259-5a97c959-bd80-408f-8de0-f880061f4a81.png)

![개요2.png](https://user-images.githubusercontent.com/102518623/160393268-c8535667-5ac8-41f4-8e75-246c8ab456c1.png)

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
    
    ![Untitled](https://user-images.githubusercontent.com/102518623/160393252-7d3e12a3-c529-4edd-b407-b1ce54abd0a0.png)
    

### 웹 서비스

- 사용 언어 : HTML5, CSS, Javascript
- 웹 프레임워크 : Bootstrap,Flask
- 웹페이지
    
    ![Untitled](https://user-images.githubusercontent.com/102518623/160393095-03f06091-1f7e-46a1-b0ae-2deb9a7655e5.png)
    
    ![Untitled](https://user-images.githubusercontent.com/102518623/160393239-d12c879a-eee3-493a-9fb3-1228632efdf8.png)
