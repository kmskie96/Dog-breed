from yolo import YOLO
from yolo import detect_video

names = ['BEAGLE', 'BULLDOG','CHIHUAHUEÑO', 'GOLDEN RETRIEVER',
         'KOREA JINDO DOG', 'MALTESE', 'POMERANIAN', 'POODLE',
         'SHIH TZU', 'WELSH CORGI', ]

yolo = YOLO(model_path='./trained_weights_final.h5', 
            class_names=names)
yolo.score = 0.3

fname = 'KakaoTalk_20220322_231035199.mp4'
detect_video(yolo,
             f'./video/{fname}',
             f'./video/detect{fname}')
# detect_video(yolo, 0, './video/webcam.mp4')