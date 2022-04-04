from flask import Flask, request, render_template, redirect, Response
import numpy as np
from PIL import Image
from yolo import YOLO
import random
from os.path import join, basename, splitext, dirname
import sqlite3
from haversine import haversine, Unit

# import os
pnu = (35.2339681, 129.0806855)

names = ['BEAGLE', 'BULLDOG', 'CHIHUAHUEÑO', 'GOLDEN RETRIEVER',
         'KOREA JINDO DOG', 'MALTESE', 'POMERANIAN', 'POODLE',
         'SHIH TZU', 'WELSH CORGI', ]

kor_names = ['비글', '불독', '치와와', '골든리트리버',
             '진돗개', '말티즈', '포메라니안', '푸들',
             '시추', '웰시코기']

eng_kor = dict(zip(names, kor_names))

anchors = np.array(
    [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
     (59, 119), (116, 90), (156, 198), (373, 326)],
).astype('float32')

# path_to_project = r'C:\project\Machine Learning Project\yolov3\tf2-keras-yolo3-master'
model_path = r"C:\project\Machine Learning Project\yolov3\tf2-keras-yolo3-master\trained_weights_final.h5"

yolo = YOLO(model_path=model_path,
            class_names=names,
            anchors=anchors,
            score=0.6,   # 해당 점수 이상만 출력
            )

app = Flask(__name__)


@app.route('/main')
def index():
    return render_template('index.html', fname=f'dog.jpg', resultTxt=[], report=[])


@app.route('/')
def video():
    return render_template('video.html')


@app.route('/pic', methods=['POST'])
def pic():

    try:
        if request.files['selimg'].filename != '':
            f = request.files['selimg']
        else:
            f = request.files['capimg']
        img = f'./static/image/{f.filename}'
        f.save(img)
    except:
        img = './static/image/dog.jpg'
    finally:
        result_image, box_infos = yolo.detect_image(
            Image.open(img), box_info=True)
        result_name = f'{dirname(img)}/{splitext(basename(img))[0]}_result.jpg'
        result_image = result_image.convert('RGB')  # RGBA 오류
        result_image.save(result_name)

        resultData_dict = {}
        for breed, _, score in box_infos:
            with sqlite3.connect('doginfo.db') as db:
                cur = db.cursor()
                sql = 'select dname, dinfo from doginfo where dname=?'
                result = cur.execute(sql, (eng_kor[breed], ))

            breed_kor, dog_info = result.fetchall()[0]

            resultData_dict.setdefault(breed, {
                'breed_kor': breed_kor,
                'score': [],
                'dog_info': dog_info,
            })

            resultData_dict[breed]['score'].append(f'{score*100:.2f}')

        resultData = [
            [
                f"{resultData_dict[breed]['breed_kor']}({breed})",
                '%, '.join(
                    list(sorted(resultData_dict[breed]['score'], reverse=True))),
                resultData_dict[breed]['dog_info']
            ]
            for breed in resultData_dict
        ]

        # 높은 점수순으로 정렬
        resultData = sorted(resultData, key=lambda data: data[1], reverse=True)

        reportData_1 = []
        report_dist_list_1 = []
        reportData_2 = []
        report_dist_list_2 = []
        if resultData == []:
            resultData.append([f'발견된 객체가 없습니다.', f'00.00', ''])
        else:
            with sqlite3.connect('test.db') as db:
                cur = db.cursor()
                sql = 'SELECT * from test_table where 품종=? AND 상태=?'

                breed = next(iter(resultData_dict))

                # 구조, 목격 견
                result1 = cur.execute(sql, (breed, '구조'))
                for r in result1.fetchall():
                    reportData_1.append(r)

                result2 = cur.execute(sql, (breed, '목격'))
                for r in result2.fetchall():
                    reportData_1.append(r)

                for row in reportData_1:
                    coor = (row[-2], row[-1])
                    try:
                        dist = haversine(pnu, coor, unit='km')
                        dist = round(dist, 1)
                        report_dist_list_1.append(
                            (dist, row[1], row[2], row[3], row[4], row[5], row[6]))
                    except:
                        pass
                    report_dist_list_1.sort(
                        key=lambda x: x[0], reverse=False)

                # 실종견
                result = cur.execute(sql, (breed, '실종'))
                for r in result.fetchall():
                    reportData_2.append(r)
                for row in reportData_2:
                    coor = (row[-2], row[-1])
                    try:
                        dist = haversine(pnu, coor, unit='km')
                        dist = round(dist, 1)
                        report_dist_list_2.append(
                            (dist, row[1], row[2], row[3], row[4], row[5], row[6]))
                    except:
                        pass
                    report_dist_list_2.sort(
                        key=lambda x: x[0], reverse=False)

        return render_template('index.html', fname=f'{basename(result_name)}',
                               resultTxt=resultData, report1=report_dist_list_1, report2=report_dist_list_2)


if __name__ == '__main__':
    app.run(
        host='192.168.0.2',
        port=5000,
        debug=True,             # 수정이 되면 자동으로 서버를 restart.
    )
