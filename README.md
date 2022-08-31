# 2020 AI Grand Challenge 이상행동 감지 Track
   
<p align="center"><img src="https://user-images.githubusercontent.com/63839581/118762705-9e549e80-b8b1-11eb-9295-cc2e3cc9f969.jpg" width="1050"></p>
   
## Description

> #### 이 프로젝트는 [Yolov5](https://github.com/ultralytics/yolov5)를 기반으로 작성되었음을 밝힙니다.


이 저장소는 과학기술정보통신부에서 주최한 2020 AI Grand Challenge 공모전 이상행동 감지 Track 문제를 해결하는 것을 목표로 합니다.   
   
이상행동 감지 Track의 주제는 실시간 cctv 영상내에서 발생하는 실신 행위에 대한 검출 및 추적입니다. 그리고 대회 규정은 대표적으로 다음과 같습니다.   
   
(1) 제출하는 모델의 용량은 1GB 이내로 한다.      
(2) 20초 길이를 가진 15fps의 동영상 500clip에 대해서 6시간 이내에 추론 완료한다.    
(3) 10초 이상 실신한 사람에 대해서 검출하여 json 파일 포맷으로 좌표 저장한다.   
(4) 10초 이내에 실신한 사람이 일어나는 경우 실신으로 판단하지 않는다.   
   
주어진 문제를 해결하기 위해 문제의 정의를 크게 **실신한 사람에 대한 검출**(Detection)과 **검출된 사람에 대한 추적**(Tracking)으로 나누어서 진행했습니다.   

## Requirement
   
Python 3.6 or later with all [requirements.txt](https://github.com/sckim0430/2020-AI-Grand-Challenge-Abnormal-behavior-detection-Track/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:   
```bash
$ pip install -r requirements.txt
```

## Pretrained Weights
   
[YOLOv5l](https://github.com/sckim0430/2020-AI-Grand-Challenge-Abnormal-behavior-detection-Track/releases)  
   
## Custom Dataset
   
./data/에서 Dataset 환경 설정하는 [YAML](https://github.com/sckim0430/2020-AI-Grand-Challenge-Abnormal-behavior-detection-Track/blob/master/data/lying_person.yaml)을 생성합니다.   
      
./data/에서 각각 **images**와 **lables** 하위 폴더를 생성하고, 각 하위 폴더 내에 **train**, **validation**폴더를 생성하여 데이터를 저장합니다.
   
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118771505-97cc2400-b8bd-11eb-8a0f-54302ddf269c.jpg" width="600"></p>
   
   
labels 폴더 내의 파일명은 images 폴더내의 파일명과 동일하게 명시하며, 레이블링 형식은 yolov3를 따릅니다.   
Labeling Tool은 [labelImg](https://github.com/tzutalin/labelImg)를 사용하였습니다.   
   
## Train
   
[다음](https://github.com/sckim0430/2020-AI-Grand-Challenge-Abnormal-behavior-detection-Track/blob/master/train_cmd.txt) 과 같이 train.py를 실행합니다.   
학습된 weight 파일은 ./runs 하위 폴더에 생성됩니다.

## Tracking
   
대회 주제는 실신한 사람을 대상으로 검출 및 추적하는 것이므로 실신한 사람의 특성이 움직임이 거의 없다는 점을 이용하여 Iou Matching을 통해 Tracking을 수행했습니다.
   
알고리즘은 다음과 같습니다.   
   
- 매 frame마다 검사를 진행하고, model이 쓰러짐을 검출했을 때, 해당 시점으로부터 10초 뒤인 145~150 frame 후의 영상들에 대해서도 동일한 객체가 존재하는지 Iou를 비교하여 존재 할 경우 box 좌표와 conf score와 stack을 가지는 Dict들의 List인 Detection Manager에 저장한다.   
   
- 앞서 저장한 Detection Manager를 통해 stack 조건을 통해 Detection Manager의 해당 요소를 제거 여부를 결정하고, box 좌표를 통해 다음 frame에서 동일 객체를 detection한 경우 새로운 box 좌표를 부여하고, 그렇지 않은 경우는 현재 box 좌표를 유지하고 stack을 쌓는다.   
   
추론 시간 제한 사항을 해결 하기 위해 **batch prediction**을 사용하여 모델 추론 시간을 최대한 감축시켰으며, **NMS Ensemble**을 통해 성능을 향상 시켰다. 대회 테스트 셋 기준으로 성능이 약 **5%** 향상되었습니다.
또한, 일정 frame마다 검사하여 동일 객체가 존재하지 않을 경우 해당 객체를 제거하는 방식은 모델 성능 및 외부요인(가려짐, 잘림 등)에 크게 좌지우지되므로 145~150 frame을 미리 검사하여 Detection Manager를 통해 관리하는 방식을 사용하여 추론 시간을 단축시켰습니다.   
   
전체 검출 및 추적 코드인 [predict.py](https://github.com/sckim0430/2020-AI-Grand-Challenge-Abnormal-behavior-detection-Track/blob/master/predict.py)는 다음과 같이 실행합니다.    
    
 ```bash
$ python predict.py [동영상 클립 폴더 경로]
```
   
## Result
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118776957-8423bc00-b8c3-11eb-8c0a-6bf4cee6cf81.png" width="500"></p>   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118777025-97368c00-b8c3-11eb-8d59-554c8bd7a4ee.png" width="500"></p>   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118777095-a6b5d500-b8c3-11eb-85d8-a95cd199e915.png" width="500"></p>   
   
   
대회에 사용된 총 학습 영상은 약 2500장입니다.   
250장의 Test set에 대해서 검출(Detection) 성능평가를 한 결과, mAP@Iou0.5에 대해선 99.8%, mAP@Iou0.5:0.95에 대해선 80.2%를 얻었습니다.    
Tracking 성능을 포함한 대회 최종 성적은 mAP@Iou0.75에 대해서 60.2%를 기록하여 5위를 기록했습니다.   
   
## Contact
   
another0430@naver.com
