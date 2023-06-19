# 연구노트 

본 연구노트는 release 후 제거될 예정입니다. 
김형민 학생의 연구노트를 참고하였습니다.

## 연구 목표

Co-speech motion generation with diffusion model

## 계획 및 진행 상황 

1주차 (03.20 ~ 03.24): 

- [X] Motion 생성 모델 관련 자료조사

- [x] AI-Hub 데이터셋 다운로드 및 audio, skeleton, text 파일 추출 

- [x] TED 데이터셋[1]에 대해 Trimodal[2] 코드 돌려보고 구조 확인/검토

2주차 (03.27 ~ 03.31) : 

- [X] MLD: Motion latent diffusion[3] 논문 리뷰

- [X] MDM: Motion Diffusion Model[4] 논문 리뷰

- [X] MLD 코드 돌려보고 구조 확인 및 검토

3주차 (04.03 ~ 04.07) : 

- [X] DiffGesture[5] 코드 돌려보고 구조 확인 및 검토

4주차 (04.10 ~ 04.14) : 

- [X] DiffGesture[5] 코드에 한국어 음성으로 돌려보고 결과 검토

5주차 (05.01 ~ 05.05) : 

- [X] AI-Hub 데이터 로더 구현

6주차 (05.08 ~ 05.12) : 

- [X] AI-Hub 데이터셋을 DiffGesture와 trimodal[2]로 돌려보고 성능 비교

7주차 (05.15 ~ 05.19) : 

- [X] AI-Hub 데이터셋 forced alignment 적용 (형민)

8주차 이후 (05.22 ~ ) : 

- [x] AI-Hub 데이터셋 forced alignment 학습 결과 검토
- [x] Full body용 오토인코더 학습
- [X] Text encoding feature 추가


## 연구 설계 

### 1. baseline 확보 : AI-Hub 

AI-Hub의 베이스라인을 돌려 보고 데이터 전처리도 수행한다. (~ 8주차)


#### Reviews 

- [X] [Generative models for human’s motion](/note/MotionGeneration.pdf)

#### References

1. [Youtube Gesture Dataset](https://github.com/youngwoo-yoon/youtube-gesture-dataset)

2. [Gesture Generation from trimodal context](https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context)

3. [Motion Latent Diffusion](https://github.com/ChenFengYe/motion-latent-diffusion)

4. [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)

5. [Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation](https://github.com/Advocate99/DiffGesture)

