# Capstone Design Project - Hanyang-Chatbot
## 인공지능을 이용한 챗봇 솔루션 개발

## API 명세서
URL 추가 예정

## 구조
![ver1 1](https://user-images.githubusercontent.com/12870549/58152750-46760380-7ca8-11e9-966d-228c522dc6e8.png)

## 전처리기
### 정제
1. 공백문자 정리
2. 대문자 -> 소문자
3. 유니코드 변환 이후 [유니코드 카테고리](https://www.fileformat.info/info/unicode/category/index.htm) 중 Mn, C*, P*제거
4. 초성 변환 예) ㅅㅌ -> 셔틀
5. [불용어](https://www.ranks.nl/stopwords/korean) 제거

![image](https://user-images.githubusercontent.com/12870549/58176137-9d95cb80-7cdc-11e9-8cb1-05d9247b8f7e.png)
### 형태소 분석

[Khaiii](https://github.com/kakao/khaiii) 와 [mecab-ko](https://bitbucket.org/eunjeon/mecab-ko-dic)를 함께 사용하였다. [이유](https://www.notion.so/mhlee/Khaiii-vs-Mecab-444874c394d3461eb6d8ece8b6615a5d) / [품사 정보](https://docs.google.com/spreadsheets/d/1-9blXKjtjeKZqsf4NzHeYJCrr49-nXeRF6D80udfcwY/edit?usp=sharing)

![image](https://user-images.githubusercontent.com/12870549/58176400-244aa880-7cdd-11e9-8bd1-7bc8641008bb.png)
### 키워드 분리
![image](https://user-images.githubusercontent.com/12870549/58177973-4b56a980-7ce0-11e9-9dd8-45adbb74f0f5.png)
### 토큰화
![image](https://user-images.githubusercontent.com/12870549/58178014-5f9aa680-7ce0-11e9-9742-954f31db95bf.png)
![image](https://user-images.githubusercontent.com/12870549/58180415-084b0500-7ce5-11e9-8e95-741ff1fde7e7.png)


### 유사도 분석

## Deep Learning Models
### [BERT](https://arxiv.org/pdf/1810.04805.pdf) - [transformer](https://arxiv.org/abs/1706.03762)모델의 Encoder 부분을 사용한 사전 훈련 모델


### Bert Pretraining
1. 구글의 사전훈련 모델을 사용하지 않은 이유
    
    모델 사용 중 전처리 과정에서 NFC 정규화 문제 발견하였다. - [관련된 issue](https://github.com/google-research/bert/issues/138)

    문제를 해결하는 코드를 [pull request](https://github.com/google-research/bert/pull/512) 하였고, 한국어의 사전훈련이 제대로 되지 않았을 것으로 판단 하였다. 
    
    *(현재는 NFC 정규화하지 않은 버전의 모델이 올라와 있음)*
2. 학습 데이터
   
   [Wikipedia](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C) 말뭉치를 사용 하였다.[wikiextractor](https://github.com/attardi/wikiextractor)를 사용해 본문만을 추출하고 추가적인 데이터 정제와 모델 훈련데이터 생성을 위한 형식으로 변경하였다.
3. 형태소 사전
   
   기존 단어 사전은 다국어 지원으로 사전의 크기가 크며 사용된 [wordpiece](https://arxiv.org/pdf/1609.08144.pdf)도 충분히 좋지만 형태소를 분석한 사전을 사용 해보고자 하였다.

   1. Wiki에서 많이 사용되는 상위 형태소 7449개를 추출하였다.
   2. Wiki 데이터에 대한 지나친 편향을 줄이기 위해 [국립국어원의 자주사용되는 한국어 어휘 데이터](https://ko.wiktionary.org/wiki/%EB%B6%80%EB%A1%9D:%EC%9E%90%EC%A3%BC_%EC%93%B0%EC%9D%B4%EB%8A%94_%ED%95%9C%EA%B5%AD%EC%96%B4_%EB%82%B1%EB%A7%90_5800)의 형태소 1842개를 추가하여 9169개의 형태소 사전을 만들었다.
   3. 결과
    ![voca2](https://user-images.githubusercontent.com/12870549/58157120-a4a7e400-7cb2-11e9-9ff7-2e98aeb991c9.png)
    - 전체 토큰의 수가 줄어 들었다. (약 1.5%)
    - [메나헴], [베긴] 과 같은 흔하지 않은 고유명사가 [UNK] 토큰화 되었다.
    - [UNK] 토큰의 수가 증가했다.(0.2% -> 1%)
   4. 아쉬운 점
     - 완전히 재구성하지 말고 기존 wordpiece 사전에 형태소만을 추가하는 방법또한 좋았을 것
     - 띄어쓰기 정보X
4. 학습
    
    [google-research/bert](https://github.com/google-research/bert)의 텐서플로우 구현코드에 형태소 분석기를 추가하여 학습을 진행했다. 학습 파라미터는 [BERT paper](https://arxiv.org/pdf/1810.04805.pdf)를 따라했으며, pretraining tip에 따라 학습데이터의 문장의 길이를 128로 두고 1차 학습한 뒤, 길이 512로 2차 학습을 하였다. 학습은 구글 클라우드 플랫폼의 Nvidia Tesla T4를 사용하였다.

- 1차 학습(max_seq_length - 128) LOSS, 10만 스텝
![pre_loss1](https://user-images.githubusercontent.com/12870549/58158773-0ddd2680-7cb6-11e9-94fa-bcaa8260b422.png)
- 2차 학습(max_seq_length - 512) LOSS, 100만 스텝
![pre_loss2](https://user-images.githubusercontent.com/12870549/58158771-0ddd2680-7cb6-11e9-8c0c-6e644e49ba25.png)

### Fine Tunings
1. 인공지능 QA

    [SQUAD](https://rajpurkar.github.io/SQuAD-explorer/) 의 한국판인 [KorQUAD](https://korquad.github.io/)의 데이터셋을 사용했다. [KorQuAD introduction](https://www.slideshare.net/SeungyoungLim/korquad-introduction)

    ![image](https://user-images.githubusercontent.com/12870549/58159814-39611080-7cb8-11e9-9d16-4bd39234358f.png)

    질문과 문단(input)에 대한 문단에서의 정답인덱스의 시작과 끝(output)을 학습한다. 
    ![squad-loss-log](https://user-images.githubusercontent.com/12870549/58160979-631b3700-7cba-11e9-8264-8045c9f48f28.png)

    - 1차 학습모델사용(주황색) / 2차 학습모델사용(파란색) 로스
    - 평가셋에 대한 F1 score
  
    |                           | F1 score | Exact Match | Loss(smoothed) | Epochs |
    |:-------------------------:|----------|-------------|----------------|--------|
    | 1차 사전훈련 모델(주황선) | 74.66    | 53.06       | 0.6043         | 3      |
    | 2차 사전훈련 모델(파랑선) | 83.43    | 62.57       | 0.2793         | 3      |

    최종적으로 평가셋을 포함한 데이터로 학습하였다. (평가셋 F1 score 92.68)

#### 실전 테스트

***DB에서 [tf-idf](https://ko.wikipedia.org/wiki/Tf-idf)알고리즘 를 통해 답이 있을 만한 문단을 찾은 뒤, Top 1 문단에서 답을 추론***
![image](https://user-images.githubusercontent.com/12870549/58178432-2ca4e280-7ce1-11e9-99a2-f6cfcfb11bd4.png)

```
Q. 한양대의 마스코트 하냥이를 만든 곳은?
A. 토이츄러스
```
2. 감성 분석

[Naver sentiment movie corpus](https://github.com/e9t/nsmc/)데이터를 사용하여 긍, 부정에 대한 학습을 하였다.

![image](https://user-images.githubusercontent.com/12870549/58162013-4da70c80-7cbc-11e9-9510-3fc5570bb9e5.png)

#### 정확도(주황선 - 학습데이터, 파란선 - 평가데이터)
![image](https://user-images.githubusercontent.com/12870549/58162772-c2c71180-7cbd-11e9-8ea5-1c9e16405062.png)

하나의 Logistic regression 레이어 추가로 손쉽게 학습데이터 정확도 96%이상, 평가데이터 정확도 93%를 얻을 수 있었다.

#### 실전 테스트
*** 1.0 에 가까우면 긍정, 0.0 에 가까우면 부정 ***

![image](https://user-images.githubusercontent.com/12870549/58179390-13049a80-7ce3-11e9-837c-ad13f83c6508.png)
![image](https://user-images.githubusercontent.com/12870549/58180828-ad65dd80-7ce5-11e9-9a82-736f7df5952c.png)
![image](https://user-images.githubusercontent.com/12870549/58181033-00d82b80-7ce6-11e9-9560-884a1419edc2.png)


3. 문장 임베딩

[bert-as-service](https://github.com/hanxiao/bert-as-service) like.... 작성 중



## 실행
1. MongoDB 서버 실행
2. Server 1 - (CPU)
```bash
python app.py --ip --port
```
3. Server 2 - (GPU)

[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 설치 이후

![image](https://user-images.githubusercontent.com/12870549/58275636-7fa99300-7dd0-11e9-9baa-60ba1aff82de.png)

```bash
인공지능 QA
docker run --runtime=nvidia -p 8501:8501 \
  --mount type=bind,source=/home/rhodochrosited/tensor_serving_models/search/,\
  target=/models/search -e MODEL_NAME=search -t tensorflow/serving:latest-gpu \
  --per_process_gpu_memory_fraction=0.3 &
```
```bash
문장임베딩
docker run --runtime=nvidia -p 8502:8501 \
  --mount type=bind,source=/home/rhodochrosited/tensor_serving_models/similarity/,target=/models/similarity -e \
  MODEL_NAME=similarity -t tensorflow/serving:latest-gpu \
  --per_process_gpu_memory_fraction=0.3 &
```
```bash
감성분석
docker run --runtime=nvidia -p 8503:8501 \
  --mount type=bind,source=/home/rhodochrosited/tensor_serving_models/sentiment/,target=/models/sentiment -e \
  MODEL_NAME=sentiment -t tensorflow/serving:latest-gpu \
  --per_process_gpu_memory_fraction=0.3 &
```