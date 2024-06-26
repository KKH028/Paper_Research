# Faster R-CNN

[Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) <br>
__Shaoqing Ren, Kaiming He,
Ross Girshick, and Jian Sun__

## 1. Introduction

기존 fast R-CNN에서 object detection 작업을 end-to-end모델로 학습시켰지만 <br>
region proposal을 selective search로 해야한다는 단점을 극복한것이 바로 Faster R-CNN이다. <br>
이때부터 본격적으로 RealTime Object Detection이 가능해졌고 성능 또한 놀라울정도로 향상되었다. <br>

> " **Anchor box 개념을 도입하며 Region Proposal 또한 전체 네트워크속으로 가져오는것(RPN)** "이 Faster R-CNN의 핵심이다.


## 2. Orders of algorithm

<img src='https://drive.google.com/uc?export=download&id=1svoAwO3PTvF-FRGUc288_JmP6jVaNALk' height="400" width="350">
<img src='https://drive.google.com/uc?export=download&id=1KnCQGJnct_3QyjZYAAzbDXiUt-0nJEfe' height="400" width="400">

Faster R-CNN의 순서는 다음과 같다.

1. input이미지를 pre-trained된 CNN에 통과시켜 피쳐맵을 추출한다.

2. 이를 Selective Search가 아닌 RPN(Region Proposal Network)에 전달하여 RoI 박스를 계산한다. (Anchor박스 사용)

3. 이렇게 얻은 RoI박스에 RoI pooling을 진행한다.

4. Classification을 수행해 Object detection을 한다.

5. Alternate training으로 RPN과 Classification을 번갈아 훈련시킨다.

> 핵심 아이디어는 **RPN(Region Proposal Network)**이다. <br>
**Selective Search를 제거함으로써 GPU를 통한 RoI 박스 계산이 가능**해졌으며, <br>
RoI 박스 계산 역시도 학습시켜 정확도를 높일 수 있었다.

## 3. Core concepts

논문에서 소개되는 주요 개념들(Anchor box, RPN, loss function)을 살펴보겠다.

### 3-1. Anchor Box

#### 1) 정의

<center><img src='https://drive.google.com/uc?export=download&id=1fiJn13TvBZUTd8A92DD6RuYrvTx-z0OE' height="300" width="450"></center> <br>


*   특정 객체 클래스의 크기와 종횡비를 캡처하도록 정의되며 일반적으로 교육 데이터 세트의 객체 크기를 기반으로 선택된다. 감지하는 동안 미리 정의된 anchor box가 이미지 전체에 바둑판식으로 배열이 되고, 네트워크는 배경 결합에 대한 intersection over union 및 모든 타이릴된 앵커 상자에 대한 오프셋과 같은 확률 및 기타 속성을 예측한다.

*   네트워크는 bounding box를 직접 예측 하는 것이 아는 타일링된 anchor box에 해당하는 확룰과 미세 조정을 예측한다. 최종 feature map에는 각 클래스에 대한 객체 감지가 나타나고, anchor box를 사용 함으로 여러개체, 다양한 크기의 개체 및 겹치는 개체를 감지 할수 있다.


#### 2) Process
- 앵커 박스의 위치는 네트워크 출력의 위치를 입력 이미지에 다시 다른값을 주어 결정하게 된다. 프로세스는 네트워크 출력에 대해 모두 복제가 되며, 그 결과 전체 이미지에 걸쳐 타일링된 anchor box가 생성이 된다. 생성된 anchor box는 클래스의 특정 예측을 나타낸다.

<center><img src='https://drive.google.com/uc?export=download&id=1LGNOUQTbF132k5WOWGkQ0RwzTOsTnylD' height="250" width="750"> </center> <br>

- 예를 들어 위치당 두 개의 예측을 수행하는 두 개의 anchor box가 있다.
> 이 anchor box는 이미지 전체에 바둑판식으로 배열이 되며, 타이링된 anchor box와 네트워크 출력의 수가 같기 때문에 네트워크는 모드 출력에 대한 예측을 생성하게 된다.

<center><img src='https://drive.google.com/uc?export=download&id=1DoDtvtha8kArmx886TTTwQkgk2TRGYns' height="350" width="650"> <br>

- Selective search를 통해 region proposal를 추출하지 않을 경우, 원본 이미지를 일정 간격의 grid로 나눠 각 gird cell을 bounding box로 간주하여 feature map에 encode하는 Dense Sampling방식을 사용한다.

- sub-sampling ratio를 기준으로 grid를 나누게 된다. 원본 이미지의 크기가 800$×$800이며, sub-sampling ratio가 1/100이라고 하면, CNN 모델에 입력시켜 얻은 최종 feature map의 크기는 8$×$8가 된다. 여기서 feature map의 각 cell은 원본 이미지의 100$×$100만큼의 영역에 대한 정보가 합축되어 있으며, 원본이미지에서는 8$×$8개만큼의 bounding box가 생성된다.

<center><img src='https://drive.google.com/uc?export=download&id=1T4hUJbn843wpZ5_xaIVxRaUWTrvLed-k' height="400" width="1000"> </center> <br>

- 고정된 크기의 bounding box를 사용할 경우, 다양한 크기의 객체를 포착하지 못하여, 이러한 문제점을 해결하고 지정한 위치에 사전에 정의한 서로 다른 크기(scale)와 가로세로비(aspect ratio)를 가지는 bouding box인 Achor Box를 생성하여 다양한 크기의 객체를 포착하는 방법을 제시한다.
> 논문에서 (128,256,512)scale3개와 (1:1,1:2,2:1)aspect ratio3개를 가진 9개의 서로 다른 anchor box를 정의 하였다.

#### 3) 공식

<img src='https://drive.google.com/uc?export=download&id=1b82cNGmZR-kNOfDCdBKZih1-nRiHLI46' height="250" width="100"> <br>

*   Scale은 anchor box의 widght(=$w$), height(=$h$)의 길이
*   aspect ratio는 widht(=$w$), height(=$h$) 비율
> 예를 들어 asepct ratio가 1:1 일 때 <br>
anchor box의 넓이는 $s^2 = s × s$ 이 된다






<center> <img src='https://drive.google.com/uc?export=download&id=1L-ibgsD8b9D0njJsLqjNKGWXFdkR8cRc' height="350" width="550"></center> <br>


- anchor box는 원본 이미지의 각 grid cell의 중심을 기준으로 생성한다.
> 원본 이미지에서 sub-sampling ratio를 기준으로 anchor box를 생성하는 기준점인 anchor를 고정을 하고 anchor를 기준으로 사전에 정의한 anchor box 9개를 생성한다.
>> 위에 이미지의 크기는 600 $×$ 800 이며, sub-sampling ratio = 1/16이다
>> anchor의 생성수는 600/16 $\times$ 800/16를 한 1900개가 생성이 된다.
>> anchorbox는 anchor 생성수 1900 $\times$ 9인 총 17100개가 만들어 진다

- **anchor box를 사용하면, 고정된 크기의 bounding box를 사용할때보다 9배많은  bounding box를 생성하여, 다양한 크기의 객체를 얻기가 가능하다**



#### 4) 장점

*   앵커박스를 사용하면 모든 잠재적 위치에서 별도의 예측을 계산하는 슬라이딩 윈도우처럼 이미지를 스캔하지 않고, <br>
전체 이미지를 한 번에 처리할 수 있어서 실시간 물체 감지 시스템이 가능하다.

<center><img src='https://drive.google.com/uc?export=download&id=1NAB_H17Ieu7TepIHqmQcihVko4ktgmj2' height="200" width="550"> <br>
슬라이딩 윈도우의 구조 <br> </center>

*   CNN은 입력 이미지를 convolutional방식으로 처리하여 입력의 공간 위치는 출력의 공간 위치 관련 될 수 있다. covolutional은 down sampling의 한 종류로 cnn이 전체 이미지에 대한 이미지 특징을 한 번에 추출할 수 있는 것을 의미하는데 이때 추출된 것을 해당 이미지에 다시 연결할 수 있다. <br>
앵커 박스를 사용하면 이미지에서 특징을 추출하기 위한 슬라이딩 윈도우 접근 방식을 대체하여 <br> **비용을 크게 줄일수 있으며, 검출, 특징 인코딩 및 분류를 통해 효율적으로 객체를 감지할 수 있다.**

#### 5) Localization Errors and Refinement
- anchor box사이의 거리는 cnn에 존재하는 다운샘플링의 양의 함수이다. 다운샘플링 은 불필요한 타일링 된 anchor box를 생성하여 Localization Errors를 일으킨다.

<center><img src='https://drive.google.com/uc?export=download&id=1zbTzGB2oX4zyF1NAWwbrwX8tnqf_DgA1' height="350" width="550"> </center> <br>

- Localization Errors를 수정하기 위해서는 anchor box의 위치와 크기를 조정하는 각 타일링 된 anchor box에 적용할 offeset을 학습시킨다.

<center><img src='https://drive.google.com/uc?export=download&id=1eazLySV8XSqV1LW4PmfXVDV3_umCr3Jm' height="400" width="400"> </center> <br>

- 다운샘플링레이어를 제거하기 위해서는 다운샘플링을 줄이는데 이때 convolution2d layer 및 maxPooling2d layer를 줄여준다. 특징 추출 계층은 공간해상도가 더 높지만 네트워크 아래에 있는 계층에 비해 더 적은 의미정보를 추출한다.

### 3-2. RPN (Region Proposal Network)

<center><img src='https://drive.google.com/uc?export=download&id=1jGFTEMnufxo6USgwhwmOO6jqxSsaZY0c' height="350" width="850"> <br>
<center>Faster R-CNN의 구조 <br>

<center><img src='https://drive.google.com/uc?export=download&id=1AxwP9jZf4caMqPtW6aLsuEsj1j4yJQxR' height="300" width="450"> <br>
RPN의 내부

> 처음 컨볼루션 연산 후 생성된 공유 피쳐맵은 RPN으로 들어가 RoI를 계산하게된다. <br>

1. CNN으로 뽑은 피쳐 맵을 입력으로 받는다. 이 때, 피쳐맵의 크기를 HxWxC로 잡는다. <br>
> 각각 세로, 가로, 채널 수이다.

2. proposal generation을 위해 피쳐맵에 3x3 컨볼루션을 256 혹은 512 채널만큼 수행한다. <br>
> 위 그림에서 intermediate layer에 해당한다. 이 때, padding을 1로 설정해주어 가로, 세로의 크기를 보존한다. <br> 수행 결과 HxWx256 or HxWx512 크기의 두 번째 intermediate layer 피쳐 맵을 얻는다.

3. 두 번째 피쳐맵을 입력 받아서 각각 classification과 bounding box regression 예측 값을 계산해주어야 한다. <br> 입력 이미지의 크기에 상관없이 동작할 수 있도록 1x1 컨볼루션을 수행한다.

4. Classification을 수행하기 위해서 1x1 컨볼루션을 2(오브젝트 인지 아닌지 나타내는 지표 수)x9(앵커 개수) 체널 수 만큼 수행해주며, <br>
그 결과로 HxWx18 크기의 피쳐맵을 얻는다. HxW 상의 하나의 인덱스는 피쳐맵 상의 좌표를 의미하고, <br> 그 아래 18개의 체널은 각각 해당 좌표를 앵커로 삼아 k(9)개의 앵커 박스들이 object인지 아닌지에 대한 예측 값을 담고 있다. <br>
> 즉, 한번의 1x1 컨볼루션으로 H x W 개의 앵커 좌표들에 대한 예측을 모두 수행한 것이다.

5. 두 번째로 Bounding Box Regression 예측 값을 얻기 위한 1x1 컨볼루션을 4(좌표)x9(앵커) 채널 수 만큼 수행한다.

6. 앞서 얻은 값들로 RoI를 계산한다. <br>
>먼저 Classification을 통해서 얻은 object일 확률 값들을 정렬한 다음,
Top N-rank(높은 순)으로 K개의 앵커만 추려낸다. <br>
그 다음 K개의 앵커들에 각각 Bounding box regression을 적용해준다. <br>
그리고 Non-Maximum-Suppression을 적용하여 RoI을 구해준다.

> 이렇게 RPN을 이용해서 RoI 박스를 만들어내내는 구조를 알아보았다. <br>
> - 이렇게 찾은 RoI를 첫 번째 피쳐맵에 project한 다음 RoI Pooling을 적용하고, <br>
> - 이것을 dectector로 계산하면 object가 무엇인지 종류도 알아낼 수 있다. <br>
>> 이 부분은 Fast R-CNN 구조를 그대로 계승한다. <br>


> 이 RPN 내부에서 수행되는 작업들을 조금 더 디테일하게 알아보자.

#### + generating Anchor boxes

<center><img src='https://drive.google.com/uc?export=download&id=1RokGDO6cCtPjg6bMzcoOR9ZK3yo2NnlB' height="300" width="600"></center> <br>

- 앵커 박스는 proposal을 만들어내는 핵심 요소이다. <br>
> 논문에서는 k = 9개의 앵커박스를 각각의 피쳐맵 센터에 배치시킨다. <br>
> 각 앵커박스는 3개의 비율(ratio)와 3개의 크기(size)를 갖는다.

- 6개의 파라미터($d_x,d_y,d_w,d_h$ 좌표, $p_{obj},p_{bg}$클래스 확률 0,1)를 이용해 각 앵커들은 하나의 proposal을 만들어낸다. <br>
> - box regression은 selective search RoI박스를 바운딩박스로 맞춰주는 작업과 비슷하다. <br>
> - 하지만 앵커박스를 사용하면 spatial(3x3)한 부분에만 웨이트를 사용하고, <br>
각 앵커박스를 학습하기 때문에 고정된 피쳐스케일이라도 여러 박스를 예측할 수 있다. <br>
(최대 앵커박스의 갯수만큼 = 여기선 9개)


- 이후 모든 grid cell을 통과시켜나오면 box regression과 class probabilty에 대한 각 parameter map이 생긴다. <br>
> 이것은 HxWx36 (box) ,HxWx18 (cls) 이 된다. <br>
- 여기에 NMS (non-max suppression)을 적용하면 우리가 필요로하는 최종적인 RoI box (proposal)을 얻을 수 있다. <br><br>

<center><img src='https://drive.google.com/uc?export=download&id=1UnpyYi3aMtx1w93RbQcprWqIciECSJZq' height="300" width="800"> <br>
<center>RPN의 진행 구조





### 3-3. loss function

#### 0) 앵커 레이어
- 먼저 RPN을 training하기위해, 각 앵커 박스에 object가 있는지 없는지 binary분류 label을 배정한다.
- 앵커 레이어에서 앵커 박스에 배정하는데 두종류의 기준이 있다.
> 1. ground-truth box와 겹치는 IoU가 가장 높은 앵커 또는 앵커들
> 2. ground-truth box와 겹치는 IoU가 0.7 이상인 앵커
- IoU가 0.3보다 낮으면 background에 음수를 할당한다. <br>
> 긍정도 부정도 아닌 앵커는 train목표에 기여하지않는다.

#### 1) RPN Loss
-  RPN에서는 위 앵커 레이어를 통해서 만들어진 앵커 박스들로 **앵커 타겟 박스**를 만든다.
- 이 과정에서 두가지 layer에 따른 loss가 나오는데 classification layer와 bounding box regression layer이다.
- RPN Loss = Classification loss + BBR loss (multi task loss)


<center><img src='https://drive.google.com/uc?export=download&id=1Qt3-OSdxIozWOs320vTBP_G0vFHtHfG5' height="250" width="1000">

##### - 정의 : $ L(\{p_i\},\{t_i\}) = \displaystyle \frac{1}{N_{cls}} \displaystyle \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*) $
> $cls$와 $reg $ layer는 각각 $ \{p_i\},\{t_i\} $로 구성되어있다.

###### (1) **classification loss :** $\displaystyle \frac{1}{N_{cls}} \displaystyle \sum_i L_{cls}(p_i, p_i^*) $
> object인지, background인지 구분하는 layer, object가 있는지 없는지에 따른 확률
- $ i $ : mini-batch안 앵커박스의 인덱스 값
- $ p_i $ : classification할때 앵커박스 $i$에 object가 있는지, background인지 예측된 **확률**
- $ p^*_i $ : ground truth **label**
>> if 앵커박스 : positive면(object가 있으면), $ p^*_i =1$ <br>
> if 앵커박스 : negaitive면(object가 없고 배경이면), $ p^*_i =0$ <br>
>>> 예시) $(p_i(obj), p_i(bg))$  <br>
>>> $p_1= (0.6, 0.4) => p^*_1= (1, 0)$ : object <br>
>>> $p_2= (0.2, 0.8) => p^*_2= (0, 1)$ : background

- $ L_{cls}(p_i, p_i^*) $ : classification loss(log) <br>
> $ L_{cls}(p_i, p_i^*)$ = $-(p_i^*(obj) \times log(p_i(obj))+p^*_i(bg) \times log(p_i(bg))) $  <br>
- $ \displaystyle \frac{1}{N_{cls}}$ : mini-batch 사이즈로 정규화 (논문에서 $N_{cls}=256$)

###### (2) **bounding box regression loss :** $ \lambda \frac{1}{N_{reg}} \displaystyle\sum_i p_i^* L_{reg}(t_i, t_i^*) $
> ground truth box와 object가 있는 앵커박스와의 loss 값

<img src='https://drive.google.com/uc?export=download&id=1NyoDrLth5kmjPQ5KfXrJOQD1d3XBpaYa' height="350" width="500">

- $ t_i $ : **ground truth box**
> $ t_x = (x-x_a) / w_a,$ <br> $t_y=(y-y_a)/h_a,$ <br>$
t_w = log(w/w_a),$ <br> $ t_h = log(h/h_a) $
>> $ x_a, x, x^* : $ proposal x, 앵커박스의 x좌표,anchor target 박스의 x좌표
- $ t_i^* $ : **object가 있는 앵커박스와 연관된 bounding box**
> $ t^*_x = (x^*-x_a) / w_a, $ <br>
$t^*_y=(y^*-y_a)/h_a, $ <br>
$t^*_w = log(w^*/w_a), $ <br>
$t^*_h = log(h^*/h_a) $

-  $ L_{reg} $ : regression loss(smooth $ L_1 $) <br>
> $ L_{reg}(t_i, t_i^*) = R(t_i-t_i^*) $, $R$ : the rubust loss function (smooth $ L_1 $)<br>
> $ = L_1(t_x-t^*_x) + L_1(t_y-t^*_y) + L_1(log(t_w)-log(t^*_w)) + L_1(log(t_h)-log(t^*_h))$ <br>
> * bounding box regression offset <br>
> * 앵커위치 개수로 정규화 <br>
>> parameter $ \lambda $로 맞춰준다.

- $N_{reg}$ : 앵커 location의 갯수($N_{reg}$ ~ $2,400$)

#### 2) Top(Detection) layer Loss
- RPN loss와 유사하게 Classification loss + Bounding Box regression loss로 구성되어있다.
- RPN loss와 차이점
> RPN은 object와 background이라는 두가지 class만 처리하지만 <br>
> top layer에서는 **background를 포함한 모든 object의 class들**을 처리한다.
- Fast R-CNN의 loss 구하는 원리와 동일하다.

##### - 정의

$ L(\{p_i\},\{t_i\}) = \displaystyle \sum_i L_{cls}(p_i, p_i^*) + \sum_i p_i^* L_{reg}(t_i, t_i^*) $

###### (1) **classification loss :** $\displaystyle \sum_i L_{cls}(p_i, p_i^*) $
> object가 **어느 class의 object인지(obj1,obj2,obj3...), background인지** 구분하는 layer

- $ L_{cls}(p_i, p_i^*) $ : cross-entropy loss <br>
> 예시) $ L_{cls}(p_i, p_i^*)$ = $-(p_i^*(bg) \times log(p_i(bg))+
p^*_i(obj1) \times log(p_i(obj1)))
- (p^*_i(obj2) \times log(p_i(obj2))
+ p^*_i(obj3) \times log(p_i(obj3))$

###### (2) **bounding box regression loss :** $ \displaystyle \sum_i p_i L_{reg}(t_i, t_i^*) $
> RPN의 positive anchor(proposal)인 object에 대한 mini-batch loss

- $ t_i $ : **ground truth box**
- $ t_i^* $ : **object가 있는 앵커박스와 연관된 bounding box**
- $ t_i $와 $ t_i^* $의 class별 IoU 확률을 regression한후 maxIoU값이 0.5보다 큰 경우

> * $ p_i L_{reg} $
>> $ p_i^*=1$일때(object가 있을때)만 활성화되고 <br> $ p_i^*=0$일땐(object가 없을때, background) 사라진다,<br>
: background의 regression 좌표값은 고려하지않는다.


