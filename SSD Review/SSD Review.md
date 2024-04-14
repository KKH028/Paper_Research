# SSD : Single Shot MultiBox Detector

## 1. Introduction

- 1-stage detector인 YOLO는 45 frames per second(FPS:Frame Per Sec)로 7 FPS인 Faster R-CNN보다 **속도가 크게 향상**되었지만 <br>
YOLO mAP : 63.4%과 Faster R-CNN : mAP 74.3% 비교했을때 **정확도가 낮다**는것을 알 수 있다. 이를 개선하기위해 만든것이 SSD이다.
> 2-stage detector인 Faster R-CNN의 정확도와 1-stage detector의 성능을 가지는 모델
- 논문의 저자들이 요약한 **contribution**은 다음과 같다.
> 1) small convolutional Predictor filter들을 사용하여 feature map에 적용해 <br>
고정된 default bounding boxes의 category[Confidence] score 및 box offset을 예측하는 것이다. <br>
> 2) detection의 높은 정확도를 얻기 위해 다양한 크기(scale)의 feature map에서 다양한 크기(scale)의 prediction을 생성하고, aspect ratio(종횡비)별로 예측을 구분한다. <br>
> 3) 이러한 design feature들은 저해상도 input image에서도 간단한 end-to-end training과 높은 정확도로 이어져, <br>
속도와 정확도의 trade-off를 더욱 개선한다.



## 2 The Single Shot Detector (SSD)

SSD 접근 방식은 고정 크기의 바운딩 박스와 해당 박스에 존재하는 객체 클래스 인스턴스의 점수를 생성하는 전방향 합성곱 신경망에 기반하며, 최종 검출을 위해 NMS 단계를 거칩니다. 초기 네트워크 레이어는 고품질 이미지 분류에 사용되는 표준 아키텍처를 기반으로 합니다(분류 레이어 이전에 잘라냄). 이를 '베이스 네트워크'라고 부르며, 여기에 보조 구조를 추가하여 다음과 같은 주요 기능을 가진 검출을 생성합니다:

### 2.1 Model

Multi-scale feature maps for detection. We add convolutional feature layers to the end
of the truncated base network. These layers decrease in size progressively and allow
predictions of detections at multiple scales. The convolutional model for predicting
detections is different for each feature layer (cf Overfeat[4] and YOLO[5] that operate
on a single scale feature map).

감지를 위한 다중 스케일 피처 맵. 우리는 잘라낸 기본 네트워크의 끝에 컨볼루션 피처 레이어를 추가합니다. 이 레이어는 점진적으로 크기가 작아지며 여러 스케일에서 감지 예측을 가능하게 합니다. 감지를 예측하기 위한 컨볼루션 모델은 각 피처 레이어마다 다르며 (Overfeat[4] 및 YOLO[5]와 같이 단일 스케일 피처 맵에서 작동하는 모델과는 다릅니다).

Convolutional predictors for detection. Each added feature layer (or optionally an existing feature layer from the base network) can produce a fixed set of detection predictions using a set of convolutional filters. These are indicated on top of the SSD network
architecture in Fig. 2. For a feature layer of size m × n with p channels, the basic element for predicting parameters of a potential detection is a 3 × 3 × p small kernel
that produces either a score for a category, or a shape offset relative to the default box
coordinates. At each of the m × n locations where the kernel is applied, it produces an
output value. The bounding box offset output values are measured relative to a default box position relative to each feature map location (cf the architecture of YOLO[5] that
uses an intermediate fully connected layer instead of a convolutional filter for this step).

감지를 위한 컨볼루션 예측기. 각 추가된 피처 레이어(또는 기본 네트워크에서 선택적으로 기존 피처 레이어)는 일련의 컨볼루션 필터를 사용하여 고정된 집합의 감지 예측을 생성할 수 있습니다. 이는 그림 2의 SSD 네트워크 아키텍처 상단에 표시됩니다. 크기가 m × n이고 p개의 채널을 갖는 피처 레이어에서 잠재적인 감지의 매개변수를 예측하는 기본 요소는 3 × 3 × p 작은 커널입니다. 이 커널은 카테고리에 대한 점수 또는 기본 상자 좌표에 대한 모양 오프셋 중 하나를 생성합니다. 커널이 적용되는 m × n 개의 위치마다 출력 값을 생성합니다. 경계 상자 오프셋 출력 값은 각 피처 맵 위치에 대한 기본 상자 위치와 상대적으로 측정됩니다(YOLO[5]의 아키텍처와 비교하면, 이 단계에서 컨볼루션 필터 대신 중간의 완전히 연결된 레이어를 사용합니다).

Default boxes and aspect ratios. We associate a set of default bounding boxes with
each feature map cell, for multiple feature maps at the top of the network. The default
boxes tile the feature map in a convolutional manner, so that the position of each box
relative to its corresponding cell is fixed. At each feature map cell, we predict the offsets
relative to the default box shapes in the cell, as well as the per-class scores that indicate
the presence of a class instance in each of those boxes. Specifically, for each box out of
k at a given location, we compute c class scores and the 4 offsets relative to the original
default box shape. This results in a total of (c + 4)k filters that are applied around each
location in the feature map, yielding (c + 4)kmn outputs for a m × n feature map. For
an illustration of default boxes, please refer to Fig. 1. Our default boxes are similar to
the anchor boxes used in Faster R-CNN [2], however we apply them to several feature
maps of different resolutions. Allowing different default box shapes in several feature
maps let us efficiently discretize the space of possible output box shapes.


기본 상자와 종횡비. network의 top에서 여러 피처 맵에 대해 각 feature map cell에 일련의 기본 경계 상자를 연관시킵니다. 기본 상자는 피처 맵을 컨볼루션 방식으로 타일 처리하여 해당하는 셀에 대한 각 상자의 위치가 고정됩니다. 각 피처 맵 셀에서는 해당 셀의 기본 상자 모양에 대한 오프셋과 해당 상자에 클래스 인스턴스가 존재하는지를 나타내는 클래스별 점수를 예측합니다. 특정 위치에서 k개의 상자 중 하나의 경우, 원본 기본 상자 모양에 상대적인 c개의 클래스 점수와 4개의 오프셋을 계산합니다. 이로 인해 피처 맵의 각 위치 주변에 (c+4)k개의 필터가 적용되어, m × n 피처 맵에 대해 (c+4)kmn개의 출력을 얻게 됩니다. 기본 상자에 대한 설명은 그림 1을 참조하십시오. 저희의 기본 상자는 Faster R-CNN[2]에서 사용되는 앵커 상자와 유사하지만, 서로 다른 해상도의 여러 피처 맵에 적용됩니다. 서로 다른 기본 상자 모양을 여러 피처 맵에 허용하면 가능한 출력 상자 모양의 공간을 효율적으로 이산화할 수 있습니다.

###  Default boxes and aspect ratios

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/9f64d855-c1fc-487a-b9f6-c2e91ef36da9)


- 모델에서 Prediction을 수행하는 각 feature map들마다 bounding box regression과 object detection을 수행하기 위해 해당 feature map에 3x3 predictor filter를 가지고 convolution을 하게된다. <br> 
> SSD에서는 이때 Faster-RCNN 에서 region proposal을 위해 사용했던 앵커(Anchor)박스와 유사한 개념인 **default box**를 사용하게 된다. <br>

- 각 feature map마다 어떻게 몇개의 bounding box를 뽑는지에 대해 살펴보자. <br>
> 논문에서는 각 feature map에 차례대로 4개,6개,6개,6개,4개,4개의 default box들을 먼저 선정했다. 


> - 그 중 두번째의 19x19 Con7 피쳐맵에 대해서 bounding box를 생성하는 과정을 하나의 예제로 살펴보면 <br> 
이 feature map에서는 6개의 default bounding box를 만들고 이 박스와 대응되는 자리에서 <br>
예측되는 박스의 offset과 class score를 예측한다. <br> 
이것을 선정한 default box의 갯수만큼 반복하여 다양한 object를 찾아낸다. 즉,<br>
>> __6(default box) X (20개:Classes-PASCAL VOC기준 + 1:object인지 아닌지)=21+4(offset 좌표)) = 150(ch)__ <br>
이것이 default boxes(바운딩박스들)이 담고있는 정보이다.

![ssd3](https://github.com/KKH0228/Paper-Research/assets/166888979/1d06e360-9998-42c7-8648-1df38408414b)

> 이를 모든 feature map 6개의 로케이션에 대해서 뽑아내면 논문에서 소개된 <br>
38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 <br>
= 총 8732개의 바운딩박스를 뽑는 과정이다. <br>

#### + Matching strategy

훈련하는 동안 어떤 default box들이 ground truth 탐지에 해당하는지 결정하고 그에 따라 네트워크를 훈련해야 합니다.

- 각각의 ground truth box에 대해서, 위치, 종횡비 및 배율이 다른 default box들을 선택

- 각 ground truth box를 최상의 자카드 오버랩(IoU)이 있는 default box와 매칭시키는 것으로 시작

- 임계값(0.5)보다 높은 jaccard 오버랩을 사용하여 default box들을 ground truth와 매칭

이것은 학습 문제를 단순화하여 네트워크가 최대 중첩이 있는 default box만 선택하도록 요구하지 않고 multiple overlapping default box들에 대해 높은 점수를 예측할 수 있도록 합니다.

### Choosing scales and aspect ratios for default boxes

![ssd5](https://github.com/KKH0228/Paper-Research/assets/166888979/7bdb3e8b-2801-440f-a4a5-aa7e3a690f7f)

- SSD에서 다양한 레이어의 피쳐맵들을 사용하는것은 scale variance를 다룰 수 있다. <br>
- bounding box를 찾는데에 있어 위의 그림처럼 8x8의 feature map에서는 default box가 <br> 
상대적으로 작은 물체(고양이)를 찾는데에 높은 IoU가 매칭될 것이고, <br> 
4x4 feature map에서는 상대적으로 큰 물체(강아지)에게 매칭 될 것이다. <br>
> **즉 앞쪽에 resolution이 큰 feature map에서는 작은 물체를 감지하고, <br>
뒤쪽에 resolution이 작은 feature map에서는 큰 물체를 감지할 수 있다**는 것이 <br>
multiple feature map 사용의 메인 아이디어이다. <br>

- 그렇다면 default boxes들은 어떻게 다양하게 만들어주는지 살펴볼 수 있다.


> $S_k = S_{min} + \frac{S_{max} - S_{min}}{m-1}(k-1), k ∈[1,m]$ <br>
> $ S_{min} = 0.2, S_{max} = 0.9 $
* $S_k$ = scale
* $m$ = feature map의 갯수
* $k$ = feature map index

- 위의 식으로 m개의 feature map에서 bounding box를 뽑아낸다. <br>
- 각 k값을 차례대로 넣어보면 PASCAL VOC 기준으로 <br>
$S_k = 0.1, 0.2, 0.375, 0.55, 0.725, 0.9$
라는 scale을 얻을 수 있다. <br>
> $S_0 = 0.1$ (chapter3.1의 PASCAL VOC 2007에서 conv4_3의 scale을 0.1로 setting)

- 이것은 전체 Input 이미지에서의 각 비율을 의미한다. <br>
> 즉 Input이 300x300 이미지이기 때문에 0.1은 30픽셀, 0.2는 60픽셀,각각 <br>
$ 30,60,112.5,165,217.5,270 pixels $ <br>
이와 같이 인풋 이미지를 대변한다. <br>

- 이렇게 scale이 정해지면 아래의 식으로 default box의 크기가 정해진다. <br>
> $a_r ∈$ {$1,2,3,\frac{1}{2},\frac{1}{3} $} <br>
$w^a_k = S_k \sqrt{a_r} $ <br>
$h^a_k = S_k /\sqrt{a_r} $ <br><br>


* For the aspect ratio of 1, we also add a default box whose scale is $S'_k = \sqrt{S_kS_k+1}$, resulting in 6 default boxes per feature map location.
> aspect ratio가 1인 경우 scale이 $S'_k = \sqrt{S_kS_k+1}$인  default box도 추가하여 feature map location당 6개의 default box가 생성됩니다.


* 아래의 식으로 default box의 중심점을 구할 수 있다.
> $(\displaystyle \frac{i+0.5}{|f_k|},\frac{j+0.5}{|f_k|})$ <br>
$|f_k|$ = k번째 feature map의 가로세로 크기 , $i,j ∈ [0,|f_k|]$ <br>

![ssd6](https://github.com/KKH0228/Paper-Research/assets/166888979/fb984ebb-af71-412b-949c-f4bed55975b7)


* In practice, one can also design a distribution of default boxes to best fit a specific dataset. How to design the optimal tiling is an open question as well.

논문 3.1 PASCAL VOC2007에서 논문에서 테스트한 SSD 모델의 디폴트 박스에 대한 설정이 설명되어 있다.\
For conv4_3, conv10_2 and conv11_2, we only associate 4 default boxes at each feature map location – omitting aspect ratios of 1/3 and 3. For all other layers, we put 6 default boxes as described in Sec.

### Loss function
- training 목표 : mutiple object detection

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/c9a9ba77-261b-481b-9a19-6f9cd5d38ae8)


#### 0) **SSD training 목표는 MultiBox에서 파생되었으나, multiple object categories를 다루는것이 확장되었다.**
- MultiBox의 Loss식은 다음과 같다.
> $F_{conf}(x,c)=-\displaystyle\sum_{i,j}x_{ij}log(c_i)-\displaystyle\sum_{i}(1-\displaystyle\sum_jx_{ij})log(1-c_i)$
> - binary loss와 유사
> - $x_{ij} = $1 or 0
> - **IoU가 가장 높은 box만 가져와서** $\sum x_{ij}=1$이지만
- SSD는 IoU가 가장 높은 box뿐만 아니라 <br>
jaccard overlap(중복되는 부분)이 **thredhold(0.5)이상이면 모두 default box들**로 보기때문에 $\sum x^{p}_{ij}\geq1$이 된다.
> 그래서 전체 loss에서 default box들의 개수인 **$N$으로 나눠주는 이유**이다.

#### 1) $L_{conf}(x,c)$(cross entropy)

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/7a60c134-76ac-4f00-9804-491b76fcf97d)

- $x^p$$_{ij}$값은 물체의 j번째 ground truth와 i번째 default box간의 IOU가 0.5 이상이면 1, 0.5미만이면 0을 대입해준다. <br>
따라서 물체의 j번째 ground truth와 i번째 default box 간의 IOU가0.5 미만인 default box는 식에서 0이 되어 사라지게된다.

**$\Rightarrow$ 너무많은 default box가 있다. 그래서 back ground를 가리키고 있는  default box를 날려서 default box의 수를 최소화하는 작업이다.**

#### 2) $L_{loc}(x,l,g)$

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/5ba0f0fa-3042-4244-829a-8bc16e49a81b)


- $ d $ : **default box**

- $ g $ : **ground truth box**
> $\hat{g}_{j}^{cx}$= (${g}_{j}^{cx}$-${d}_{i}^{cx}$) / ${d}_{i}^{w}$,<br>
$\hat{g}_{j}^{cy}$= (${g}_{j}^{cy}$-${d}_{i}^{cy}$) / ${d}_{i}^{h}$,<br> 
$\hat{g}_{j}^{w}$ = $log({g}_{j}^{w}/{d}_{i}^{w}), $ <br> 
$\hat{g}_{j}^{h}$ = $log({g}_{j}^{h}/{d}_{i}^{h})$

*$\hat{g}$ = Bounding Box Target


- $ l $ : **prediction box**
> $\hat{l}_{i}^{cx}$= (${l}_{i}^{cx}$-${d}_{i}^{cx}$) / ${d}_{i}^{w}$,<br>
$\hat{l}_{i}^{cy}$= (${l}_{i}^{cy}$-${d}_{i}^{cy}$) /  ${d}_{i}^{h}$,<br> 
$\hat{l}_{i}^{w}$ = $log({l}_{i}^{w}$/${d}_{i}^{w}$),<br>
$\hat{l}_{i}^{h}$ = $log({l}_{i}^{h}$/${d}_{i}^{h}$)


-  $ L_{reg} $ : **regression loss(smooth $ L_1 $)** 
> $ L_{reg}$( $\hat{l}_{i}^{m}$,$\hat{g}_{j}^{m}$) = R($\hat{l}_{i}^{m}$,$\hat{g}_{j}^{m}$) , $R$ : the rubust loss function (smooth $ L_1 $)<br>
> $ = L_1$($\hat{l}_{i}^{cx}$-$\hat{g}_{j}^{cx}$)$ + L_1$($\hat{l}_{i}^{cy}$-$\hat{g}_{j}^{cy}$)+  $L_1$(log($\hat{l}_{i}^{w}$)-(log($\hat{g}_{j}^{w}$ ))+$L_1$(log($\hat{l}_{i}^{h}$)-log($\hat{g}_{j}^{h}$ ))

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/2988b272-a87f-43f2-96a5-9b1dc27117a3)



#### 3) $L_{loc}$(x,l,g)식에 나오는 용어정리
* x : Conv Predictor Filter에 의해 생성된 각 Feature Map의 Grid Cell
* x<sub>i<sub> : x에 포함되어는 특정 디폴트 박스의 offset과 class score.
* $x^{p}_{ij}$ : 

* N : ground truth와 매치된 default box의 개수
* l : predicted box (예측된 상자)
* g : ground truth box
* d : default box
* cx, cy : 그 박스의 x, y좌표
* w, h : 그 박스의 width, heigth
* $\alpha$ 에는 1을 대입해준다. 
* ground truth 와 default box의 IOU값이 0.5이상인것만 고려한다.<br>
=background object인 negative sample 에 대해서는 Localization Loss를 계산하지 않는다.


## 3. Hard negative mining

- 사실 매칭 스텝 후에 생성된 대부분의 default boxes(bounding boxes)은 object가 없는 background이다. <br>
이것은 Faster R-CNN에서도 대두되었던 문제인데 negative인, 즉 background들을 사용해 트레이닝하는것은 모델에 도움이 되지않는다. <br>
Faster R-CNN에서는 256개의 미니배치에서 128개의 positive, 128개의 negative를 사용해서 훈련하도록 권고했으나 <br> 
negative가 너무 많으면 이 비율은 어쩔수없이 유지하기 힘들다. <br>
> SDD에서는 이러한 positive와 negative의 inbalance문제에 대해 confidence loss가 높은 negative값들을 뽑아 <br> 
> **positive와 negative를 1:3 비율**로 사용하길 제안했다. 이것으로 더 빠른 최적화와 안정적인 트레이닝을 이끌었다.


## 4. Data augmentation
- 전체 원본 트레이닝 데이터 세트 input 이미지를 사용한다.
- object와 jaccard IoU가 최소인 0.1, 0.3, 0.5, 0.7, 0.9이 되도록 패치를 샘플링한다.
- 랜덤하게 패치를 샘플링한다.
> 샘플링시 $\frac{1}{2}$와 $2$사이로 랜덤하게 정한다.

## 5. Experimental Results

### 5-1. Base Network
- 논문에서의 실험은 모두 **VGG16**을 기반으로 train하는데 layer를 다음과 같이  수정한다. <br>

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/ecfd7b21-0887-4100-be2a-6a00ba9290b0)


1. VGG16 내 full connected layer 6,7을 subsampling parameter를 생성하는 **convolutional layer**로 변환한다.
2. pool5 layer(linear)를 $2$x$2$-s2, $3$x$3$-s1의 *atrous 알고리즘*을 사용하여 pool5 layer를 바꾼다.
> 일반 VGG16을 사용하는 경우 pool5를 2×2-s2 fc6 및 fc7의 subsampling parameter가 아니라 예측을 위해 conv5 3을 추가하면 결과는 거의 같지만 속도는 약 20% 느려진다.<br>
> **즉, *atrous 알고리즘*이 더 빠르다.**
3. drop out과 full connected layer 8을 없앤다.

#### + *A'trous 알고리즘 (dilated convolution)* <br>
<img src='https://drive.google.com/uc?export=download&id=12ey06VHMA4uSykBaIyXRDmDmPUJ-tVeJ' height="300" width="300"> <br>
- 기존 컨볼루션 필터가 수용하는 픽셀 사이에 간격을 둔 형태이다. <br>
입력 픽셀 수는 동일하지만, 더 넓은 범위에 대한 입력을 수용할 수 있게 된다.
- 즉, convolutional layer에 또 다른 parameter인 **dilation rate**를 도입했다.<br> 
> dilation rate는 **커널 사이의 간격**을 정의한다. <br>
> dilation rate가 2인 3x3 커널은 9개의 parameter를 사용하면서, 5x5 커널과 동일한 view를 가지게 된다.
- 적은 계산 비용으로 Receptive Field 를 늘리는 방법이라고 할 수 있다. <br>
이 A'trous 알고리즘은 필터 내부에 zero padding 을 추가해서 강제로 Receptive Field 를 늘리게 되는데, <br>
위 그림에서 진한 파란색 부분만 weight가 있고 나머지 부분은 0으로 채워지게 된다. <br>
이 Receptive Field는 필터가 한번 보는 영역으로 사진의 Feature를 파악하고, 추출하기 위해서는 넓은 Receptive Field 를 사용하는 것이 좋다. <br>
dimension 손실이 적고, 대부분의 weight가 0이기 때문에 연산의 효율이 좋다. <br>
공간적 특징을 유지하는 Segmentation에서 주로 사용되는 이유이다. 

#### Classifier : Conv 3X3X(4X(classes+4))가 나오는 이유,방식
> * **same conv 연산**하기때문

![classifier](https://github.com/KKH0228/Paper-Research/assets/166888979/3c25a63f-4712-47f1-9282-5ad814a4743b)

> * 결과 : 38x38x(4X(classes+4)) <br>
> * 19x19x(6X(classes+4)), 10x10x(6X(classes+4)), 5x5x(6X(classes+4)), <br>
3x3x(4X(classes+4)), 1x1x(4X(classes+4))도 위와 같은 방식

### 5-2. More default box shapes is better.
>- 기본적으로 6개의 default box를 사용하지만 aspect ratio=$\displaystyle \frac{1}{3}, 3$을 제거하면 성능이 0.6% 감소한다.
>- 다양한 default box shape을 사용하면 네트워크에서 box를 예측
하는 작업이 더 쉬워진다는 것을 알 수 있다.

## 6. Conclusions
- 핵심은 top layer에서 multiple feature map이 연결된 다양한 크기의 convolutional bounding box를 출력한다는 것이다.
- 가능한 box shape의 공간을 효율적으로 모델링할 수 있다.
> 단점 : 작은 feature map에서 큰 object만 detection한다.

---
Reference

[참고1](https://csm-kr.tistory.com/4)
[참고2:atrous algorithm](https://eehoeskrap.tistory.com/431)

[paper](https://arxiv.org/pdf/1512.02325.pdf) 
[1](https://csm-kr.tistory.com/4) 
[2](https://eehoeskrap.tistory.com/431) 
[3](http://www.okzartpedia.com/wordpress/index.php/2020/07/16/ssd-single-shot-detector/) 
[4](https://ys-cs17.tistory.com/12)
