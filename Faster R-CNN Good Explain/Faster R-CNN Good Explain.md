# Object Detection and Classification using R-CNNs

In this post, I’ll describe in detail how R-CNN (Regions with CNN features), a recently introduced deep learning based object detection and classification method works. R-CNN’s have proved highly effective in detecting and classifying objects in natural images, achieving mAP scores far higher than previous techniques. The R-CNN method is described in the following series of papers by Ross Girshick et al.

R-CNN (Girshick et al. 2013)*
Fast R-CNN (Girshick 2015)*
Faster R-CNN (Ren et al. 2015)*
This post describes the final version of the R-CNN method described in the last paper. I considered at first to describe the evolution of the method from its first introduction to the final version, however that turned out to be a very ambitious undertaking. I settled on describing the final version in detail.

Fortunately, there are many implementations of the R-CNN algorithm available on the web in TensorFlow, PyTorch and other machine learning libraries. I used the following implementation:

https://github.com/ruotianluo/pytorch-faster-rcnn

Much of the terminology used in this post (for example the names of different layers) follows the terminology used in the code. Understanding the information presented in this post should make it much easier to follow the PyTorch implementation and make your own modifications.

이 글에서는 최근 소개된 딥러닝 기반 객체 탐지 및 분류 방법인 R-CNN(Regions with CNN features)에 대해 자세히 설명합니다. R-CNN은 자연 이미지에서 객체를 감지하고 분류하는 데 매우 효과적이며, 이전 기술보다 높은 mAP 점수를 얻습니다. R-CNN 방법은 Ross Girshick 등의 연구진이 발표한 논문 시리즈에서 설명됩니다.

* R-CNN(Girshick et al. 2013)*
* Fast R-CNN(Girshick 2015)*
* Faster R-CNN(Ren et al. 2015)*

이 글에서는 마지막 논문에서 설명된 R-CNN 방법의 최종 버전에 대해 설명합니다. 처음부터 최종 버전까지 방법의 진화를 설명하려고 했지만, 이는 매우 야심찬 작업이었습니다. 따라서 최종 버전을 자세히 설명하기로 결정했습니다.

다행히도 TensorFlow, PyTorch 및 기타 머신러닝 라이브러리에서 R-CNN 알고리즘의 많은 구현이 웹 상에 있습니다. 다음 구현을 사용했습니다.

https://github.com/ruotianluo/pytorch-faster-rcnn

이 글에서 사용되는 많은 용어(예: 다른 레이어의 이름)는 코드에서 사용하는 용어를 따릅니다. 이 글에서 제시된 정보를 이해하면 PyTorch 구현을 따르고 자신의 수정을 만드는 것이 훨씬 쉬워집니다.

## Network Organization
A R-CNN uses neural networks to solve two main problems:

1. Identify promising regions (Region of Interest – ROI) in an input image that are likely to contain foreground objects
2. Compute the object class probability distribution of each ROI – i.e., compute the probability that the ROI contains an object of a certain class. The user can then select the object class with the highest probability as the classification result.

R-CNNs consist of three main types of networks:

1. Head
2. Region Proposal Network (RPN)
3. Classification Network

R-CNNs use the first few layers of a pre-trained network such as ResNet 50 to identify promising features from an input image. Using a network trained on one dataset on a different problem is possible because neural networks exhibit “transfer learning” (Yosinski et al. 2014)*. The first few layers of the network learn to detect general features such as edges and color blobs that are good discriminating features across many different problems. The features learnt by the later layers are higher level, more problem specific features. These layers can either be removed or the weights for these layers can be fine-tuned during back-propagation. The first few layers that are initialized from a pre-trained network constitute the “head” network. The convolutional feature maps produced by the head network are then passed through the Region Proposal Network (RPN) which uses a series of convolutional and fully connected layers to produce promising ROIs that are likely to contain a foreground object (problem 1 mentioned above). These promising ROIs are then used to crop out corresponding regions from the feature maps produced by the head network. This is called “Crop Pooling”. The regions produced by crop pooling are then passed through a classification network which learns to classify the object contained in each ROI.

As an aside, you may notice that weights for a ResNet are initialized in a curious way:

네트워크 구성
R-CNN은 두 가지 주요 문제를 해결하기 위해 신경망을 사용합니다.

1. 전경 객체를 포함할 가능성이 높은 입력 이미지의 유망한 영역 (ROI) 식별
2. 각 ROI의 객체 클래스 확률 분포 계산 - 즉, ROI가 특정 클래스의 객체를 포함하는 확률을 계산합니다. 그런 다음 사용자는 가장 높은 확률을 가진 객체 클래스를 분류 결과로 선택할 수 있습니다.

R-CNN은 세 가지 주요 유형의 신경망으로 구성됩니다.

1.Head
2. Region Proposal Network (RPN)
3. Classification Network

R-CNN은 입력 이미지에서 유망한 기능을 식별하기 위해 사전 훈련된 ResNet 50과 같은 네트워크의 처음 몇 개의 레이어를 사용합니다. 다른 문제에서 하나의 데이터 집합에서 훈련된 네트워크를 사용하는 것은 신경망이 "전이 학습"(Yosinski et al. 2014)을 보여주기 때문에 가능합니다. 네트워크의 처음 몇 개의 레이어는 여러 가지 다른 문제에 걸쳐 구별 기능이 우수한 엣지 및 색상 덩어리와 같은 일반적인 기능을 감지하는 방법을 배웁니다. <br><br> 이후 레이어에서 학습된 특징은 더 높은 수준의 문제 특정 기능입니다. 이러한 레이어는 제거하거나 역전파 중에 이러한 레이어의 가중치를 미세 조정할 수 있습니다. 사전 훈련된 네트워크에서 초기화된 처음 몇 개의 레이어는 "head" 네트워크를 구성합니다. head 네트워크에서 생성된 합성곱 특성 맵은 이후 유망한 ROI를 생성하기 위해 사용되는 RPN(Region Proposal Network)을 통해 전달됩니다. <br><br> RPN은 전경 객체를 포함할 가능성이 높은 유망한 ROI를 생성하기 위해 합성곱 및 완전 연결 레이어 시리즈를 사용합니다. (위에서 언급한 문제)이러한 유망한 ROI는 그 후 head 네트워크에서 생성된 특성 맵에서 해당 영역을 자르는 데 사용됩니다. 이를 "Crop Pooling"이라고 합니다. crop pooling에서 생성된 영역은 각 ROI에 포함된 객체를 분류하는 것을 학습하는 분류 네트워크를 통해 전달됩니다.

부가적으로, ResNet의 가중치가 특이한 방식으로 초기화되는 것을 알 수 있습니을 볼 수 있을 것입니다.

부가적인 설명으로, ResNet의 가중치(weights) 초기화 방법이 꽤 특이하게 느껴질 수 있습니다.



If you are interested in learning more about why this method works, read my post about initializing weights for convolutional and fully connected layers.

만약 이 방법이 왜 동작하는지에 대해 더 알고 싶다면, 합성곱 및 완전 연결 레이어의 가중치 초기화에 대한 내 포스트를 읽어보세요.

## Network Architecture
The diagram below shows the individual components of the three network types described above. We show the dimensions of the input and output of each network layer which assists in understanding how data is transformed by each layer of the network. w and h represent the width and height of the input image (after pre-processing).



## 네트워크 아키텍처
아래 다이어그램은 위에서 설명한 세 가지 네트워크 유형의 개별 구성 요소를 보여줍니다. 각 네트워크 레이어의 입력과 출력 차원을 표시하여 네트워크의 각 레이어가 데이터를 어떻게 변환하는지 이해하는 데 도움이 됩니다. 여기서 w와 h는 입력 이미지의 너비와 높이를 나타냅니다(전처리 후)


## Implementation Details: Training
In this section, we’ll describe in detail the steps involved in training a R-CNN. Once you understand how training works, understanding inference is a lot easier as it simply uses a subset of the steps involved in training. The goal of training is to adjust the weights in the RPN and Classification network and fine-tune the weights of the head network (these weights are initialized from a pre-trained network such as ResNet). Recall that the job of the RPN network is to produce promising ROIs and the job of the classification network to assign object class scores to each ROI. Therefore, to train these networks, we need the corresponding ground truth i.e., the coordinates of the bounding boxes around the objects present in an image and the class of those objects. This ground truth comes from free to use image databases that come with an annotation file for each image. This annotation file contains the coordinates of the bounding box and the object class label for each object present in the image (the object classes are from a list of pre-defined object classes). These image databases have been used to support a variety of object classification and detection challenges. Two commonly used databases are:



## 구현 세부 사항: 훈련

이 섹션에서는 R-CNN을 훈련하는 데 필요한 단계에 대해 자세히 설명하겠습니다. 훈련이 어떻게 작동하는지 이해하면 추론을 이해하는 것이 훨씬 쉬워집니다. <br><br> 추론은 훈련에 포함된 단계의 하위 집합을 사용하기 때문입니다. 훈련의 목표는 RPN 및 분류 네트워크의 가중치를 조정하고 head 네트워크의 가중치를 미세 조정하는 것입니다(이러한 가중치는 ResNet과 같은 사전 훈련된 네트워크에서 초기화됩니다). RPN 네트워크의 역할은 유망한 ROI를 생성하는 것이고, 분류 네트워크의 역할은 각 ROI에 대해 객체 클래스 점수를 할당하는 것입니다. 따라서 이러한 네트워크를 훈련하기 위해 우리는 해당하는 ground truth가 필요합니다 <br><br> 즉, 이미지에 있는 객체 주위의 바운딩 박스의 좌표 및 해당 객체의 클래스입니다. 이 ground truth는 각 이미지에 대한 주석 파일이 함께 제공되는 무료로 사용할 수 있는 이미지 데이터베이스에서 제공됩니다. 이 주석 파일에는 이미지에 존재하는 각 객체에 대한 바운딩 박스의 좌표와 객체 클래스 레이블이 포함되어 있습니다(객체 클래스는 사전 정의된 객체 클래스 목록에서 제공됩니다). <br><br> 이러한 이미지 데이터베이스는 다양한 객체 분류 및 감지 도전 과제를 지원하는 데 사용되었습니다. 두 가지 일반적으로 사용되는 데이터베이스는:

* **Intersection over Union (IoU) Overlap**: We need some measure of how close a given bounding box is to another bounding box that is independent of the units used (pixels etc) to measure the dimensions of a bounding box. This measure should be intuitive (two coincident bounding boxes should have an overlap of 1 and two non-overlapping boxes should have an overlap of 0) and fast and easy to calculate. A commonly used overlap measure is the “Intersection over Union (IoU) overlap, calculated as shown below.

- **Intersection over Union (IoU) Overlap** : 주어진 바운딩 박스가 다른 바운딩 박스와 얼마나 가까운지를 측정하는 데 사용되는 측정 기준이 필요합니다. 이 측정은 바운딩 박스의 치수를 측정하는 데 사용되는 단위와 독립적이어야 합니다(예: 픽셀 등). 이 측정은 직관적이어야 합니다(두 겹치는 바운딩 박스는 1의 겹침을 가져야 하며 두 겹치지 않는 박스는 0의 겹침을 가져야 함) 및 빠르고 쉽게 계산할 수 있어야 합니다. 일반적으로 사용되는 겹침 측정 방법 중 하나는 "Intersection over Union (IoU) overlap"입니다. 아래와 같이 계산됩니다.

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/331ac5bd-ef34-461e-8d46-7cab9b296a5f)


With these preliminaries out of the way, lets now dive into the implementation details for training a R-CNN. In the software implementation, R-CNN execution is broken down into several layers, as shown below. A layer encapsulates a sequence of logical steps that can involve running data through one of the neural networks and other steps such as comparing overlap between bounding boxes, performing non-maxima suppression etc.

이러한 사전 준비를 마치고 나면 R-CNN을 훈련하는 소프트웨어 구현의 세부 사항에 대해 알아보겠습니다. 소프트웨어 구현에서는 R-CNN 실행을 여러 계층으로 분해합니다. 계층은 데이터를 신경망 중 하나를 통해 실행하는 등의 논리적 단계의 일련을 캡슐화하며, 경계 상자 간의 겹침을 비교하거나 비최대 억제 등의 다른 단계를 포함할 수 있습니다.


```python

```
