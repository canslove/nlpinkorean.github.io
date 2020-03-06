---
layout: post
published: True
title: Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)
ktitel: 신경망 기계 학습 모델의 시각화 (Seq2seq + Attention 모델의 메커니즘)
origina_date: 2018-05-09
date: 2018-12-17
author: 찬
original_author: Jay Alammar
---

최근 10년 동안의 자연어 처리 연구 중에 가장 영향력이 컸던 3가지를 꼽는 [서베이](https://docs.google.com/document/d/18NoNdArdzDLJFQGBMVMsQ-iLOowP1XXDaSVRmYN0IyM/mobilebasic)에서 여러 연구자들이 꼽았던 연구가 바로 2014년에 발표됐던 sequence-to-sequence (Seq2seq) + Attention 모델입니다 ([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), [Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)).

그 이후로 현재까지 수많은 연구들이 이런 encoder 와 decoder를 가지는 seq2seq 모델 형태를 가지고 있는데요, 이에 대해 쉽게 잘 설명이 된 영문 [블로그](https://jalammar.github.io) 글이 있어 저자의 허락을 받고 번역해 가져와 보았습니다. 원문은 아래의 링크에서 확인해주세요. 
([Jay Alammar - Visualizeing machine learning one concept at a time](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)).

이 글 외에도 다른 신경망 관련된 여러 최신 모델들과 개념이 잘 설명돼 있어 시간이 되신다면 추가로 확인해보셔도 좋을 것 같습니다.
또한 추후에 본 블로그에서 Transformer 와 Bert+ELMo 포스팅들도 가져와 번역할 예정입니다.

아래의 번역 글은 마우스를 올리시면 (모바일의 경우 터치) 원문을 확인하실 수 있습니다. 혹시 번역에 심각한 오류 혹은 오탈자를 확인하신다면 밑의 Disqus 댓글 창에 남겨주시면 감사하겠습니다.  
<p align="center">(이하 본문)</p>

---

## 신경망 기계 번역 모델의 시각화 (Seq2seq + Attention 모델의 메커니즘) by Jay Alammar

<div class="tooltip" markdown="1">
**주의사항:** 아래의 애니메이션은 비디오입니다. 클릭하시거나 마우스를 위로 올려두시면 재생됩니다. 
<span class="tooltiptext">
**Note:** The animations below are videos. Touch or hover on them (if you're using a mouse) to get play controls so you can pause if needed.
</span>
</div>


<div class="tooltip" markdown="1">
Sequence-to-sequence (Seq2seq) 모델은 기계 번역, 문서 요약, 그리고 이미지 캡셔닝 등의 문제에서 아주 큰 성공을 거둔 딥러닝 모델입니다.
구글 번역기도 2016년 말부터 이 모델을 실제 서비스에 <a href="https://blog.google/products/translate/found-translation-more-accurate-fluent-sentences-google-translate/">이용하고</a> 있습니다.
이 seq2seq 모델은 두 개의 선구자적인 논문에 의해 처음 소개되었습니다. (<a href="https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf">Sutskever et al., 2014</a>, <a href="http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf">Cho et al., 2014</a>).
<span class="tooltiptext">
Sequence-to-sequence models are deep learning models that have achieved a lot of success in tasks like machine translation, text summarization, and image captioning. Google Translate started using such a model in production in late 2016. These models are explained in the two pioneering papers (Sutskever et al., 2014, Cho et al., 2014).
</span>
</div>


<div class="tooltip" markdown="1">
그러나 이 모델을 구현을 할 수 있을 정도로까지 잘 이해하기 위해서는 모델 자체뿐만 아니라 이 모델의 기초에 이용된 수많은 기본 개념들을 이해하여야 합니다.
저는 이런 여러 개념들을 시각화하여 한 번에 볼 수 있다면 많은 사람들이 더 쉽게 이해할 수 있을 거라고 생각했습니다.
그것이 바로 제가 이번 포스트에서 목표하는 바입니다. 
다만, 이 포스트를 제대로 이해하시기 위해서는 딥러닝에 대한 사전 지식이 조금 필요하다는 점을 주의해주세요.
이 글이 앞에 언급했던 두 개의 논문들을 이해하는데 조금이라도 도움이 되길 바랍니다.
<span class="tooltiptext">
I found, however, that understanding the model well enough to implement it requires unraveling a series of concepts that build on top of each other. I thought that a bunch of these ideas would be more accessible if expressed visually. That's what I aim to do in this post. You'll need some previous understanding of deep learning to get through this post. I hope it can be a useful companion to reading the papers mentioned above (and the attention papers linked later in the post).
</span>
</div>


<div class="tooltip" markdown="1">
Seq2seq 모델은 글자, 단어, 이미지의 feature 등의 아이템 시퀀스 를 입력으로 받아 또 다른 아이템의 시퀀스를 출력합니다. 
학습된 모델은 다음과 같이 작동합니다:
<span class="tooltiptext">
A sequence-to-sequence model is a model that takes a sequence of items (words, letters, features of an images...etc) and outputs another sequence of items. A trained model would work like this:
</span>
</div>

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


<br />

<div class="tooltip" markdown="1">
신경망 기계 번역의 경우에 대해서 본다면, 입력은 일련의 단어로 이루어진 sequence 이며 맨 앞 단어부터 차례대로 모델에서 처리됩니다.
그리고 출력으론 비슷한 형태의 그러나 다른 언어로의 단어 sequence 가 나오게 됩니다:
<span class="tooltiptext">
In neural machine translation, a sequence is a series of words, processed one after another. The output is, likewise, a series of words:
</span>
</div>

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## 모델 안을 들여보기

<div class="tooltip" markdown="1">
이제 모델 안을 자세히 들여다보겠습니다. Seq2seq 모델은 하나의 <span class="encoder">encoder</span> 와 하나의 <span class="decoder">decoder</span> 로 이루어져 있습니다.
<span class="tooltiptext">
Under the hood, the model is composed of an <span class="encoder">encoder</span> and a <span class="decoder">decoder</span>.
</span>
</div>



<div class="tooltip" markdown="1">
<span class="encoder">encoder</span> 는 입력의 각 아이템을 처리하여 거기서 정보를 추출한 후 그것을 하나의 벡터로 만들어냅니다 (흔히 <span class="context">context</span> 라고 불립니다). 
입력의 모든 단어에 대한 처리가 끝난 후 <span class="encoder">encoder</span> 는 <span class="context">context</span>를 <span class="decoder">decoder</span> 에게 보내고 출력할 아이템이 하나씩 선택되기 시작합니다.
<span class="tooltiptext">
The <span class="encoder">encoder</span> processes each item in the input sequence, it compiles the information it captures into a vector (called the <span class="context">context</span>). After processing the entire input sequence, the <span class="encoder">encoder</span> send the <span class="context">context</span>  over to the <span class="decoder">decoder</span>, which begins producing the output sequence item by item.
</span>
</div>

<video width="100%" height="auto" loop autoplay  controls>
  <source src="/images/seq2seq/seq2seq_3.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>



<br />

<div class="tooltip" markdown="1">
물론 seq2seq 모델의 한 예시인 신경망 기계 번역도 동일한 구조를 가지고 있습니다.
<span class="tooltiptext">
The same applies in the case of machine translation.
</span>
</div>

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_4.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<div class="tooltip" markdown="1">
기계 번역의 경우에서는 <span class="context">context</span> 가 하나의 벡터의 형태로 전달됩니다.
<span class="encoder">encoder</span> 와 <span class="decoder">decoder</span> 는  둘 다 recurrent neural networks (RNN)를 이용하는 경우가 많습니다 (RNN에 대한 간단한 개요가 필요하시다면 Luis Serrano 의 유튜브 비디오 [A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg) 를 참고해주세요).
<span class="tooltiptext">
The <span class="context">context</span>  is a vector (an array of numbers, basically) in the case of machine translation. The <span class="encoder">encoder</span> and <span class="decoder">decoder</span>  tend to both be recurrent neural networks (Be sure to check out Luis Serrano's [A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg) for an intro to RNNs).
</span>
</div>

<div class="img-div" markdown="0">
    <img src="/images/seq2seq/seq2seq_context.png" />
<div class="tooltip" markdown="1">
<span class="context">context</span> 는 float 으로 이루어진 하나의 벡터입니다. 우리의 시각화 예시에서는 더 높은 값을 가지는 소수를 더 밝게 표시할 예정입니다.
<span class="tooltiptext">
The <span class="context">context</span>  is a vector of floats. Later in this post we will visualize vectors in color by assigning brighter colors to the cells with higher values.
</span>
</div>
</div>

<div class="tooltip" markdown="1">
이 <span class="context">context</span> 벡터의 크기는 모델을 처음 설정할 때 원하는 값으로 설정할 수 있습니다. 하지만 보통 <span class="encoder">encoder</span> RNN의 hidden unit 개수로 정합니다. 
이 글의 시각화 예시에서는 크기 4의 <span class="context">context</span> 벡터를 이용하는데요, 실제 연구에서는 256, 512, 1024 와 같은 숫자를 이용합니다.
<span class="tooltiptext">
You can set the size of the <span class="context">context</span>  vector when you set up your model. It is basically the number of hidden units in the <span class="encoder">encoder</span> RNN. These visualizations show a vector of size 4, but in real world applications the <span class="context">context</span> vector would be of a size like 256, 512, or 1024.
</span>
</div>

<br />
<div class="tooltip" markdown="1">
Seq2seq 모델 디자인을 보게 되면 하나의 RNN 은 한 타임 스텝마다 두 개의 입력을 받습니다.
하나는 sequence 의 한 아이템이고 다른 하나는 그전 스텝에서의 RNN의 hidden state입니다.
이 두 입력들은 RNN에 들어가기 전에 꼭 vector로 변환 되어야 합니다.
하나의 단어를 벡터로 바꾸기 위해서 우리는 "[word embedding](https://machinelearningmastery.com/what-are-word-embeddings/)" 이라는 기법을 이용합니다. 
이 기법을 통해 단어들은 벡터 공간에 투영되고, 그 공간에서 우리는 단어 간 다양한 의미와 관련된 정보들을 알아낼 수 있습니다.
(가장 유명한 예로 다음 식이 있습니다: [king - man + woman = queen](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)).
<div class="transcomment">(역주: 한글로 된 word2vec 과 관련된 글으로 <a href="https://shuuki4.wordpress.com/2016/01/27/word2vec-관련-이론-정리/">다음 글</a>을 추천합니다.)</div>
<span class="tooltiptext">
By design, a RNN takes two inputs at each time step: an input (in the case of the encoder, one word from the input sentence), and a hidden state. The word, however, needs to be represented by a vector. To transform a word into a vector, we turn to the class of methods called "[word embedding](https://machinelearningmastery.com/what-are-word-embeddings/)" algorithms. These turn words into vector spaces that capture a lot of the meaning/semantic information of the words (e.g. [king - man + woman = queen](http://p.migdal.pl/2017/01/06/king-man-woman-queen-why.html)).
</span>
</div>

<br />

<div class="img-div" markdown="0">
    <img src="/images/seq2seq/seq2seq_embedding.png" />
<div class="tooltip" markdown="1">
앞서 설명한 대로 encoder에서 단어들을 처리하기 전에 먼저 벡터들로 변환해주어야 합니다. 
우리는 <a href="https://en.wikipedia.org/wiki/Word_embedding">word embedding</a> 알고리즘을 이용해 변환합니다.
또한 우리는 <a href="http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/">pre-trained embeddings</a> 을 이용하거나 우리가 가진 데이터 셋을 이용해 직접 학습시킬 수 있습니다. 보통 크기 200 혹은 300의 embedding 벡터를 이용하지만, 이 포스트에서는 예시로서 크기 4의 벡터를 이용합니다.
<span class="tooltiptext">
We need to turn the input words into vectors before processing them. That transformation is done using a <a href="https://en.wikipedia.org/wiki/Word_embedding">word embedding</a> algorithm. We can use <a href="http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/">pre-trained embeddings</a> or train our own embedding on our dataset. Embedding vectors of size 200 or 300 are typical, we're showing a vector of size four for simplicity.
</span>
</div>
</div>

<div class="tooltip" markdown="1">
여기까지 모델에서 등장하는 주요 벡터들을 소개해보았는데요, 이제 RNN의 원리에 대해서 간단히 다시 돌아보고 시각화에서 우리가 쓸 기호들을 설명하겠습니다:
<span class="tooltiptext">
Now that we've introduced our main vectors/tensors, let's recap the mechanics of an RNN and establish a visual language to describe these models:
</span>
</div>


<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_RNN_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<br />
<div class="tooltip" markdown="1">
이와 같이 타임 스텝 #2 에서는 두번째 단어와 첫 번째 hidden state을 이용하여 두 번째 출력을 만듭니다.
본 포스트의 뒷부분에서는, 이와 유사한 애니메이션을 이용해 신경망 기계 번역 모델을 설명하겠습니다.
<span class="tooltiptext">
The next RNN step takes the second input vector and hidden state #1 to create the output of that time step. Later in the post, we'll use an animation like this to describe the vectors inside a neural machine translation model.
</span>
</div>

<br />

<div class="tooltip" markdown="1">
밑의 영상을 보시면, <span class="encoder">encoder</span> 혹은 <span class="decoder">decoder</span> 에서 일어나는 각 진동은 한 번의 스텝 동안 출력을 만들어내는 과정을 의미합니다.
<span class="encoder">encoder</span> 와 <span class="decoder">decoder</span> 는 모두 RNN이며, RNN은 한번 아이템을 처리할 때마다 새로 들어온 아이템을 이용해 그의 hidden state를 업데이트 합니다. 이 hidden state 는 그에 따라 <span class="encoder">encoder</span> 가 보는 입력 시퀀스 내의 모든 단어에 대한 정보를 담게 됩니다.
<span class="tooltiptext">
In the following visualization, each pulse for the <span class="encoder">encoder</span> or <span class="decoder">decoder</span>  is that RNN processing its inputs and generating an output for that time step. Since the <span class="encoder">encoder</span> and <span class="decoder">decoder</span>  are both RNNs, each time step one of the RNNs does some processing, it updates its <span class="context">hidden state</span>  based on its inputs and previous inputs it has seen.
</span>
</div>


<div class="tooltip" markdown="1">
그러면 시각화된 <span class="encoder">encoder</span>의 <span class="context">hidden states</span>  볼까요?
여기서 한가지 짚고 넘어갈 점은 마지막 단어의 <span class="context">hidden state</span>가 바로 우리가 <span class="decoder">decoder</span> 에게 넘겨주는 <span class="context">context</span> 라는 것입니다.
<span class="tooltiptext">
Let's look at the <span class="context">hidden states</span>  for the <span class="encoder">encoder</span>. Notice how the last <span class="context">hidden state</span>  is actually the <span class="context">context</span>  we pass along to the <span class="decoder">decoder</span>.
</span>
</div>

<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_5.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


<br />

<div class="tooltip" markdown="1">
<span class="decoder">decoder</span> 도 그만의 <span class="decoder">hidden states</span> 를 가지고 있으며 스텝마다 업데이트를 하게 됩니다.
우리는 아직 모델의 큰 그림을 그리고 있기 때문에 위의 영상에서는 그것을 표시하지 않았습니다.
<span class="tooltiptext">
The <span class="decoder">decoder</span>  also maintains a <span class="decoder">hidden states</span>  that it passes from one time step to the next. We just didn't visualize it in this graphic because we're concerned with the major parts of the model for now.
</span>
</div>

<div class="tooltip" markdown="1">
그렇다면 이제 이 seq2seq 모델을 다른 방법으로 시각화해보도록 하겠습니다.
아래의 영상은 이전 것들 보다 조금 더 정적인데요, 하나의 합쳐진 RNN 이 아닌 각 스텝마다 RNN 을 표시하는 방법입니다.
이렇게 하면 각 스텝마다 입력과 출력을 정확히 볼 수 있습니다.
<span class="tooltiptext">
Let's now look at another way to visualize a sequence-to-sequence model. This animation will make it easier to understand the static graphics that describe these models. This is called an "unrolled" view where instead of showing the one <span class="decoder">decoder</span>, we show a copy of it for each time step. This way we can look at the inputs and outputs of each time step.
</span>
</div>


<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_6.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<br />



# 이제 Attention 을 해봅시다
<div class="tooltip" markdown="1">
연구를 통해 <span class="context">context</span> 벡터가 이런 seq2seq 모델의 가장 큰 걸림돌인 것으로 밝혀졌습니다.
이렇게 하나의 고정된 벡터로 전체의 맥락을 나타내는 방법은 특히 긴 문장들을 처리하기 어렵게 만들었습니다.
이에 대한 해결 방법으로 제시된 것이 바로 "Attention" 입니다 [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) and [Luong et al., 2015](https://arxiv.org/abs/1508.04025). 
이 두 논문이 소개한 attention 메커니즘은 seq2seq 모델이 디코딩 과정에서 현재 스텝에서 가장 관련된 입력 파트에 집중할 수 있도록 해줌으로써 기계 번역의 품질을 매우 향상 시켰습니다.
<span class="tooltiptext">
The <span class="context">context</span>  vector turned out to be a bottleneck for these types of models. It made it challenging for the models to deal with long sentences. A solution was proposed in [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) and [Luong et al., 2015](https://arxiv.org/abs/1508.04025). These papers introduced and refined a technique called "Attention", which highly improved the quality of machine translation systems. Attention allows the model to focus on the relevant parts of the input sequence as needed.
</span>
</div>


<img src="/images/seq2seq/seq2seq_attention.png" />

<div class="img-div" markdown="0">
<div class="tooltip" markdown="1">
스텝 7 에서 attention 메커니즘은 영어 번역을 생성하려 할 때 <span class="decoder">decoder</span>가 단어 "étudiant" ("학생"을 의미하는 불어)에 집중하게 합니다.
이렇게 스텝마다 관련된 부분에 더 집중할 수 있게 해주는 attention model 은 attention 이 없는 모델보다 훨씬 더 좋은 결과를 생성합니다.
<span class="tooltiptext">
At time step 7, the attention mechanism enables the <span class="decoder">decoder</span>  to focus on the word "étudiant" ("student" in french) before it generates the English translation. This ability to amplify the signal from the relevant part of the input sequence makes attention models produce better results than models without attention.
</span>
</div>

</div>

<br />
<div class="tooltip" markdown="1">
계속해서 개략적인 차원에서 attention 모델을 살펴보도록 하겠습니다.
attention 모델과 기존의 seq2seq 모델은 2가지의 차이점을 가집니다:
<span class="tooltiptext">
Let's continue looking at attention models at this high level of abstraction. An attention model differs from a classic sequence-to-sequence model in two main ways:
</span>
</div>

<div class="tooltip" markdown="1">
첫 번째로 <span class="encoder">encoder</span> 가 <span class="decoder">decoder</span>에게 넘겨주는 데이터의 양이 attention 모델에서 훨씬 더 많다는 점입니다.
기존 seq2seq 모델에서는 그저 마지막 아이템의 hidden state 벡터를 넘겼던 반면 attention 모델에서는 *모든* 스텝의 <span class="context">hidden states</span>를  <span class="decoder">decoder</span>에게 넘겨줍니다:
<span class="tooltiptext">
First, the <span class="encoder">encoder</span> passes a lot more data to the <span class="decoder">decoder</span>. Instead of passing the last hidden state of the encoding stage, the <span class="encoder">encoder</span> passes _all_ the <span class="context">hidden states</span>  to the <span class="decoder">decoder</span>:
</span>
</div>


<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_7.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<br />

<div class="tooltip" markdown="1">
두 번째로는 attention 모델의 <span class="decoder">decoder</span>가 출력을 생성할 때에는 하나의 추가 과정이 필요합니다. <span class="decoder">decoder</span>는 현재 스텝에서 관련 있는 입력을 찾아내기 위해 다음 과정을 실행합니다:
<span class="tooltiptext">
Second, an attention <span class="decoder">decoder</span>  does an extra step before producing its output. In order to focus on the parts of the input that are relevant to this decoding time step, the <span class="decoder">decoder</span>  does the following:
</span>
</div>


<div class="tooltip" data-markdown="1">
1. <span class="encoder">encoder</span> 에서 받은 전체 <span class="context">hidden states</span>을 봅니다 -- 각 스텝에서의 <span class="context">encoder hidden states</span>는 이전의 맥락에 대한 정보도 포함하고 있지만 그 스텝에서의 입력 단어와 가장 관련이 있습니다. <br>
2. 각 스텝의 <span class="context">hidden state</span>마다 점수를 매깁니다 (일단 지금은 어떻게 점수를 매기는지에 대해서는 얘기하지 않겠습니다) <br>
3. 매겨진 점수들에 softmax를 취하고 이것을 각 타임 스텝의 <span class="context">hidden states</span>에 곱해서 더합니다. 이를 통해 높은 점수를 가진 <span class="context">hidden states</span>는 더 큰 부분을 차지하게 되고 낮은 점수를 가진 <span class="context">hidden states</span>는 작은 부분을 가져가게 됩니다. <br>
<span class="tooltiptext">
1. Look at the set of encoder <span class="context">hidden states</span> it received -- each <span class="context">encoder hidden states</span>  is most associated with a certain word in the input sentence  <br>
2. Give each <span class="context">hidden states</span>  a score (let's ignore how the scoring is done for now)  <br>
3. Multiply each <span class="context">hidden states</span>  by its softmaxed score, thus amplifying <span class="context">hidden states</span>  with high scores, and drowning out <span class="context">hidden states</span>  with low scores  <br>
</span>
</div>


<video width="100%" height="auto" loop autoplay controls>
   <source src="/images/seq2seq/seq2seq_attention_process.mp4" type="video/mp4">
   Your browser does not support the video tag.
</video>

<br><br>

<div class="tooltip" markdown="1">
이러한 점수를 매기는 과정은 <span class="decoder">decoder</span>가 단어를 생성하는 매 스텝마다 반복됩니다.
<span class="tooltiptext">
This scoring exercise is done at each time step on the <span class="decoder">decoder</span> side.
</span>
</div>


<div class="tooltip" markdown="1">
이제 이때까지 나온 모든 과정들을 합친 다음 영상을 보고 어떻게 attention 이 작동하는지 정리해보겠습니다:
<span class="tooltiptext">
Let us now bring the whole thing together in the following visualization and look at how the attention process works:
</span>
</div>


<div class="tooltip" markdown="1">
1. attention 모델에서의 decoder RNN 은 <span class="embedding">\<END\></span>과 추가로 <span class="decoder">initial decoder hidden state</span>을 입력받습니다.
1. decoder RNN 은 두 개의 입력을 가지고 새로운 <span class="decoder">hidden state</span>벡터를 출력합니다. (<span class="decoder">h</span><span class="step_no">4</span>). RNN의 출력 자체는 사용되지 않고 버려집니다.
1. Attention 과정: encoder의 <span class="context"> hidden state</span> 모음과 decoder 의 hidden state <span class="decoder">h</span><span class="step_no">4</span> 벡터를 이용하여 그 스텝에 해당하는 context 벡터 (<span class="step_no">C</span><span class="decoder">4</span>) 를 계산합니다.
1. <span class="decoder">h</span><span class="step_no">4</span> 와  <span class="step_no">C</span><span class="decoder">4</span> 를 하나의 벡터로 concatenate (연결, 이어쓰기) 합니다.
1. 이 벡터를 <span class="ffnn">feedforward 신경망</span> (seq2seq 모델 내에서 함께 학습되는 layer 입니다) 에 통과 시킵니다.
1. <span class="ffnn">feedforward 신경망</span>에서 나오는 <span class="logits_output">출력</span>은 현재 타임 스텝의 출력 단어를 나타냅니다.
1. 이 과정을 다음 타임 스텝에서도 반복합니다.
<span class="tooltiptext">
The attention decoder RNN takes in the embedding of the <span class="embedding">\<END\></span> token, and an <span class="decoder">initial decoder hidden state</span>.
The RNN processes its inputs, producing an output and a <span class="decoder">new hidden state</span> vector (<span class="decoder">h</span><span class="step_no">4</span>). The output is discarded.
Attention Step: We use the <span class="context">encoder hidden states</span> and the <span class="decoder">h</span><span class="step_no">4</span> vector to calculate a context vector (<span class="step_no">C</span><span class="decoder">4</span>) for this time step.
We concatenate <span class="decoder">h</span><span class="step_no">4</span> and <span class="step_no">C</span><span class="decoder">4</span> into one vector.
We pass this vector through a <span class="ffnn">feedforward neural network</span> (one trained jointly with the model).
The <span class="logits_output">output</span> of the feedforward neural networks indicates the output word of this time step.
Repeat for the next time steps
</span>
</div>

<video width="100%" height="auto" loop autoplay controls>
   <source src="/images/seq2seq/seq2seq_attention_tensor_dance.mp4" type="video/mp4">
   Your browser does not support the video tag.
</video>

<br><br>

<div class="tooltip" markdown="1">
이 attention 을 이용하면 각 decoding 스텝에서 입력 문장에서 어떤 부분을 집중하고 있는지에 대해 볼 수 있습니다:
<span class="tooltiptext">
This is another way to look at which part of the input sentence we're paying attention to at each decoding step:
</span>
</div>


<video width="100%" height="auto" loop autoplay controls>
  <source src="/images/seq2seq/seq2seq_9.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<div class="tooltip" markdown="1">
여기서 한가지 짚고 넘어갈 것은 현재 모델이 아무 이유 없이 출력의 첫 번째 단어를 입력의 첫 번째 단어와 맞추는 (align) 것이 아니란 것입니다.
학습 과정에서 입력되는 두 개의 언어를 어떻게 맞출지는 학습이 됩니다 (우리의 예시에는 불어와 영어입니다).
얼마나 이것이 정확하게 학습 되는지를 알아보기 위해서 앞서 언급했던 attention 논문들에서는 다음과 같은 예시를 보여줍니다:
<span class="tooltiptext">
Note that the model isn't just mindless aligning the first word at the output with the first word from the input. It actually learned from the training phase how to align words in that language pair (French and English in our example). An example for how precise this mechanism can be comes from the attention papers listed above:
</span>
</div>

<div class="img-div" markdown="0">
<img src="/images/seq2seq/seq2seq_attention_sentence.png" />
<div class="tooltip" markdown="1">
모델이 "European Economic Area"를 제대로 출력할 때 모델이 얼마나 잘 주의를 하고 있는지를 볼 수 있습니다.
영어와는 달리 불어에서는 이 단어들의 순서가 반대입니다 ("européenne économique zone").
문장 속의 다른 단어들은 다 비슷한 순서를 가지고 있습니다.
<span class="tooltiptext">
You can see how the model paid attention correctly when outputing "European Economic Area". In French, the order of these words is reversed ("européenne économique zone") as compared to English. Every other word in the sentence is in similar order.
</span>
</div>
</div>

<br>

<div class="tooltip" markdown="1">
이제 구현을 할 준비가 됐다고 느껴지신다면 TensorFlow의 [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)를 꼭 확인해보세요. 
<span class="tooltiptext">
If you feel you're ready to learn the implementation, be sure to check TensorFlow's [Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt).
</span>
</div>

<br />

<div class="tooltip" markdown="1">
여기에 그려진 것들은 제가 Udacity에서 하고 있는 강의 [Natural Language Processing Nanodegree Program](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892)의 한 수업 중 일부분입니다. 
이 강의에서 관련된 응용 부분들과 Transformer 모델 ([Attention Is All You Need](https://arxiv.org/abs/1706.03762))과 같은 attention 메커니즘을 활용한 최근 연구 등등 더욱더 자세한 것을 다루고 있으니 관심이 있으시다면 확인해보세요. 
<span class="tooltiptext">
I hope you've found this useful. These visuals are early iterations of a lesson on attention that is part of the Udacity [Natural Language Processing Nanodegree Program](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892). We go into more details in the lesson, including discussing applications and touching on more recent attention methods like the Transformer model from  [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
</span>
</div>

<div class="tooltip" markdown="1">
이 글에 대해서 피드백이 있으시다면 제 트위터 아이디 [@jalammmar](https://twitter.com/jalammar)로 연락주세요.
<span class="tooltiptext">
I'd love any feedback you may have. Please reach me at [@jalammmar](https://twitter.com/jalammar).
</span>
</div>