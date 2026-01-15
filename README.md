# AI Model Training Guide

[![Bright Data Promo](https://github.com/bright-kr/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://brightdata.co.kr/)

이 가이드는 OpenAI의 도구 세트를 활용한 파인튜닝을 통해 AI 모델 성능을 향상시키는 방법을 다룹니다:

- [AI 및 모델 학습 소개](#introduction-to-ai-and-model-training)
- [파인튜닝 준비하기](#getting-ready-for-fine-tuning)
- [단계별 파인튜닝](#step-by-step-fine-tuning)
- [성공을 위한 전략](#strategies-for-success)

## Introduction to AI and Model Training

AI 시스템은 학습 및 추론과 같은 인간과 유사한 인지 작업을 수행합니다. 이러한 모델은 알고리즘을 사용하여 데이터로부터 예측을 수행하며, 머신러닝은 명시적 프로그래밍이 아니라 경험을 통해 성능이 개선되도록 합니다.

AI 학습은 아이들이 배우는 방식과 유사합니다. 즉, 패턴을 관찰하고 예측을 수행하며 실수로부터 학습합니다. 모델은 향후 예측을 위해 패턴을 인식하도록 데이터를 제공받고, 성능은 예측값을 알려진 결과와 비교하여 측정합니다.

모델을 처음부터 만드는 것은 사전 지식 없이 패턴 인식을 가르치는 것을 의미하므로 방대한 리소스가 필요하며, 데이터가 제한적일 경우 종종 최적 이하의 결과를 초래합니다.

파인튜닝은 일반적인 패턴을 이미 이해하고 있는 사전 학습된 모델에서 시작한 뒤, 특화된 데이터셋로 추가 학습을 진행합니다. 이 접근 방식은 보통 더 적은 리소스로 더 나은 결과를 제공하므로, 특화 데이터가 제한적인 경우에 이상적입니다.

## Getting Ready for Fine-Tuning

특화된 데이터셋에 대한 추가 학습을 통해 기존 모델을 강화하는 것은 제로부터 구축하는 것에 비해 유리해 보이지만, 성공적인 파인튜닝은 몇 가지 핵심 요소에 달려 있습니다.

### Model Selection Strategy

파인튜닝을 위한 베이스 모델을 선택할 때는 다음 요소를 고려합니다:

**Task Alignment:** 목표와 기대 기능을 명확히 정의해야 합니다. 유사한 작업에서 성능이 좋은 모델을 선택하십시오. 원래 목적과 목표 목적 간 불일치는 효과를 떨어뜨릴 수 있습니다. 예를 들어 텍스트 생성에는 [GPT-3](https://openai.com/index/gpt-3-apps/)가 유리할 수 있으며, 텍스트 분류에는 [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) 또는 [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta)가 더 적합할 수 있습니다.

**Model Size and Complexity:** 더 큰 모델은 더 복잡한 패턴을 포착하지만 더 많은 연산 자원을 요구하므로, 성능과 리소스 요구사항 간 적절한 균형을 찾으십시오.

**Evaluation Metrics:** 특정 작업에 관련된 성능 측정 지표를 선택하십시오. 분류 프로젝트는 정확도를 우선할 수 있고, 언어 생성은 [BLEU](https://medium.com/@priyankads/evaluation-metrics-in-natural-language-processing-bleu-dc3cfa8faaa5) 또는 [ROUGE](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) 점수가 유용할 수 있습니다.

**Community and Resources:** 커뮤니티 지원이 탄탄하고 구현 리소스가 풍부한 모델을 선택하십시오. 명확한 파인튜닝 가이드라인과 신뢰할 수 있는 사전 학습 체크포인트를 제공하는 것을 우선하십시오.

### Data Acquisition and Processing

데이터셋의 품질과 다양성은 파인튜닝된 모델 성능에 큰 영향을 미칩니다. 다음 핵심 사항을 고려하십시오:

**Types of Data Needed:** 필요한 데이터 유형은 작업과 모델의 사전 학습 방식에 따라 달라집니다. NLP 작업에는 보통 책, 기사, 소셜 미디어, 전사본 등에서 얻은 텍스트가 필요합니다. 수집 방법에는 Web스크레이핑, 설문조사, 또는 플랫폼 API가 포함됩니다. 예를 들어, 다양한 최신 데이터를 대량으로 수집할 때 [web scraping with AI](https://brightdata.co.kr/blog/web-data/ai-web-scraping)는 유용합니다.

**Data Cleaning and Annotation:** 정제는 관련 없는 콘텐츠 제거, 누락 정보 처리, 형식 표준화를 포함합니다. 어노테이션은 모델 학습을 위해 데이터에 라벨을 붙이는 것을 의미합니다. [Bright Data](/)와 같은 도구는 이러한 필수 프로세스를 간소화할 수 있습니다.

**Incorporating Diverse Datasets:** 다양하고 대표성 있는 데이터셋는 모델이 여러 관점에서 학습하도록 하여, 더 일반화되고 신뢰할 수 있는 예측을 생성합니다. 예를 들어 영화 리뷰용 감성 분석 모델을 파인튜닝할 때는 실제 분포를 반영하도록 다양한 영화 유형, 장르, 감성 수준의 피드백을 포함하십시오.

### Configuring Your Training Environment

선택한 AI 모델과 프레임워크에 맞는 하드웨어 및 소프트웨어를 갖추어야 합니다. Large Language Models는 일반적으로 상당한 연산 능력을 필요로 하며, 보통 GPU로 제공합니다.

TensorFlow 또는 PyTorch 같은 프레임워크는 AI 모델 학습의 표준 선택지입니다. 관련 라이브러리와 의존성을 설치하면 워크플로에 원활히 통합할 수 있습니다. 예를 들어 OpenAI에서 개발된 모델을 파인튜닝할 때는 [OpenAI API](https://openai.com/index/openai-api/)가 필요할 수 있습니다.

## Step-by-Step Fine-Tuning

파인튜닝의 기본을 다룬 후, 이제 실용적인 자연어 처리 적용 사례를 살펴보겠습니다.

[OpenAI API for fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)을 사용하여 사전 학습된 모델을 파인튜닝하겠습니다. 현재 파인튜닝은 gpt-3.5-turbo-0125(권장), gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, babbage-002, davinci-002, 그리고 실험적 gpt-4-0613 같은 모델에서 동작합니다. GPT-4 파인튜닝은 여전히 실험 단계이며, 자격이 있는 사용자는 [fine-tuning interface](https://platform.openai.com/finetune)를 통해 액세스를 요청할 수 있습니다.

### Preparing Your Dataset

연구 [indicates](https://arxiv.org/abs/2304.03439)에 따르면 GPT-3.5는 분석적 추론에서 한계가 있습니다. 2022년에 공개된 Law School Admission Test 분석적 추론 문제(AR-LSAT)를 사용하여 이 영역에서 `gpt-3.5-turbo`를 강화해 보겠습니다. 이 데이터셋는 [publicly available](https://github.com/csitfun/LogiEval/blob/main/Data/ar_lsat.jsonl)합니다.

파인튜닝된 모델의 품질은 학습 데이터에 의해 직접 좌우됩니다. 각 데이터셋 예시는 OpenAI의 [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) 대화 형식을 따라야 하며, role, content, 그리고 선택적 name 필드를 포함하는 메시지 리스트를 [JSONL](https://jsonlines.org/) 파일로 저장해야 합니다.

`gpt-3.5-turbo` 파인튜닝에 필요한 형식은 다음과 같습니다:

```json
{"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
```

이 구조는 `system`, `user`, `assistant` 역할 간의 대화를 구성하는 메시지 리스트를 포함합니다. `system` 역할의 content는 파인튜닝된 시스템의 동작을 정의합니다.

다음은 AR-LSAT 데이터셋에서 형식화된 예시입니다:

![AR-LSAT dataset example](https://github.com/bright-kr/training-ai-models/blob/main/images/AR-LSAT-dataset-example.png)

중요한 데이터셋 고려 사항은 다음과 같습니다:

- OpenAI는 파인튜닝에 최소 10개 예시를 요구하며, `gpt-3.5-turbo`에는 50~100개 예시를 권장합니다. 최적 수치는 사용 사례에 따라 달라집니다. 하이퍼파라미터 조정을 위해 학습 파일과 함께 검증 파일을 만들 수 있습니다.
- 파인튜닝 및 파인튜닝된 모델 사용에는 베이스 모델에 따라 달라지는 토큰 기반 요금이 부과됩니다. 자세한 내용은 [OpenAI's pricing page](https://openai.com/api/pricing)에서 확인할 수 있습니다.
- 토큰 제한은 선택한 모델에 따라 다릅니다. gpt-3.5-turbo-0125의 경우 최대 컨텍스트 길이 16,385로 인해 각 학습 예시는 16,385 토큰으로 제한되며, 더 긴 예시는 잘립니다. OpenAI의 [counting tokens notebook](https://cookbook.openai.com/examples/How_to_count_tokens_with_tiktoken.ipynb)을 사용하여 토큰 수를 계산하십시오.
- OpenAI는 잠재적 오류를 식별하고 학습/검증 파일 형식을 검증하기 위한 [Python script](https://cookbook.openai.com/examples/chat_finetuning_data_prep)를 제공합니다.

### Setting Up API Access

OpenAI 모델을 파인튜닝하려면 충분한 크레딧 잔액이 있는 [OpenAI developer account](https://platform.openai.com/docs/overview)가 필요합니다.

다음 단계에 따라 API 액세스를 설정하십시오:

1. [OpenAI website](https://platform.openai.com/overview)에서 계정을 생성합니다.

2. 'Settings' 아래의 'Billing' 섹션에서 계정에 크레딧을 추가하여 파인튜닝을 활성화합니다.

![Billing settings on OpenAI](https://github.com/bright-kr/training-ai-models/blob/main/images/Billing-settings-on-OpenAI-1-1024x562.png)

3. 왼쪽 상단 모서리의 프로필 아이콘을 클릭하고 "API Keys"를 선택하여 키 생성 페이지로 이동합니다.

![Accessing API keys in OpenAI's settings](https://github.com/bright-kr/training-ai-models/blob/main/images/Accessing-API-keys-in-OpenAIs-settings-1-1024x396.png)

4. 설명적인 이름을 입력하여 새 secret key를 생성합니다.

![Generating a new API key on OpenAI](https://github.com/bright-kr/training-ai-models/blob/main/images/Generating-a-new-API-key-on-OpenAI.png)

5. 파인튜닝 기능을 위해 OpenAI Python 라이브러리를 설치합니다.

```sh
pip install openai
```

6. 통신을 설정하기 위해 API key를 환경 변수로 구성합니다.

```python
import os
from openai import OpenAI

# Set the OPENAI_API_KEY environment variable
os.environ['OPENAI_API_KEY'] = 'The key generated in step 4'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
```

### Uploading Training Materials

데이터를 검증한 후, 파인튜닝 작업을 위해 [Files API](https://platform.openai.com/docs/api-reference/files/create)를 사용하여 파일을 업로드합니다.

```python
training_file_id = client.files.create(
  file=open(training_file_name, "rb"),
  purpose="fine-tune"
)
validation_file_id = client.files.create(
  file=open(validation_file_name, "rb"),
  purpose="fine-tune"
)
print(f"Training File ID: {training_file_id}")
print(f"Validation File ID: {validation_file_id}")
```

성공적으로 실행되면 학습 및 검증 데이터셋 모두에 대해 고유 식별자를 받게 됩니다.

![unique identifiers example](https://github.com/bright-kr/training-ai-models/blob/main/images/unique-identifiers-example-1024x87.png)

### Initiating a Fine-Tuning Session

파일을 업로드한 후 [user interface](https://platform.openai.com/finetune)를 통해 또는 프로그래밍 방식으로 파인튜닝 작업을 생성합니다.

다음은 OpenAI SDK로 파인튜닝 작업을 시작하는 방법입니다:

```python
response = client.fine_tuning.jobs.create(
  training_file=training_file_id.id,
  validation_file=validation_file_id.id,
  model="gpt-3.5-turbo",
  hyperparameters={
    "n_epochs": 10,
    "batch_size": 3,
    "learning_rate_multiplier": 0.3
  }
)
job_id = response.id
status = response.status

print(f'Fine-tunning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")
```

- `model`: 파인튜닝할 모델을 지정합니다(`gpt-3.5-turbo`, `babbage-002`, `davinci-002`, 또는 기존 파인튜닝된 모델).
- `training_file` 및 `validation_file`: 업로드 중 반환된 파일 식별자입니다.
- `n_epochs`, `batch_size`, `learning_rate_multiplier`: 사용자 지정 가능한 학습 파라メータ입니다.

추가 파인튜닝 옵션은 [API documentation](https://platform.openai.com/docs/api-reference/fine-tuning/create)을 참조하십시오.

이 코드는 작업 ID `ftjob-0EVPunnseZ6Xnd0oGcnWBZA7`에 대한 정보를 생성합니다:

![Example of the information generated by the code above](https://github.com/bright-kr/training-ai-models/blob/main/images/Example-of-the-information-generated-by-the-code-above.png)

파인튜닝 작업은 다른 작업 뒤에 대기열로 들어가므로 완료까지 상당한 시간이 걸릴 수 있습니다. 학습 시간은 모델 복잡도와 데이터셋 크기에 따라 수분에서 수시간까지 다양합니다.

완료되면 OpenAI로부터 이메일 확인을 받게 됩니다.

파인튜닝 인터페이스를 통해 작업 상태를 모니터링하십시오.

### Evaluating Model Performance

OpenAI는 학습 중 다음과 같은 핵심 지표를 계산합니다:

- Training loss
- Training token accuracy
- Validation loss
- Validation token accuracy

검증 지표는 두 가지 방식으로 계산됩니다. 각 스텝에서의 소규모 데이터 배치와, 각 epoch 이후 전체 검증 세트에 대한 계산입니다. 전체 검증 지표는 가장 정확한 성능 지표를 제공하며 원활한 학습 진행을 검증합니다(손실은 감소하고 토큰 정확도는 증가해야 합니다).

파인튜닝이 활성 상태일 때 다음을 통해 지표를 확인할 수 있습니다:

**1. The user interface:**

![The fine tuning UI](https://github.com/bright-kr/training-ai-models/blob/main/images/The-fine-tuning-UI.png)

**2. The API:**

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],)
jobid = 'jobid you want to monitor'

print(f"Streaming events for the fine-tuning job: {jobid}")

# signal.signal(signal.SIGINT, signal_handler)

events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=jobid)
try:
    for event in events:
        print(
            f'{event.data}'
        )
except Exception:
    print("Stream interrupted (client disconnected).")
```

이 코드는 스텝 번호, 학습 및 검증 loss 값, 총 스텝 수, 학습 및 검증 세트 모두의 mean token accuracy를 포함하는 스트리밍 이벤트를 출력합니다:

```
Streaming events for the fine-tuning job: ftjob-0EVPunnseZ6Xnd0oGcnWBZA7
{'step': 67, 'train_loss': 0.30375099182128906, 'valid_loss': 0.49169286092122394, 'total_steps': 67, 'train_mean_token_accuracy': 0.8333333134651184, 'valid_mean_token_accuracy': 0.8888888888888888}
```

### Optimizing Results

파인튜닝 결과가 기대에 미치지 못한다면, 다음 개선 전략을 고려하십시오:

**1. Dataset Refinement:**

- 모델의 특정 약점을 다루는 예시를 추가하고, 응답 분포가 기대 패턴과 일치하도록 하십시오.
- 재현 가능한 데이터 문제를 확인하고, 예시에 올바른 응답을 위해 필요한 모든 정보가 포함되어 있는지 검증하십시오.
- 다수 기여자가 참여한 데이터 전반의 일관성을 유지하고, 모든 예시에서 형식을 표준화하여 추론 시 기대와 일치시키십시오.
- 일반적으로 고품질 데이터는 저품질 정보의 대량보다 더 나은 성능을 제공합니다.

**2. Parameter Adjustments:**

- OpenAI는 세 가지 핵심 하이퍼파라미터(epochs, learning rate multiplier, batch size)를 사용자 지정할 수 있도록 합니다.
- 데이터셋 특성에 따라 내장 함수가 선택한 기본값으로 시작한 뒤, 필요에 따라 조정하십시오.
- 모델이 학습 패턴을 충분히 따르지 않으면 epoch 수를 늘리십시오.
- 모델 응답의 다양성이 부족하면 epoch를 1~2 줄이십시오.
- 수렴 문제가 발생하면 learning rate multiplier를 늘리십시오.

### Working with a Checkpointed Model

OpenAI는 현재 각 파인튜닝 작업에서 마지막 3개 epoch 체크포인트에 대한 액세스를 제공합니다. 이 체크포인트는 추론 및 추가 파인튜닝에 사용할 수 있는 완전한 모델입니다.

체크포인트에 액세스하려면 작업 완료를 기다린 뒤, 파인튜닝 작업 ID를 사용하여 [query the checkpoints endpoint](https://platform.openai.com/docs/api-reference/fine-tuning/list-checkpoints)로 조회하십시오. 각 체크포인트에는 모델 체크포인트 이름을 담고 있는 `fine_tuned_model_checkpoint` 필드가 포함됩니다. 사용자 인터페이스를 통해서도 체크포인트 이름을 조회할 수 있습니다.

[**openai.chat.completions.create()**](https://platform.openai.com/docs/api-reference/chat) 함수로 프롬프트와 모델 이름을 사용해 쿼리를 제출하여 체크포인트 모델 성능을 테스트합니다:

```python
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::9PWZuZo5",
  messages=[
    {"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D', if the fifth option is right, you generate the answer 'E'. Read the question and options thoroughly and select the correct answer from the four answer labels. Read the passage thoroughly to ensure you know what the passage entails"},
    {"role": "user", "content": "Passage: For the school paper, five students\u2014Jiang, Kramer, Lopez, Megregian, and O'Neill\u2014each review one or more of exactly three plays: Sunset, Tamerlane, and Undulation, but do not review any other plays. The following conditions must apply: Kramer and Lopez each review fewer of the plays than Megregian. Neither Lopez nor Megregian reviews any play Jiang reviews. Kramer and O'Neill both review Tamerlane. Exactly two of the students review exactly the same play or plays as each other.Question: Which one of the following could be an accurate and complete list of the students who review only Sunset?\nA. Lopez\nB. O'Neill\nC. Jiang, Lopez\nD. Kramer, O'Neill\nE. Lopez, Megregian\nAnswer:"}
  ]
)
print(completion.choices[0].message)
```

답변 딕셔너리의 응답은 다음과 같습니다:

![the result from the answer dictionary](https://github.com/bright-kr/training-ai-models/blob/main/images/the-result-from-the-answer-dictionary.png)

또한 [OpenAI's playground](https://platform.openai.com/playground/)에서 파인튜닝된 모델을 다른 모델과 비교할 수 있습니다:

![Example of the fine tuned model vs other models in the playground](https://github.com/bright-kr/training-ai-models/blob/main/images/Example-of-the-fine-tuned-model-vs-other-models-in-the-playground-1-1024x776.png)

## Strategies for Success

효과적인 파인튜닝을 위해 다음 권장 사항을 고려하십시오:

**Data Quality:** 과적합을 방지하기 위해 특화 데이터가 깨끗하고, 다양하며, 대표성을 갖추도록 하십시오. 과적합은 모델이 학습 데이터에서는 잘 동작하지만 새로운 입력에는 성능이 저하되는 현상입니다.

**Parameter Selection:** 느린 수렴 또는 최적 이하의 결과를 피하기 위해 적절한 하이퍼파라미터를 선택하십시오. 복잡하고 시간이 많이 들 수 있지만, 효과적인 학습을 위해 필수적인 단계입니다.

**Resource Planning:** 대규모 모델 파인튜닝은 상당한 연산 능력과 시간 할당을 요구한다는 점을 인지하십시오.

### Common Challenges

**Balancing Complexity:** 과적합(과도한 분산)과 과소적합(과도한 편향)을 방지하기 위해 모델 복잡도와 학습 시간 사이에서 적절한 균형을 찾으십시오.

**Knowledge Retention:** 파인튜닝 중 모델이 기존에 습득한 일반 지식을 잃을 수 있습니다. 다양한 작업 전반에서 성능을 정기적으로 테스트하여 이 문제를 완화하십시오.

**Domain Adaptation:** 파인튜닝 데이터가 사전 학습 데이터와 크게 다르면 도메인 시프트 문제가 발생할 수 있습니다. 이러한 격차를 해결하기 위해 도메인 적응 기법을 적용하십시오.

### Model Preservation

학습 후에는 모델 파라미터와 옵티마이저 상태를 포함하여 모델의 전체 상태를 저장하여 향후 사용할 수 있도록 하십시오. 이를 통해 나중에 동일한 지점에서 학습을 재개할 수 있습니다.

### Ethical Implications

**Bias Concerns:** 사전 학습된 모델에는 내재적 편향이 포함될 수 있으며, 파인튜닝이 이를 증폭시킬 수 있습니다. 편향 없는 예측이 중요한 경우 공정성 테스트를 거친 사전 학습 모델을 선택하십시오.

**Output Verification:** 파인튜닝된 모델은 그럴듯하지만 잘못된 정보를 생성할 수 있습니다. 이러한 경우를 처리하기 위한 강건한 검증 시스템을 구현하십시오.

**Performance Degradation:** 환경 또는 데이터 분포 변화로 인해 시간이 지남에 따라 모델 성능이 저하될 수 있습니다. 성능을 정기적으로 모니터링하고 필요에 따라 파인튜닝을 갱신하십시오.

### Cutting-Edge Techniques

LLM을 위한 고급 파인튜닝 접근법으로는 Low Ranking Adaptation(LoRA) 및 Quantized LoRA(QLoRA)가 있으며, 성능을 유지하면서 연산 및 비용 요구사항을 줄입니다. Parameter Efficient Fine Tuning(PEFT)은 최소한의 파라미터 조정으로 모델을 효율적으로 적응시킵니다. DeepSpeed와 ZeRO는 대규모 학습을 위한 메모리 사용을 최적화합니다. 이러한 기법은 과적합, 지식 손실, 도메인 적응을 포함한 과제를 해결하여 LLM 파인튜닝의 효율성과 효과를 향상시킵니다.

파인튜닝을 넘어, 전이 학습과 강화 학습 같은 다른 고급 학습 방법도 살펴보십시오. 전이 학습은 [one domain to related problems](https://brightdata.co.kr/blog/web-data/data-pitfalls-when-developing-ai-models)으로부터 지식을 적용하며, 강화 학습은 에이전트가 환경과 상호작용하고 보상을 최대화함으로써 최적 의사결정을 학습하도록 합니다.

AI 모델 학습을 더 깊이 탐구하기 위해 다음 유용한 리소스를 고려하십시오:

- [Attention is all you need by Ashish Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [The book "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)
- [The book "Speech and Language Processing" by Daniel Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)
- [Different ways of training LLMs](https://towardsdatascience.com/different-ways-of-training-llms-c57885f388ed)
- [Mastering LLM Techniques: Training](https://developer.nvidia.com/blog/mastering-llm-techniques-training/)
- [NLP course by Hugging Face](https://huggingface.co/learn/nlp-course/chapter1/1)

## Final Thoughts

효과적인 AI 모델을 개발하려면 상당한 양의 고품질 데이터가 필요합니다. 문제 정의, 모델 선택, 반복적 개선이 중요하지만, 진정한 차별화 요소는 데이터의 품질과 규모에 있습니다.

직접 Web스크레이핑 도구를 만들고 유지보수하는 대신, Bright Data 플랫폼의 사전 구축 또는 [custom datasets](https://brightdata.co.kr/products/datasets)를 통해 데이터 수집을 단순화하십시오.

지금 등록하고 무료 체험을 시작하십시오!