# Surrogate-Based PSA/VSA Process Modeling & Optimization

Physics-based simulation, surrogate modeling, and operating-condition optimization for an Mg-MOF-74 PSA/VSA process.

## Overview

본 프로젝트는 **PSA/VSA 공정의 반복적인 물리 시뮬레이션 비용을 줄이고, 다양한 운전 및 유입 조건에서 효율적인 운전점을 탐색하기 위한 surrogate-based process optimization 프로젝트**입니다.

Mg-MOF-74 기반 PSA/VSA 공정을 대상으로 물리 기반 simulation을 수행하여 학습 데이터를 생성하고, 이를 통해 CO₂ purity와 recovery를 예측하는 surrogate model을 구축했습니다.

학습된 surrogate model은 이후 공정 운전변수 최적화에 사용되며, 전력비와 공정 성능을 함께 고려하여 운전 조건에 따른 techno-economic performance를 평가합니다.

또한 서로 다른 배가스 조건에 대한 적용 가능성을 확인하기 위해 lime kiln 조건을 대상으로 transfer learning 실험을 수행했습니다.

---

## Project Pipeline

```text
Pretreatment Conditions
        ↓
Guard Bed Modeling
        ↓
Mg-MOF-74 PSA/VSA Simulation
        ↓
Synthetic Dataset Generation
        ↓
Surrogate Modeling
 ├─ CO₂ Purity
 └─ CO₂ Recovery
        ↓
Operating Condition Optimization
        ↓
Dynamic Scenario Evaluation
        ↓
Transfer Learning to Lime Kiln Conditions
```

---

## Key Components

### 1. Pretreatment Modeling

PSA/VSA 공정에 유입되기 전 배가스 조건을 모델링합니다.

전처리 조건으로부터 downstream 공정에서 사용할 유입 특성을 생성하고, RNN 기반 surrogate model을 이용한 예측 실험을 수행했습니다.

주요 모델:

* GRU
* LSTM

---

### 2. Guard Bed Modeling

수분과 불순물이 PSA/VSA 공정에 미치는 영향을 반영하기 위해 guard bed 영역을 별도로 모델링했습니다.

유량, 온도, 압력, 습도, 입자 크기, 공극률 및 불순물 농도 등을 기반으로 synthetic dataset을 생성했습니다.

압력 강하는 Ergun equation을 기반으로 계산하며, 수분 및 불순물 제거 성능을 함께 모델링합니다.

여러 regression model을 비교한 결과 XGBoost가 가장 높은 예측 성능을 보였습니다.

| Target          | Model   |     R² |
| --------------- | ------- | -----: |
| H₂O outlet      | XGBoost | 0.9934 |
| Impurity outlet | XGBoost | 0.9939 |

---

### 3. Mg-MOF-74 PSA/VSA Simulation

Mg-MOF-74를 흡착제로 사용하는 PSA/VSA cycle을 모델링합니다.

공정은 다음 cycle step으로 구성됩니다.

```text
Pressurization
      ↓
Adsorption
      ↓
Depressurization
      ↓
Desorption
```

시뮬레이션에서는 단순 CO₂/N₂ 분리뿐만 아니라 전처리 이후 남아 있는 불순물의 영향도 고려합니다.

반영되는 주요 조건:

* CO₂ concentration
* residual H₂O
* SOx
* NOx
* dust

이 조건들은 다음과 같은 effective adsorption parameters에 반영됩니다.

* adsorption capacity
* CO₂ affinity
* mass-transfer coefficient
* axial dispersion
* adsorbent deactivation

Latin Hypercube Sampling을 이용해 다양한 공정 조건을 생성하고 PSA/VSA simulation을 반복 수행하여 surrogate model 학습용 dataset을 구축했습니다.

---

### 4. Surrogate Modeling

PSA/VSA simulation은 하나의 운전 조건을 평가할 때도 반복적인 numerical calculation이 필요합니다.

따라서 공정 최적화 과정에서 simulation을 직접 반복 호출하는 대신, simulation 결과를 학습한 DNN surrogate model을 사용합니다.

예측 대상:

```text
Operating Conditions
        ↓
   DNN Surrogate
        ↓
 ┌─────────────────┐
 │   CO₂ Purity    │
 │   CO₂ Recovery  │
 └─────────────────┘
```

주요 입력 변수:

* adsorption time (`tADS`)
* low pressure (`PL`)
* superficial velocity (`v0`)
* pressurization time (`t_press`)
* depressurization time (`t_depress`)
* feed gas conditions
* pretreatment/deactivation factors

---

## 5. Process Optimization

학습된 surrogate model을 이용하여 PSA/VSA 운전조건을 탐색합니다.

Optimization variables:

```text
tADS
PL
v0
t_press
t_depress
```

초기 탐색에서는 Latin Hypercube Sampling으로 넓은 운전영역을 탐색하고, 이후 우수 candidate 주변에서 반복적인 local search를 수행합니다.

목적함수는 단순히 purity 또는 recovery를 최대화하는 것이 아니라 **CO₂ 처리 단위당 총 비용**을 최소화하도록 구성되어 있습니다.

고려되는 항목:

* vacuum energy
* blower energy
* product compression
* CAPEX proxy
* fixed O&M
* adsorbent cost
* unrecovered CO₂ cost
* purity penalty
* recovery penalty
* operating feasibility penalties

---

## 6. Dynamic Operating Scenario

고정된 운전조건과 시간에 따라 다시 최적화하는 dynamic strategy를 비교했습니다.

전기요금 등이 변하는 simulation scenario에서 다음 전략을 비교합니다.

* Dynamic re-optimization
* Static operation based on average conditions
* Static operation based on initial conditions

저장된 `price_volatile` simulation 결과에서는 dynamic re-optimization이 static strategy보다 낮은 총 운전비용을 기록했습니다.

> 해당 결과는 실제 plant operating data가 아닌 본 프로젝트의 simulation 및 surrogate model에 기반한 결과입니다.

---

## 7. Transfer Learning

기본 PSA/VSA dataset에서 학습한 surrogate model을 다른 배가스 조건에 적용하기 위해 **lime kiln flue gas**를 대상으로 transfer learning을 실험했습니다.

비교 대상:

* Target-only MLP
* Pretrained model + transfer learning

이를 통해 source process에서 학습한 표현을 새로운 operating domain에 재사용할 수 있는지 확인했습니다.

---

## Repository Structure

```text
.
├── preprocessing/
│   ├── pre.ipynb
│   ├── model.ipynb
│   └── test.ipynb
│
├── guard_bed/
│   ├── simulate_guard_bed.py
│   ├── model.ipynb
│   └── artifacts/
│
├── mof/
│   ├── Synthetic_Data_Generation/
│   │   ├── pas.py
│   │   └── pas_lime.py
│   │
│   ├── dataset/
│   │
│   ├── model/
│   │   └── model_dnn.ipynb
│   │
│   ├── optimization/
│   │   ├── optimize_policy.py
│   │   ├── experiment_helpers.py
│   │   ├── optimization.ipynb
│   │   └── experiment.ipynb
│   │
│   └── exp/
│       └── transfer_learning_lime.ipynb
│
└── README.md
```

---

## Tech Stack

**Language**

* Python

**Process / Numerical Simulation**

* NumPy
* SciPy
* pyAPEP

**Machine Learning**

* TensorFlow / Keras
* scikit-learn
* XGBoost

**Data Analysis**

* Pandas
* Matplotlib
* Seaborn

**Optimization**

* Latin Hypercube Sampling
* Surrogate-based search
* Local candidate refinement

---

## Limitations

본 프로젝트는 실제 상용 PSA/VSA plant의 운전 데이터를 기반으로 한 digital twin이 아니라, **물리 모델과 synthetic dataset을 기반으로 구축한 process modeling 및 optimization 연구 프로젝트**입니다.

따라서 실제 산업 공정에 적용하기 위해서는 다음 단계가 추가로 필요합니다.

* Experimental / pilot-scale data validation
* Model parameter calibration
* Adsorbent degradation validation
* Equipment-level CAPEX/OPEX model refinement
* Closed-loop optimization validation

---

## Goal

이 프로젝트의 핵심은 단순히 CO₂ purity/recovery를 예측하는 ML 모델을 만드는 것이 아니라,

**physics-based process simulation → surrogate modeling → process optimization → domain adaptation**

으로 이어지는 하나의 공정 모델링 및 최적화 workflow를 구현하는 것입니다.
