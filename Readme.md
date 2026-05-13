# PSA CO2 Capture Codebase

이 저장소는 전처리 공정, 가드베드, MOF 기반 PSA/VSA 분리, 그리고 운전 최적화까지 이어지는 CO2 포집 연구용 코드와 산출물을 모아둔 프로젝트입니다.

## 폴더별 구성

### `preprocessing/`
- 전처리(pretreatment) 구간의 합성 데이터와 모델링 노트북이 있습니다.
- `pre.ipynb`
  - 전처리 입력 조건을 바탕으로 데이터셋을 만드는 노트북입니다.
- `model.ipynb`
  - 전처리 데이터로 RNN 기반 surrogate 모델을 학습합니다.
- `test.ipynb`
  - 저장된 surrogate 모델과 번들을 불러와 성능을 검증합니다.
- 산출물
  - `test.csv`: 전처리 기준 데이터셋
  - `guard_bed_rnn_models/*.keras`: GRU/LSTM surrogate 모델
  - `guard_bed_rnn_bundle.joblib`: 스케일러 등 번들 파일

### `guard_bed/`
- 가드베드 출구 조건을 생성하거나 예측 모델을 학습하는 영역입니다.
- `simulate_guard_bed.py`
  - 전처리 기준 데이터(`preprocessing/test.csv`)를 참고해 가드베드 합성 데이터를 생성하는 스크립트입니다.
  - 유량, 온도, 압력, 습도, 입자 크기, 공극률, 오염물 농도 등을 샘플링하고, Ergun 식을 보정한 압력강하와 수분/불순물 제거 효율을 계산해 CSV를 만듭니다.
- `model.ipynb`
  - 가드베드 합성 데이터로 회귀 모델들을 비교하고 최적 모델을 저장하는 노트북입니다.
- 산출물
  - `guard_bed_synthetic_dataset.csv`: 생성된 가드베드 데이터
  - `artifacts/best_model_xgboost.joblib`: 선택된 최적 모델
  - `artifacts/model_comparison_metrics.csv`: 모델 비교 결과

### `mof/Synthetic_Data_Generation/`
- Mg-MOF-74 기반 PSA/VSA 사이클을 이용해 학습용 합성 데이터를 만드는 코드입니다.
- `pas.py`
  - 일반 전처리된 배가스 조건을 가정한 PSA 시뮬레이터입니다.
  - `MgMOF74PSA` 클래스가 흡착/감압/탈착 사이클을 계산하고, `generate_synthetic_dataset()`이 LHS 샘플링으로 유효한 학습 데이터를 만듭니다.
- `pas_lime.py`
  - 석회 소성로(lime kiln) 배가스를 대상으로 같은 구조의 데이터를 생성합니다.
  - 원가스 조성, 냉각, 제습, 탈황, 탈질, 집진 효율까지 함께 반영합니다.

### `mof/model/`
- MOF PSA 성능 surrogate 모델 학습 결과가 있습니다.
- `model_dnn.ipynb`
  - 합성 데이터셋을 이용해 `CO2_purity`, `CO2_recovery` 예측용 DNN 모델을 학습합니다.
- 산출물
  - `CO2_purity_dnn.keras`
  - `CO2_recovery_dnn.keras`

### `mof/optimization/`
- 학습된 DNN surrogate를 이용해 운전 조건을 경제성 기준으로 최적화합니다.
- `optimize_policy.py`
  - 입력 고정 조건(`co2_mol_frac`, `h2o_ppmv`, `sox_ppmv`, `nox_ppmv`, `dust_mg_Nm3`)이 주어졌을 때,
    `tADS`, `PL`, `v0`, `t_press`, `t_depress`를 탐색해 총 비용(`total_cost_per_tco2`)이 최소가 되는 운전점을 찾습니다.
  - 전력비, 미회수 CO2 비용, 흡착제 비용, purity/recovery penalty, 진공 실현성 penalty 등을 함께 계산합니다.
- `experiment_helpers.py`
  - 시간대별 전기요금, 유량 변동, 외기온 등을 포함한 동적 시나리오를 생성하고,
    동적 재최적화와 정적 운전 전략을 비교하는 실험용 함수들을 제공합니다.
- 노트북
  - `optimization.ipynb`: 최적화 실행/시각화
  - `experiment.ipynb`: 시나리오 기반 비교 실험

### `mof/exp/`
- 전이학습 실험용 노트북과 결과물이 있습니다.
- `transfer_learning_lime.ipynb`
  - 기존 MOF surrogate를 바탕으로 lime kiln 데이터셋에 대해 target-only MLP와 transfer model을 비교합니다.
- `artifacts_lime_transfer_improved/*.keras`
  - fold별 저장 모델

### `mof/dataset/`
- 학습/검증/실험용 CSV가 저장되어 있습니다.
- `data.csv`, `test.csv`
  - 일반 MOF PSA 데이터
- `lime_seed42.csv`, `test_lime.csv`
  - lime kiln 전이학습용 데이터
- `seed10.csv`~`seed90.csv`
  - 시드별 생성 데이터

## 코드 흐름

1. `preprocessing/`에서 전처리 조건 데이터와 surrogate를 만듭니다.
2. `guard_bed/simulate_guard_bed.py`로 가드베드 출구 조건 데이터를 생성합니다.
3. `mof/Synthetic_Data_Generation/`에서 PSA 합성 데이터셋을 생성합니다.
4. `mof/model/model_dnn.ipynb`로 CO2 purity/recovery surrogate를 학습합니다.
5. `mof/optimization/optimize_policy.py`와 관련 노트북으로 경제성 기반 운전 최적화를 수행합니다.
6. 필요하면 `mof/exp/transfer_learning_lime.ipynb`로 lime kiln 데이터에 대한 전이학습 실험을 수행합니다.

## 바로 실행 가능한 대표 스크립트

### 1. 가드베드 데이터 생성
```bash
python guard_bed/simulate_guard_bed.py --n-samples 3000 --seed 42
```

### 2. 일반 MOF PSA 합성 데이터 생성
```bash
python mof/Synthetic_Data_Generation/pas.py
```

### 3. Lime kiln PSA 합성 데이터 생성
```bash
python mof/Synthetic_Data_Generation/pas_lime.py
```

### 4. 최적화 실행
`mof/optimization/optimize_policy.py`는 기본적으로 아래 파일들을 사용합니다.
- 학습 데이터: `mof/model/output.csv` 또는 대체로 `mof/dataset/data.csv`
- 모델: `mof/model/CO2_purity_dnn.keras`, `mof/model/CO2_recovery_dnn.keras`

예시:
```bash
cd mof/optimization
python optimize_policy.py --train-csv ../dataset/data.csv
```

## 참고

- 노트북 기반 작업이 많아서, 재현 목적이라면 `.ipynb`와 저장된 `.keras`/`.joblib` 파일을 함께 보는 것이 좋습니다.
- 일부 모델 번들은 저장 당시와 현재 `scikit-learn` 버전이 다르면 경고가 날 수 있습니다.
- 현재 저장소에는 이미 학습 결과물(`.keras`, `.joblib`, `.csv`)이 포함되어 있어서, 학습을 다시 하지 않고도 추론/최적화 실험부터 시작할 수 있습니다.
