# 실행에 필요한 것들

## 1. 기본 환경

- Python 3.13 계열에서 작업된 흔적이 있습니다.
- 현재 저장소에서 확인된 로컬 환경 예시는 아래와 같습니다.
  - `Python 3.13.3`
  - `numpy 2.3.2`
  - `pandas 2.2.3`
  - `scikit-learn 1.7.0`
  - `tensorflow 2.20.0`
  - `matplotlib 3.10.1`
  - `seaborn 0.13.2`
  - `xgboost 3.2.0`
  - `scipy 1.16.0`
  - `joblib 1.5.1`

## 2. 필수 패키지 설치

최소 실행 기준으로는 아래 패키지가 필요합니다.

```bash
pip install numpy pandas scipy scikit-learn tensorflow matplotlib seaborn xgboost joblib jupyter
```

`mof/Synthetic_Data_Generation/pas.py`와 `pas_lime.py`를 실행하려면 `pyapep`도 필요합니다.

```bash
pip install pyapep
```

## 3. 용도별 필요 패키지

### 가드베드 데이터 생성만 할 때
- 필요 패키지
  - `numpy`
  - `pandas`
- 실행 명령
```bash
python guard_bed/simulate_guard_bed.py --n-samples 3000 --seed 42
```

### 가드베드 모델 학습 노트북(`guard_bed/model.ipynb`)을 돌릴 때
- 추가 패키지
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`
  - `seaborn`
  - `joblib`

### 전처리 노트북(`preprocessing/*.ipynb`)을 돌릴 때
- 추가 패키지
  - `tensorflow`
  - `scikit-learn`
  - `matplotlib`
  - `joblib`
  - `scipy`
  - `jupyter`

### MOF 합성 데이터 생성(`mof/Synthetic_Data_Generation/*.py`)을 돌릴 때
- 추가 패키지
  - `pyapep`
  - `numpy`
  - `pandas`

### MOF DNN 학습(`mof/model/model_dnn.ipynb`)을 돌릴 때
- 추가 패키지
  - `tensorflow`
  - `scikit-learn`
  - `matplotlib`
  - `jupyter`

### 최적화 코드(`mof/optimization/optimize_policy.py`)를 돌릴 때
- 추가 패키지
  - `tensorflow`
  - `scikit-learn`
  - `numpy`
  - `pandas`
- 필요한 파일
  - `mof/model/CO2_purity_dnn.keras`
  - `mof/model/CO2_recovery_dnn.keras`
  - 학습용 CSV (`mof/dataset/data.csv` 등)
- 실행 예시
```bash
cd mof/optimization
python optimize_policy.py --train-csv ../dataset/data.csv
```

## 4. 실행 순서 추천

처음 보는 환경이면 아래 순서가 가장 안전합니다.

1. 패키지 설치
```bash
pip install numpy pandas scipy scikit-learn tensorflow matplotlib seaborn xgboost joblib jupyter pyapep
```

2. 가드베드 데이터 생성 확인
```bash
python guard_bed/simulate_guard_bed.py --n-samples 100 --seed 42
```

3. MOF 합성 데이터 생성 확인
```bash
python mof/Synthetic_Data_Generation/pas.py
python mof/Synthetic_Data_Generation/pas_lime.py
```

4. 최적화 실행 확인
```bash
cd mof/optimization
python optimize_policy.py --train-csv ../dataset/data.csv
```

5. 노트북 확인
```bash
jupyter notebook
```

## 5. 주의할 점

- `preprocessing/test.ipynb`에는 `scikit-learn` 버전 차이 경고가 남아 있습니다. 저장된 `joblib` 번들을 다른 버전에서 열면 경고가 날 수 있습니다.
- TensorFlow 실행 시 `protobuf` 관련 경고가 보일 수 있지만, 현재 저장소 환경에서는 import 자체는 됩니다.
- `mof/Synthetic_Data_Generation/*.py`는 `pyapep.simsep.column`을 직접 사용하므로 `pyapep`가 없으면 실행되지 않습니다.
- 이미 생성된 `.keras`, `.joblib`, `.csv` 파일이 저장소에 포함되어 있어서, 전체 학습을 다시 하지 않고도 일부 실험은 바로 실행할 수 있습니다.
