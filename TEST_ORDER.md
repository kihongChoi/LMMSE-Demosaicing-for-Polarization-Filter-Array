# 테스트 순서 가이드

## 의존성 구조
```
blockproc (독립)
  ├─> d_matrix
  └─> function_lmmse_demosaicing

filter_images (독립)
  └─> mosaicking

load_dataset (독립)
  └─> script_lmmse_retraining

mosaicking
  └─> script_lmmse_retraining

d_matrix
  └─> script_lmmse_retraining

function_lmmse_demosaicing
  └─> script_lmmse_demosaicing
```

## 권장 테스트 순서

### 1단계: 기본 유틸리티 함수 테스트 (의존성 없음)

#### 1-1. `blockproc` 함수 테스트
```python
# 테스트 파일: test_blockproc.py
from function.blockproc import blockproc
import numpy as np

# 간단한 테스트
img = np.random.rand(100, 100)
def fun(x):
    return x * 2

result = blockproc(img, (10, 10), fun, border_size=(2, 2))
print(f"Input shape: {img.shape}, Output shape: {result.shape}")
```

**확인 사항:**
- 입력/출력 shape 확인
- border_size 동작 확인
- 병렬 처리 동작 확인

---

### 2단계: 데이터 로드 및 필터링 함수

#### 2-1. `load_dataset` 함수 테스트
```python
# 테스트 파일: test_load_dataset.py
from function.load_dataset import load_dataset
import os

folder_path = 'Data/Dataset'
dataset = load_dataset(save=False, folder=folder_path, nbr_of_img=1, mosaic='pfa')
print(f"Dataset loaded: {len(dataset)} images")
print(f"Image shape: {dataset[0][1].shape}")
```

**확인 사항:**
- 이미지 파일이 올바르게 로드되는지
- 데이터 타입이 float64인지
- PFA/CPFA 모드별 동작 확인

#### 2-2. `filter_images` 함수 테스트
```python
# 테스트 파일: test_filter.py
from function.filter import filter_images
import numpy as np

# 테스트 이미지 생성 (4채널: 0°, 45°, 90°, 135°)
test_img = np.random.rand(100, 100, 4)
filtered = filter_images(test_img, mosaic='pfa')
print(f"Input shape: {test_img.shape}, Output shape: {filtered.shape}")
```

**확인 사항:**
- PFA 패턴이 올바르게 적용되는지
- 출력 shape 확인

---

### 3단계: 모자이킹 파이프라인

#### 3-1. `mosaicking` 함수 테스트
```python
# 테스트 파일: test_mosaicking.py
from function.load_dataset import load_dataset
from function.mosaicking import mosaicking

# 데이터셋 로드
folder_path = 'Data/Dataset'
dataset = load_dataset(save=False, folder=folder_path, nbr_of_img=1, mosaic='pfa')

# 모자이킹
mos_dataset = mosaicking(dataset, save=False, folder_path='', mosaic='pfa')
print(f"Mosaicked image shape: {mos_dataset[0][1].shape}")
```

**확인 사항:**
- 모자이킹된 이미지 shape 확인
- 필터 패턴이 올바르게 적용되는지

---

### 4단계: D 행렬 계산 (재학습)

#### 4-1. `d_matrix` 함수 테스트 (작은 데이터셋으로)
```python
# 테스트 파일: test_d_matrix.py
from function.load_dataset import load_dataset
from function.mosaicking import mosaicking
from function.d_matrix import d_matrix

# 작은 데이터셋으로 테스트
folder_path = 'Data/Dataset'
full_dataset = load_dataset(save=False, folder=folder_path, nbr_of_img=1, mosaic='pfa')
mos_dataset = mosaicking(full_dataset, save=False, folder_path='', mosaic='pfa')

# D 행렬 계산 (시간이 오래 걸릴 수 있음)
demos_dataset, y1, y, D = d_matrix(
    full_dataset, mos_dataset, folder_path='', mosaic='pfa'
)
print(f"D matrix shape: {D.shape}")
```

**확인 사항:**
- D 행렬이 올바른 shape인지
- 계산 과정에서 에러가 없는지
- 메모리 사용량 확인

---

### 5단계: 디모자이킹 함수

#### 5-1. `function_lmmse_demosaicing` 함수 테스트
```python
# 테스트 파일: test_demosaicing.py
import numpy as np
import scipy.io as sio
from function_lmmse_demosaicing import function_lmmse_demosaicing
from PIL import Image

# D 행렬 로드
d_matrix_data = sio.loadmat('Data/D_Matrix.mat')
D = d_matrix_data['D']

# 모자이킹된 이미지 로드
mos_img = np.array(Image.open('Data/im.tif'), dtype=np.float64) / 255.0
if len(mos_img.shape) == 3:
    mos_img = np.mean(mos_img, axis=2)  # Grayscale로 변환

# 디모자이킹
demos_img = function_lmmse_demosaicing(mos_img, D)
print(f"Demosaiced image shape: {demos_img.shape}")
```

**확인 사항:**
- 출력 shape이 (rows, cols, channels, 4)인지
- 4개의 편광 각도 이미지가 생성되는지

---

### 6단계: 전체 스크립트 테스트

#### 6-1. `script_lmmse_demosaicing.py` 테스트
```bash
# 직접 실행
python script_lmmse_demosaicing.py
```

**확인 사항:**
- D_Matrix.mat 파일이 존재하는지
- Data/im.tif 파일이 존재하는지
- 결과 이미지가 저장되는지
- 시각화가 올바르게 표시되는지

#### 6-2. `script_lmmse_retraining.py` 테스트
```bash
# 직접 실행 (시간이 오래 걸릴 수 있음)
python script_lmmse_retraining.py
```

**확인 사항:**
- Data/Dataset 폴더에 이미지가 있는지
- D_Matrix_retrained.mat가 생성되는지
- 중간 결과 파일들이 저장되는지

---

## 빠른 검증 순서 (최소 테스트)

시간이 부족한 경우 다음 순서로 빠르게 검증:

1. **blockproc** - 기본 동작 확인
2. **load_dataset** - 데이터 로드 확인
3. **filter_images** - 필터 동작 확인
4. **function_lmmse_demosaicing** - 디모자이킹 동작 확인 (기존 D 행렬 사용)
5. **script_lmmse_demosaicing** - 전체 파이프라인 확인

---

## 주의사항

1. **데이터 준비:**
   - `Data/D_Matrix.mat` 파일이 있어야 디모자이킹 테스트 가능
   - `Data/Dataset/` 폴더에 `0_1.png`, `45_1.png`, `90_1.png`, `135_1.png` 등이 있어야 재학습 테스트 가능

2. **메모리:**
   - `d_matrix` 함수는 큰 메모리를 사용할 수 있음
   - 작은 이미지로 먼저 테스트 권장

3. **실행 시간:**
   - `d_matrix` 계산은 시간이 오래 걸릴 수 있음
   - `blockproc`의 병렬 처리가 제대로 동작하는지 확인

4. **에러 처리:**
   - 각 단계에서 발생하는 에러를 기록하고 수정
   - 한 단계가 완전히 통과한 후 다음 단계로 진행
