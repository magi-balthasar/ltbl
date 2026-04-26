# LTBL 개발환경 설정 가이드

**[한국어] | [English](#english)**

---

## 목차

- [공통 전제조건](#공통-전제조건)
- [Phase 1 — 태초의 바다](#phase-1--태초의-바다-현재)
- [Phase 2 — 분산 신경망](#phase-2--분산-신경망-예정)
- [Phase 3 — 절지동물](#phase-3--절지동물-예정)
- [Phase 4–5 — 어류 / 4족 보행](#phase-45--어류--4족-보행-예정)
- [Phase 6–7 — 사회적 학습 / 추상화](#phase-67--사회적-학습--추상화-예정)
- [문제 해결](#문제-해결)

---

## 공통 전제조건

### 운영체제
- macOS 13 Ventura 이상
- Ubuntu 22.04 LTS 이상
- Windows는 WSL2 환경 권장

### Python 버전
```
Python 3.11 이상
```

Python 버전 확인:
```bash
python3 --version
```

### 패키지 관리자: uv

```bash
# macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 설치 확인
uv --version
```

> `pip` 대신 `uv`를 사용한다. 의존성 해석이 빠르고 가상환경 관리가 단순하다.

### 저장소 클론

```bash
git clone https://github.com/magi-balthasar/ltbl.git
cd ltbl
```

---

## Phase 1 — 태초의 바다 (현재)

### 의존성

| 패키지 | 용도 |
|--------|------|
| `numpy` | 2D 화학 농도 그리드, 벡터 연산 |
| `scipy` | 확산 방정식 보조 연산 |
| `ray` | 8개 섬 병렬 실행 |
| `matplotlib` | 시뮬레이션 후 분석 그래프 |
| `pygame` | 실시간 시각화 |

### 설치

```bash
# 1. 가상환경 생성
uv venv

# 2. 활성화
#    macOS / Linux
source .venv/bin/activate
#    Windows (WSL2 제외)
.venv\Scripts\activate

# 3. 의존성 설치
uv pip install -r requirements.txt
```

프롬프트 앞에 `(.venv)` 가 표시되면 활성화 상태다.

### 설치 확인

```bash
python3 -c "
import numpy, scipy, ray, matplotlib, pygame, sqlite3
print('numpy   ', numpy.__version__)
print('scipy   ', scipy.__version__)
print('ray     ', ray.__version__)
print('matplotlib', matplotlib.__version__)
print('pygame  ', pygame.__version__)
print('sqlite3 built-in OK')
print()
print('All Phase 1 dependencies OK')
"
```

### 실행

```bash
# 단일 섬 — 빠른 동작 확인 (Ray 불필요)
python3 main.py --single --steps 200 --report 20

# 8개 섬 병렬 실험
python3 main.py --steps 500 --report 10

# 실시간 시각화 (별도 창)
python3 visualize.py

# 시뮬레이션 후 분석 그래프 생성
python3 main.py --steps 200
python3 analysis.py --show
```

### 하드웨어 권장 사양

| 구분 | 최소 | 권장 |
|------|------|------|
| CPU 코어 | 4 | 8 이상 |
| RAM | 8 GB | 16 GB |
| 디스크 | 1 GB | 5 GB (실험 로그) |

> Ray는 기본적으로 가용 CPU 코어를 모두 사용한다.  
> 코어 수를 제한하려면: `ray.init(num_cpus=4)`

---

## Phase 2 — 분산 신경망 (예정)

히드라, 해파리 수준의 분산 신경망 모델.  
**아직 구현되지 않음 — 의존성은 Phase 2 시작 전 확정 예정.**

### 예상 추가 의존성

| 패키지 | 용도 |
|--------|------|
| `torch` | 신경망 가중치 학습 (분산 퍼셉트론 등가물) |
| `networkx` | 뉴런 간 연결 그래프 모델링 |

```bash
# Phase 2 준비 시 실행 (미리 설치 불필요)
uv pip install torch networkx
```

### 주요 변화
- 에이전트 내부에 고정 크기 벡터 대신 **그래프 구조 신경망** 도입
- 게놈이 신경망 위상(topology)을 인코딩
- Ray 워커당 에이전트 수 증가 → GPU 활용 고려

---

## Phase 3 — 절지동물 (예정)

CPG(Central Pattern Generator) 기반 운동 모델.  
**아직 구현되지 않음.**

### 예상 추가 의존성

| 패키지 | 용도 |
|--------|------|
| `pymunk` | 2D 강체 물리 시뮬레이션 (다리, 관절) |
| `torch` | CPG 파라미터 학습 |

```bash
uv pip install pymunk torch
```

### 주요 변화
- 에이전트가 **물리 몸체(body) + 관절(joint)** 을 가짐
- 화학 그라디언트 → 운동 패턴 CPG 제어 루프
- 시뮬레이션 스텝이 물리 dt에 묶임 → 성능 요구 증가

---

## Phase 4–5 — 어류 / 4족 보행 (예정)

척수-뇌간 구조, 3D 물리 환경.  
**아직 구현되지 않음.**

### 예상 추가 의존성

| 패키지 | 용도 |
|--------|------|
| `mujoco` | 고성능 3D 물리 엔진 |
| `dm-control` | MuJoCo Python 래퍼 |
| `torch` | 정책 학습 |

```bash
uv pip install mujoco dm-control torch
```

MuJoCo는 별도 라이선스 없이 무료로 사용 가능 (2021년 이후).

### 주요 변화
- **2D → 3D** 환경 전환
- 에이전트가 중력, 마찰, 충돌을 받는 실체 몸체를 가짐
- ROS2 연동 실험 시작 가능 (하드웨어 로봇 연결)

### ROS2 (하드웨어 연동, 선택)

```bash
# Ubuntu 22.04 기준 (macOS는 Docker 권장)
sudo apt install ros-humble-desktop

# Python 패키지
uv pip install rclpy
```

---

## Phase 6–7 — 사회적 학습 / 추상화 (예정)

**아직 구현되지 않음.**

### 예상 추가 의존성

| 패키지 | 용도 |
|--------|------|
| `torch` | 언어 모델 수준 추상화 학습 |
| `transformers` | Phase 7 언어 기반 정보 압축 실험 |
| `wandb` | 대규모 실험 추적 |

```bash
uv pip install torch transformers wandb
```

### 주요 변화
- Phase 6: 에이전트 간 **신호 전달(quorum sensing → 언어 전구체)**
- Phase 7: 내부 상태를 **압축 심볼**로 인코딩하여 유전하는 구조

---

## 문제 해결

### `ModuleNotFoundError: No module named 'pygame'`
가상환경이 활성화되지 않은 상태에서 실행한 경우다.
```bash
source .venv/bin/activate   # 이 줄 먼저 실행
python3 visualize.py
```

### `command not found: ray`
동일한 원인. venv 활성화 후 재시도.

### Ray 워커가 모듈을 찾지 못하는 경우
`parallel_engine.py`는 `runtime_env={"working_dir": ...}`로 프로젝트 루트를 워커에 전달한다.  
`ltbl/` 디렉토리 안에서 실행해야 한다.
```bash
cd ~/projects/ltbl    # 반드시 프로젝트 루트에서 실행
python3 main.py
```

### pygame 창이 뜨지 않음 (macOS)
macOS에서 pygame은 메인 스레드에서만 창을 열 수 있다.  
터미널에서 직접 실행해야 하며, VS Code 통합 터미널에서 작동하지 않을 수 있다.

### SQLite DB가 없어서 `analysis.py` 실패
시뮬레이션을 먼저 실행해 데이터를 생성해야 한다.
```bash
python3 main.py --steps 100   # 데이터 생성
python3 analysis.py           # 그 다음 분석
```

### Ray 메모리 부족
`island_model.py`의 `IslandConfig`에서 `initial_agents`와 `max_agents`를 줄인다.
```python
IslandConfig(..., initial_agents=30, max_agents=200)
```

---

<a name="english"></a>

---

# Development Environment Setup Guide

**[한국어](#ltbl-개발환경-설정-가이드) | English**

---

## Prerequisites

- **OS**: macOS 13+, Ubuntu 22.04+, or Windows via WSL2
- **Python**: 3.11 or higher
- **Package manager**: [uv](https://astral.sh/uv)

```bash
# Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone
git clone https://github.com/magi-balthasar/ltbl.git
cd ltbl
```

---

## Phase 1 — Primordial Sea (Current)

```bash
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Verify:**
```bash
python3 -c "import numpy, scipy, ray, matplotlib, pygame; print('All OK')"
```

**Run:**
```bash
python3 main.py --single --steps 200   # single island, no Ray
python3 main.py --steps 500            # 8 islands via Ray
python3 visualize.py                   # real-time visualization
python3 analysis.py --show             # post-run analysis graphs
```

---

## Phase 2 — Distributed Neural Net (Planned)

```bash
uv pip install torch networkx
```

---

## Phase 3 — Arthropod (Planned)

```bash
uv pip install pymunk torch
```

---

## Phase 4–5 — Fish / Quadruped (Planned)

```bash
uv pip install mujoco dm-control torch
```

---

## Phase 6–7 — Social Learning / Abstraction (Planned)

```bash
uv pip install torch transformers wandb
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'pygame'` | venv not activated | `source .venv/bin/activate` |
| Ray can't find modules | Not running from project root | `cd ~/projects/ltbl` first |
| pygame window doesn't open (macOS) | Must run from main thread | Run from native Terminal, not VS Code |
| `analysis.py` — DB not found | No simulation data yet | Run `python3 main.py --steps 100` first |
| Ray out of memory | Too many agents | Reduce `initial_agents` / `max_agents` in `island_model.py` |
