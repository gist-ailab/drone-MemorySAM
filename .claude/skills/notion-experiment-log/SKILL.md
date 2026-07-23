---
name: notion-experiment-log
description: 실험 결과·모델 변경을 노션 실험노트에 기록하고 동적으로 갱신한다. "노션에 기록해줘 / 실험 페이지 업데이트 / 결과 올려줘 / 벤치마크 표 갱신" 류 요청에 쓴다. 페이지를 새로 만들지 말고 기존 페이지를 절 단위로 교체하는 것이 기본. 분석 자체는 seg-analysis / det-analysis 스킬.
---

# 노션 실험노트 기록

> **이 스킬은 "기록"만 한다.** 수치를 만들거나 판정하는 건 `seg-analysis`·`det-analysis`,
> 다음 모델 설계는 `model-proposal` 스킬이다. 판정이 안 끝났으면 먼저 그쪽을 돌린다.

## 0. 대상 (고정 ID)

| 대상 | ID | 용도 |
|---|---|---|
| `실험노트` DB | `8ec54838-faff-4534-bc05-590dcebcc21a` | 실험당 1행 = 1페이지 |
| `📊 벤치마크 (누적)` | `3a388e43-5b22-813d-829b-e55ae2ca3a77` | **수치의 단일 출처.** 모든 실험 페이지가 링크 |
| `🌐 Project Pages` relation | `f1a55348-9c2f-443d-8187-5c8174c1cfbf` | 드론 과제 |
| `My Paper` relation | `33d05310-a165-408a-b0b8-ec4427d1fe2c` | |

⛔ **교수님 미팅 섹션은 건드리지 않는다.** MemorySAM 메인 페이지(`2c2a16c05735423982670aeee94fb179`)의
`Meeting` h3 아래 toggle들이 그것. 그 외 영역은 자유롭게 추가·삭제 (user 승인 2026-07-21).

인증은 `~/.claude.json`의 `notion-owner` 통합 토큰을 헬퍼가 자동으로 읽는다. OAuth 불필요.
⚠️ 이 토큰은 **워크스페이스 전체 쓰기 권한**이 있다. 위 대상 밖은 만지지 말 것.

```python
import sys; sys.path.insert(0, ".claude/skills/notion-experiment-log")
from notion_api import *
```

---

## 1. 무엇을 기록하는가 — 항목 체크리스트

**수치만 덤프하면 3주 뒤에 못 읽는다.** 아래 8개가 다 있어야 페이지가 자립한다.

| # | 항목 | 왜 필요한가 | 빠지면 생기는 일 |
|---|---|---|---|
| **1** | **ckpt 검증 계층** — val-best냐 test-best냐 | test-best는 test를 보고 고른 것 = 논문 사용 불가 | 2026-07 `test 57.60 = SOTA 돌파` 오보 → 철회. 문서 곳곳에 아직 잔존 |
| **2** | **배경 — 직전 버전의 실패를 수치로** | 이 실험이 존재하는 이유 | "왜 이걸 만들었지"를 아무도 답 못 함 |
| **3** | **제안 구조 — 도면 + 설계 근거** | 무엇을 바꿨고 왜 그 선택인지 | 코드를 다시 읽어야 구조를 앎 |
| **4** | **결과 — 게이트 통과 여부 먼저, 그 다음 수치** | 판정이 표 맨 위에 와야 읽힘 | 숫자 나열만 남고 결론이 안 보임 |
| **5** | **판정 — 모듈이 실제로 일했나** | 성능 Δ와 별개로 모듈 작동 여부 | no-op 모듈을 성공으로 오독 (4연속 발생) |
| **6** | **오염·무효 플래그** | 버그·불공정비교로 무효화된 판정 | ISSUE-026처럼 뒤늦게 뒤집힘 |
| **7** | **미기록·확인 필요** | 아직 안 한 것을 명시 | 없는 걸 있다고 착각 |
| **8** | **출처 — 코드 경로만** | 재현 가능해야 함 | 내부 로그 경로가 랩 공용 문서에 노출 |

### 1-1. ckpt 검증 계층 (가장 중요)

```
val-best  = val로 고른 ckpt + 그 epoch의 test        →  ✅ 논문 사용 가능 (= "합법")
test-best = test로 고른 ckpt                          →  ❌ 사용 금지, 반드시 무효 라벨
공식 test = 서버 채점(Codabench 등), val-best 제출     →  ✅ 최상 (훔쳐보기 구조적 불가)
```
test-best 수치를 지우지는 말 것 — **빨간 callout에 격리**하고 "무효"를 명시한다.
로그에 살아 있으면 다음 세션이 다시 인용하기 때문.

### 1-2. 결과에 반드시 들어갈 수치

- 게이트 통과 여부 (기준선 대비 Δ, 사전등록 기준과 대조)
- val / test mIoU + **epoch** + ckpt 종류
- **조건별**(day/night, clear/fog/rain/snow) + **클래스별**(특히 thin class 사망)
- **모듈 토글 Δ** — `|Δ|>0.5` 이고 예측 일치도 `<0.99` 여야 유효. 아니면 **no-op**
- 게이트 개방값 (zero-init residual이 실제로 열렸는지: β, σ(a), α …)
- effective rank / CKA / adapter Δacc (병목 진단)
- 학습 상태: 서버, epoch/총epoch, 분당속도, ETA, 정체·붕괴·사망 여부

### 1-3. 근거 사슬 (`제안 구조` 안의 h3)

설계 하나하나에 **왜 그걸 골랐는지**를 붙인다. 좋은 예:

> γ init을 0이 아니라 0.1로 둔 이유 — `tanh(0)=0`이면 MLP가 첫 스텝부터 gradient를
> 못 받아 zero-init 잔차 사장(4연속 no-op)의 재판이 됨. 스모크로 실증.

문헌 근거는 arXiv id로, 코드 근거는 `file:line`으로.

### 1-4. 벤치마크에 누적할 것

실험은 **1:1 비교가 아니라 쌓여야** 비교가 된다. 벤치마크 페이지에는:
- 우리 계보 전 버전 (P28 → 최신) 한 표에
- **선행연구 수치 같은 표에** (SOTA·val 1위·test 1위 명시)
- 백본·모달리티 수를 같이 — 공정 비교인지 드러남
- 판정 규칙 (ckpt 규약, 게이트 기준, 1pt 미만 주장 금지, eval-resize 지터 등)

---

## 2. 동적 갱신 프로토콜

**원칙: 페이지를 새로 만들지 않는다. 절(section) 단위로 교체한다.**
`replace_section()`은 멱등이라 같은 스크립트를 다시 돌려도 중복이 안 생긴다.

```python
DB = "8ec54838-faff-4534-bc05-590dcebcc21a"

pid = find_page(DB, "P39")          # 1) 기존 페이지 먼저 찾는다
if pid is None:                      # 2) 없을 때만 새로 만든다
    pid = create_experiment_page("P40", ...)

replace_section(pid, "결과", [       # 3) 절 통째 교체 — 멱등
    h3("DELIVER"), table([...]), para(...),
])
print(audit(pid))                    # 4) 반드시 검사 → {'src': [], 'tone': []}
```

### 갱신 시나리오별

| 상황 | 방법 |
|---|---|
| 새 epoch 수치 | `replace_section(pid, "결과", ...)` |
| 학습 종료/사망 | 요약 callout + run 표 갱신, 상태를 "사망(SIGKILL)" 등 사실대로 |
| 판정 확정 | `replace_section(pid, "판정", ...)` |
| 버그로 판정 무효화 | 기존 수치 **유지** + "판정 보류" 문구를 덧붙임 (지우지 말 것) |
| 새 실험 시작 | DB에 행 생성 → 8항목 채움 → 벤치마크 표에 행 추가 |
| 벤치마크 갱신 | `find_table(BM, "버전")` → `rebuild_table(tid, rows)` |
| 제목이 낡음 | `edit_text(page_title_block, ...)` 아니라 `PATCH /pages/{id}` properties |

### 상태 표기 어휘 (사실대로)
`학습 중(ep N/M)` · `완주` · `정체(val N ep 무갱신)` · `붕괴` · `사망(원인)` · `학습 대기` · `미실행`

---

## 3. 페이지 구조 (이 순서 고정)

```
0. 요약 callout      3줄: 무엇을 바꿨나 / 결과 수치 / 판정
   [오염 callout]    무효화된 판정이 있으면 여기 (빨강)
1. 배경 — 직전까지의 문제     직전 버전 실패를 수치로
2. 제안 구조          도면(PNG) + 캡션 + 접힌 mermaid 소스
                     + 구성요소 표(변경 ID / 내용 / file:line)
                     + h3 `근거 사슬`  ← 왜 이 설계인지
                     + h3 `구현·세팅`  (config, 서버, 커밋)
3. 결과              게이트 통과 여부 → 표 → 조건별 → 클래스별
                     + 벤치마크 페이지 링크 (link_page)
4. 판정              작동했나 / 안 됐으면 왜(진단 수치) / 다음으로 넘긴 것
5. [표준분석]         있으면
6. [분석 그림]        있으면
7. 출처 callout       코드·config·커밋·ckpt·서버만
```

### 도면
`put_diagram(pid, after_id, mmd_text, caption)` 한 번이면 렌더+업로드+토글까지 끝난다.
- 바뀐 노드 `:::new`(노랑) · 제거 `:::off`(회색 점선) · 버그 지점 `:::bug`(빨강) · 이전 세대 `:::old`
- **classDef에 `color:`를 반드시 넣는다** — 빼면 노션 다크모드에서 글자가 안 보인다(실제 사고)
- 노션이 mermaid를 자동 렌더하지만, PNG를 같이 넣어야 스크롤 중에 바로 보인다

---

## 4. 말투 (사용자 본인이 쓴 것처럼)

기존 페이지 예시: *"3090 * 8대로 A800*4 4 batch와 유사하게 성능 수치가 나올 수 있도록 세팅"* /
*"논문 기록 수치 63.48, 재현 수치 63.21"*

- **개조식.** `~함 / ~됨 / ~없음 / ~실패 / ~미달 / 기록 없음`
- 한 문장 = 한 사실. 표를 쓸 수 있으면 표.
- 추정은 `(추정)`, 미검증은 `검증 기록 없음` 명시
- **금지**: "구조적 성과", "유일한 성과", "의미가 있다", "주목할 점", "흥미롭게도", "무려", "확연히", "시사한다"
- 강조(볼드)는 수치와 판정에만. 이모지는 callout 아이콘만.

`audit()`이 위 금지어를 자동 검사한다.

---

## 5. 출처 규칙 (랩 공용 문서)

| ✅ 써도 됨 | ⛔ 절대 금지 |
|---|---|
| `semseg/...`, `configs/...`, `tools/...` | `.claude_logs/...` **전부** |
| 커밋 해시, 브랜치명 | monitor-log / arch-evolution / registry.md |
| ckpt·산출물 경로(`/drone_nas/...`) | status/current.md / plan.md / history-*.md |
| 논문명·venue·arXiv id | issues-and-fixes.md / experiments/analysis/*.md |
| 서버명(jarvis/hpca100/lecun) | NAS 볼트(`/nas_jm/Research/...`, `fact_related.md`) |

분석 결과를 인용할 땐 **산출 도구**(`tools/module_ablation.py`)와 **결과 디렉터리**로 대체한다.
이슈는 번호(`ISSUE-026`)와 **수정 커밋·코드 경로**로 지칭한다.

---

## 6. 함정 (실제로 겪은 것)

1. **test-best 오염** — `registry.md:25`류에 아직 무효 수치가 "달성"으로 남아 있다. 그대로 옮기지 말 것.
2. **stale canonical** — 레포의 관련연구 문서가 "canonical"이라 표시돼 있어도 낡았을 수 있다. 날짜 확인.
3. **오귀속** — "DFormer DELIVER 68.0"은 실제로 OmniSegmentor 수치였다. 논문 원문 확인.
4. **3-클러스터 규칙** — DELIVER 66.30=B2·val / 53.0=B2·test / 59.18=**B0**·val. 섞으면 틀린 비교.
5. **증강 버그가 세대 비교를 오염** — ISSUE-026(ColorAugSSD). 비교 전 학습 시점의 증강 설정 일치 확인.
6. **노션 100블록 한도** — `children`은 한 번에 100개. 넘으면 나눠서 PATCH.
7. **다크모드 대비** — mermaid classDef에 `color:` 필수.
8. **표 행 삭제 불가** — `rebuild_table`은 덮어쓰기+추가만. 행을 줄이려면 개별 `delete`.

---

## 7. 마무리 체크

```python
hits = audit(pid)
assert not hits["src"],  hits["src"]    # 내부 문서 경로 유출
assert not hits["tone"], hits["tone"]   # AI 티 나는 표현
```
그리고 사람 눈으로 한 번 — 브라우저로 열어 도면이 렌더되는지, 표가 안 깨졌는지 확인한다.

기록이 끝나면 **레포 문서도 갱신**한다(`status/current.md`, `experiments/registry.md`).
노션만 최신이고 레포가 낡으면 다음 세션이 낡은 쪽을 읽는다.
