착수(running) 후 본문 수정 금지 — 바꿀 일은 맨 아래 '정정' 절에 날짜와 함께 덧붙인다.

# MCubeS 결측 강건성(학습 0): 논문 모델 C2D2 768 2시드의 EMM15·RMM(0.25/0.5/0.75) test 102 — RMMSS(AAAI'26) Table 2 의 MCubeS 기준선(CMNeXt·MAGIC·MAGIC++·AnySeg·RMMSS)과 같은 0채움 정의로 비교

- id: DRN-261105-02
- kind: analysis
- 문제(PRB): -
- 질문: MCubeS 에서도 두 패스 학습의 이득이 강건성으로 나타나는가(clean 은 중립 −0.10). 선행 비교표(RMMSS Table 2: EMM MCubeS RMMSS 43.22, MAGIC++ 42.53, MAGIC 42.61, CMNeXt 40.67, AnySeg 39.33 / RMM 44.18·44.61·42.25·41.25·42.50)가 있으므로 측정해 Table 4 에 MCubeS 절을 붙일 수 있는가
- 가설: C2D2 768 2시드 EMM15 평균 ≥ 50(선행 최고 43.22 보다 ≥ +5) 이고 w/o C2D2 768 대비 EMM Δ ≥ +3(DELIVER 에서 E1→Q2 가 +8.05). RMM 도 같은 부호
- 바꾼 변수(change): 측정만. 입력 = C2D2 768 s827·s828 val-best + w/o C2D2 768 s827·s828(강건성 짝) + E1Mc s827(참고). 변수 없음
- 기준선: RMMSS Table 2 MCubeS 보고값(0채움·15조합 평균, 백본 B0/B2, split 표기 없음 → test 102 관행으로 간주하고 각주) + 우리 w/o C2D2 768 같은 시드
- 성공 기준: 4모달(image·aolp·dolp·nir) 15조합 정규화 후 0채움(2503.18445 정의, DELIVER 와 같은 코드 경로), RMM r=0.25/0.5/0.75, test 102 전수, batch 4·seed 0. 가설 충족 시 Table 4 에 MCubeS 절(기준선 = RMMSS 표 인용) + 본문 한 문장. EMM Δ(C2D2 − w/o) < +1 이면 '두 패스 학습은 MCubeS 에서 clean·강건성 모두 중립' 으로 적고 숨기지 않는다
- 반증 조건: C2D2 EMM 이 w/o C2D2 보다 낮으면 두 패스 학습의 MCubeS 전이 실패로 적고, DELIVER 전용 효과임을 한계에 명시
- configs: -

## 정정
