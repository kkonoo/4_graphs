# Part 2 후반(10–13) 인수인계 프롬프트

새 세션을 시작할 때 아래 `---` 사이를 통째로 붙여넣으세요.

**그 전에 할 것**
1. 이 폴더(`D:\자료\S_M_3_Neural_network\Claude outputs\git_graphs`)를 세션에 연결
2. (권장) 지금까지 작업을 커밋·푸시 — 01–09가 전부 아직 커밋 전입니다
   ```bash
   git add -A && git commit -m "Part 1 (03-07) + Part 2 (08-09)" && git push
   ```

---

`4_graphs` 강의 사이트 작업을 이어서 해줘. Part 1(01–07)과 Part 2의 08·09는 끝났고, 이제 **10–13**을 쓸 차례야.

## 강의 개요

- **과목**: 네트워크와 딥러닝 — 생물의학 데이터를 위한 그래프 이론과 신경망
- **대상**: 습식실험 경험은 있지만 계산 분석 경험은 거의 없는 대학원생
- **언어**: 한국어 단일
- **실습**: Google Colab. 개념은 본문, 코드는 접힌 🐍 블록
- **저장소**: 연결된 `git_graphs` 폴더 (Quarto + GitHub Pages, remote `kkonoo/4_graphs`)

## 먼저 할 일

1. 템플릿 파악: `chapters/08_nn_basics.qmd`, `chapters/09_training.qmd` (Part 2 형식), `chapters/06_propagation.qmd` (Part 1 형식), `chapters/07_practice.qmd` 끝의 "🗺️ Part 1을 마치며" (13의 회고 형식)
2. 그림 규칙: `figs/style.py`, `figs/c08_training.py`, `figs/c09_curves.py` (캐시 쓰는 패턴)
3. `_quarto.yml`, `index.qmd`, `chapters/91_glossary.qmd` 구조 확인

## 현재 상태

| 챕터 | 줄 수 | 그림 | 핵심 실측 |
|---|---|---|---|
| 08 회귀에서 신경망으로 | 404 | 3 | 선형 3층 = 로지스틱 회귀 / XOR 은닉 2개 성공 sigmoid 77·tanh 52·ReLU 18 (시드 100) / softmax 두 번이면 loss 바닥 0.5514 |
| 09 학습의 실제 | 379 | 4 | 특징선택 누수 0.77 vs 0.51 / TCGA 유방암 MLP 0.817 ≥ EN 0.792 ≥ L2 0.788 > RF 0.725 / 합성 batch 뒤집으면 RF 0.52 |

- `_quarto.yml`: 08·09 등록 완료. 사이드바에 10–13이 `# - text:` / `#   href:` 쌍으로 주석 처리돼 있음 → 챕터 끝낼 때마다 해제 + `render:` 목록에 추가
- `index.qmd`: 08·09 callout 완료. "10 ~ 13 · 준비 중" callout을 챕터별로 교체
- `91_glossary.qmd`: 08·09 용어 추가 완료. 10–13 용어 일부는 이미 표에 있음(오토인코더, 잠재 공간, 어텐션, 노드 임베딩, node2vec, transductive/inductive, 메시지 전달, GCN/GraphSAGE/GAT, 과평활화, WL 검정, 그래프 트랜스포머) → 새 용어만 추가
- 전체 렌더 검증(2026-09-19): 12페이지, 경고 0, 깨진 내부 링크·앵커·이미지 0
- 확인된 앵커: `05_community.qmd#c05-s4`(스펙트럴), `06_propagation.qmd#c06-s2`(RWR), `04_random_null.qmd#c04-s2`(null model)

## 저장소 규칙 (01–09와 동일)

- 파일: `chapters/NN_slug.qmd`, `figs/cNN_*.py` → `images/cNN/*.png`, 본문에서 `../images/cNN/이름.png`
- 슬러그: `10_architectures`, `11_node_embedding`, `12_gnn`, `13_frontiers`
- 앵커 `{#cNN-sM}` / `{#cNN-sM-K}`, 상호참조 `🔗 [05 §4](05_community.qmd#c05-s4)`. 아직 없는 챕터는 링크 말고 **볼드**
- 챕터 템플릿: front matter + `ai-context` 메타(10–20줄, 구체적 수치와 함수명; 큰따옴표·`&` 쓰지 말 것) → 🎯 학습 목표(접힌 callout) → 🤔 질문 → `##` 섹션 3–4개 → 📝 요약(8–10, ⚠️ 3–4) → ➕ 더 읽을거리(DOI 링크)
- 문체: 개조식 + **볼드 리드인**, 비교는 표로, ⚠️ 함정 3–5개/챕터, `#`(h1) 금지
- callout 제목에 ⚠️ 붙이지 말 것 (아이콘 중복)
- ⚠️ ```` ```{python} ```` 실행 청크 금지 (CI에 jupyter 없음). plain ```` ```python ````만
- `assets/`, `_extensions/`, `.github/` 건드리지 말 것
- 분량 목표 250–360줄 (실습 블록이 2개면 400줄 전후까지 허용)

## ⚠️ 작업 원칙 — 제일 중요

- **본문의 모든 수치는 직접 돌려서 얻은 값.** 실습 코드는 본문에 들어가는 그대로 실행해서 확인
- 재현 안 되면 재현된 척하지 말 것. 계획한 메시지와 반대로 나오면 **측정한 대로** 쓰고 사용자에게 알릴 것
  - 예: 09에서 "샘플 수백 개면 딥러닝이 진다"를 계획했지만 TCGA 유방암에서는 MLP가 조금 앞섰음 → "누가 이길지는 재봐야 안다"로 서술, 문헌(Smith 2020, Christodoulou 2019)은 인용으로만
- 문헌 인용은 DOI를 실제로 확인한 것만 (WebSearch/WebFetch). 확인 못 하면 문장을 뺄 것
- 무거운 실험은 결과를 `figs/cache/cNN_*.csv`에 저장하고 그림 스크립트가 캐시를 재사용 (09 패턴). 원자료 캐시(`*.npz`)는 `figs/cache/.gitignore`로 제외
- 난수는 모두 고정 (numpy `default_rng`, `torch.manual_seed`, sklearn `random_state` — saga 같은 solver 포함)

## 환경 — 09에서 알게 된 것

- **네트워크**: 클라우드 컨테이너와 이 컴퓨터 셸 모두 **GitHub(raw/media.githubusercontent.com)와 PyPI만** 열려 있음. download.pytorch.org, UCSC Xena, UCI, OpenML, GEO, 10x, figshare, zenodo, huggingface 전부 차단
  - 검증된 데이터: TCGA는 cBioPortal datahub (`https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/<study>/...`), 단일세포는 `sc.datasets.pbmc3k_processed()` (raw.githubusercontent.com/chanzuckerberg/cellxgene, 2,638세포, 8개 세포유형)
  - `sc.datasets.pbmc3k()` 원본은 falexwolf.de라 불가
  - **미확인**: PyG `Planetoid('Cora')`는 github.com/kimiyoung/planetoid에서 받으므로 될 가능성 높음 — 먼저 확인. `TUDataset`(ENZYMES)은 chrsmrrs.com이라 막힐 가능성 높음
- **설치**: `pip install torch`(PyPI에서, CPU로 동작), quarto는 GitHub releases의 .deb, `pip install scanpy`는 pandas를 3→2.3으로 내림 → pickle 캐시가 깨짐. 캐시는 npz/CSV로
- **렌더 검증**: 이 컴퓨터엔 quarto가 없음 → 폴더를 tar로 묶어(`.git` 제외) 연결 폴더 안에 두고 stage → 클라우드에서 `quarto render` → 링크·앵커 검사 스크립트 → Playwright로 callout 펼친 스크린샷. 임시 tar는 끝나고 삭제
- **백그라운드 실험**: 도구 호출 10분 제한. `setsid nohup python3 x.py > x.log 2>&1 &`로 띄우고 로그 확인. 대기 루프에 `pgrep -f 이름` 쓰지 말 것(래퍼 셸까지 잡힘), `pkill -f`도 자기 셸을 죽일 수 있음
- **sklearn 1.8**: `penalty="elasticnet"`에 FutureWarning — Colab 호환 위해 penalty는 유지하고 경고만 끔
- CPU 2코어: 09의 전체 실험은 몇 시간 걸렸음. 반복 수·그리드를 처음부터 작게 잡을 것

## Part 1·09가 깔아놓은 연결

| 앞 | → | 뒤 |
|---|---|---|
| 05 §4 스펙트럴 (라플라시안 고유벡터) | → | **11** 노드 임베딩 = 행렬 분해 |
| 06 §2 RWR (선형 전파) / 08 §1-2 "선형층을 쌓아도 선형" | → | **12** message passing = RWR에 학습되는 W + 활성함수 |
| 06 §1 relational classification | → | **12** GNN의 뼈대 |
| 04 §2 null model / 09 §3 정직한 비교 | → | **13** 논문 읽는 눈 (baseline) |
| 09 §1-3 n은 무엇인가 (pseudoreplication) | → | **13** "n은 세포인가 환자인가" |
| 01 §3 임계값·k / 07 study bias | → | **13** 그래프를 어떻게 만들었나, 데이터 편향 |

관통 주제 세 가지 (index.qmd): ① 엣지는 선택이었다 ② null 없이는 어떤 숫자도 의미가 없다 ③ 그림은 측정이 아니다 (③은 13에서 회수)

## 남은 챕터 계획

### 10. 알아둘 만한 아키텍처 🏗️
- CNN: 지역성 + 가중치 공유 = 서열 위를 미끄러지는 모티프 스캐너. RNN은 짧게
- **Autoencoder / VAE** → scVI 계열의 뼈대
- Attention은 개념만 (12의 GAT, 13의 graph transformer 복선)
- ⚠️ 목적은 이름 외우기가 아니라 **어떤 귀납 편향을 넣었는가**
- 🐍 pbmc3k autoencoder latent → PCA와 비교. ⚠️ 비교는 UMAP 그림이 아니라 **수치로** (예: 세포유형 kNN 일치도, Leiden ARI) → 주제 ③
- 📊 아이디어: 모티프를 심은 합성 DNA 서열에서 1D CNN 필터가 모티프를 찾는지 (실측)

### 11. 그래프 임베딩 🪢
- encoder–decoder 프레임: **similarity를 어떻게 정의하느냐가 전부**
- DeepWalk → node2vec (p, q로 BFS/DFS 성향). 구현은 랜덤워크 + gensim Word2Vec(PyPI)로 가능
- 🔗 [05 §4](05_community.qmd#c05-s4) 스펙트럴과 같은 뿌리 — 행렬 분해 관점 (Qiu et al. 2018 WSDM, DOI 확인할 것)
- ⚠️ 한계 3가지: transductive, 노드 특징 못 씀, 그래프마다 재학습 → GNN이 필요한 이유
- 🐍 node2vec → 05의 커뮤니티와 비교 / 📊 p, q에 따라 임베딩이 어떻게 달라지는지 (실측)

### 12. 그래프 신경망 🕸️
- **message passing = aggregate + update**, 🔗 [06 §2](06_propagation.qmd#c06-s2) RWR의 "학습되는 버전"
- 왜 CNN을 그래프에 못 쓰나 (이웃 수가 다르고 순서가 없다)
- GCN / GraphSAGE / GAT 비교표, WL test와 GIN, 1-WL 한계
- ⚠️ over-smoothing — **실측**(층 수 1–16에서 노드 표현 간 거리 붕괴 + 정확도), over-squashing, homophily가 깨질 때
- 🐍 PyG 2.x로 Cora, GCN vs GAT — ⚠️ 기존 노트북 `D:\자료\S_M_3_Neural_network\09_Graph_Neural_Networks.ipynb`는 PyG 1.x라 그대로 못 씀
- ⚠️ 09 교훈 적용: GCN 옆에 **노드 특징만 쓴 로지스틱 회귀/MLP** baseline을 같은 split으로

### 13. 최근 흐름과 논문 읽는 눈 🔭
- 생물학 응용: 단일세포(kNN 그래프 GNN), 공간(GraphST·STAGATE 계열), 지식 그래프(약물 재창출), 분자
- 최근 흐름 개념만: graph transformer + positional encoding, heterogeneous graph, relational deep learning, graph + LLM, KG foundation model — **웹 검색으로 최신 확인** (CS224W 최신 커리큘럼)
- 생성 모델(GraphRNN, 분자 diffusion) 한 단락
- ⚠️ **논문 읽는 눈**: ✅ 로지스틱 회귀·랜덤 포레스트 baseline이 있는가 ✅ 그래프를 어떻게 만들었는지(k, 임계값) 밝혔는가 → 🔗 01 ✅ n은 샘플인가 세포인가 → 🔗 09 ❌ "GNN을 썼다"는 성능 주장의 근거가 아니다
- 🐍 pbmc3k kNN 그래프 → PyG Data → GCN 세포유형 분류 → **로지스틱 회귀와 비교** (결과는 실측대로)
- 마지막에 **강의 전체 회고** (07 끝 "🗺️ Part 1을 마치며"와 같은 형식)

## 챕터 하나 끝낼 때마다

1. `_quarto.yml` — `render:`에 추가 + 사이드바 해당 두 줄 주석 해제
2. `index.qmd` — "10 ~ 13 · 준비 중"에서 그 챕터를 빼고 01–09 형식의 callout으로 (질문 / 번호 목록 / ⚠️ 함정 / ➡️ 읽기)
3. `chapters/91_glossary.qmd` — 새 용어 추가 (Part 2 표, 챕터 번호) + 상단 ai-context 목록
4. 렌더 검증: 경고 0, 깨진 링크·앵커 0, 그림 스크린샷 확인
5. 파일은 이 폴더에 직접 씀 (zip 전달 불필요). 커밋은 사용자가 요청할 때만

---
