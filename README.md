# RAG-пайплайн для банковского Q&A на русском (Альфа-Банк × МФТИ)

Локальная (полностью на open-source моделях, без проприетарных API) RAG-система: на вход —
вопрос пользователя, на выход — обоснованный ответ на русском, опирающийся на **фиксированный**
корпус из базы знаний банка. Если в корпусе нет ответа или вопрос персональный/нерелевантный,
система отвечает ровно `Нет ответа.`.

## Задача и метрика

- **Вход:** `questions.csv` (6977 вопросов), `websites.csv` (1937 страниц базы знаний),
  `sample_submission.csv` (эталон, близкий к ground truth).
- **Выход:** `submission.csv` со столбцами `q_id, answer_new` (6977 строк).
- **Метрика:** `Score = BERTScore-recall × L`, где `L` — штраф за длину по отношению
  `r = len(ответа) / len(эталона)`: `L = 1` при `r ≤ 1.5`, далее линейно убывает до 0 при `r = 3`
  и равно 0 дальше. То есть **переумноженный по длине ответ обнуляется**.
- 32% эталонных ответов — это ровно строка `Нет ответа.`, поэтому "решение отвечать или
  воздержаться" (abstain) критично для итогового скора.

## Архитектура пайплайна

```
вопрос
  │
  ├─ Retrieval (гибрид):  BM25  +  плотный поиск (BGE-M3, cosine)
  │        └─ слияние RRF  →  bge-reranker-v2-m3  →  top-5 фрагментов
  │
  ├─ Generation:  T-pro-it-2.1 (Qwen3-32B, дообученный под RU/банкинг) через vLLM
  │        └─ answer-eager промпт: по умолчанию даёт ответ в 2–3 предложения,
  │           Нет ответа. — только для персональных/нерелевантных вопросов
  │
  └─ Post-processing:  жёсткое ограничение длины (под L-gate),
           каноническая строка "Нет ответа."
```

### Текущая конфигурация (`src/rag/config.py`)

| Компонент | Значение |
|---|---|
| Эмбеддинги | `BAAI/bge-m3` (dim 1024) |
| Ретривер | `hybrid` — BM25 + dense, слияние RRF (`rrf_k=60`), `k_bm25=30`, `k_dense_fuse=30` |
| Reranker | `BAAI/bge-reranker-v2-m3`, финальный `top-5` |
| Генератор | `t-tech/T-pro-it-2.1`, `temperature=0.0`, `max_tokens=128`, thinking выключен |
| Abstain | `abstain_threshold=0.0` — порог по скору отключён, abstain решает LLM |
| Длина | `len_unit="words"`, жёсткое усечение под L-gate |
| VRAM | `gpu_mem_util=0.80` (реранкер 5.5 ГБ co-resident с vLLM) |

Индекс: 5735 фрагментов из 1883/1937 документов (отброшен только мусор < 20 символов).

## Структура репозитория

```
src/rag/
  schema.py        — структуры данных
  config.py        — вся конфигурация (модели, пороги, длина)
  indexer/         — очистка, чанкинг, эмбеддинги, сборка индекса
  retriever/       — dense, bm25, hybrid (RRF + reranker), factory, кэш
  generator/       — промпт, LLM (vLLM), abstain-логика, контроль длины
  eval/            — метрика (BERTScore × L), abstain-метрики, dev-набор
  pipeline.py      — сборка всего вместе
scripts/
  build_index.py     — строит artifacts/ (1 мин на GPU)
  run_pipeline.py    — генерирует submission.csv (6977 строк, 2 часа на H100)
  evaluate.py        — dev n=300: abstain P/R/F1 + 4-квадрантная разбивка
  smoke.py           — быстрый прогон на нескольких вопросах
  metric_probe.py    — референсные скоры all-abstain (без vLLM)
  length_sweep.py    — офлайн-подбор усечения по dev_preds.parquet (без vLLM)
  threshold_sweep.py — офлайн-подбор порога abstain (без vLLM)
tests/               — 43 юнит-теста (без GPU)
```

## Зависимости и окружение

Тяжёлые ML-зависимости (`sentence-transformers`, `bert-score`, `pyarrow`, `vllm`) вынесены в
extra `gpu` и импортируются лениво — чтобы лёгкое локальное окружение и юнит-тесты оставались
быстрыми.

Nvidia драйвер >= 580 (CUDA 13).

## Как запустить

```bash
# Лёгкое окружение (локально) — логика + юнит-тесты, без GPU
uv sync && uv run pytest -q              # 43 теста

# GPU-box (драйвер >= 580)
uv sync --extra gpu                      # + sentence-transformers, bert-score, pyarrow, vllm
uv run python scripts/build_index.py     # -> artifacts/ (1 мин)
uv run python scripts/evaluate.py        # dev n=300: abstain rate + разбивка
uv run python scripts/run_pipeline.py    # -> submission.csv (6977 строк, 2 часа)
```
