### RAG regression gate: FAILED
prompt `answer-v1.3-concise` · model `qwen/qwen3.8-27b` · judge `openai/gpt-oss-120b` · subset `ci` (n=12)

| metric | value | baseline | floor | max drop | status |
|---|---|---|---|---|---|
| faithfulness | n/a | 1.000 | 0.8 | 0.05 | ❌ metric missing |
| answer_relevancy | 0.000 | 0.750 | 0.7 | 0.1 | ❌ below floor 0.7 |
| citation_accuracy | n/a | 1.000 | 0.8 | 0.1 | ❌ metric missing |
| refusal_correctness | 0.167 | 0.800 | 0.75 | 0.1 | ❌ below floor 0.75 |
| context_recall | 0.900 | 0.875 | 0.7 | 0.1 | ✅ |
