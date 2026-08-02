# Fine-tuning an LLM for Tool Use, in Portuguese

Teaching a small model to do **function calling**: deciding when to call a tool, which one
to call, with which arguments, and when to call none at all.

Since no ready-made dataset exists for fictional tools in Portuguese, steps 1 to 3
**build the training data from scratch**, using another LLM as the generator.

## References

- **[Unsloth](https://docs.unsloth.ai/)**, the library used for training: `FastModel` with `load_in_4bit=True`,
  `get_peft_model` for the LoRA, `train_on_responses_only` so the loss counts only on the
  assistant's turns, and `finetune_vision_layers=False`, which dissolved into a single
  parameter the complication of Gemma 4 being multimodal.
- **[Bonus unit 1 of the Hugging Face agents course](https://huggingface.co/learn/agents-course/bonus-unit1/fine-tuning)**
  uses the same skeleton: an `-it` base model, LoRA, `SFTTrainer`, and equivalent delimiters
  (`[TOOL_CALLS]` there, `<tool_call>` here). The meaningful difference is that the course
  uses a ready-made dataset, and here it was generated.
- **[Learning curves](https://huggingface.co/learn/llm-course/chapter3/5)** (Hugging Face
  LLM course); how to read training loss against validation loss, and the patterns of
  overfitting, underfitting and instability.

Outputs of this project:

- **W&B run:** https://wandb.ai/annajuliaasfiag/tool-use-ptbr

---

## The pipeline

The notebooks are numbered in execution order.

| # | notebook | reads | writes |
|---|---|---|---|
| 1 | `1_gerar_queries.ipynb` | n/a | `dados/tools_catalogo.json`, `dados/queries_geradas.csv` |
| 2 | `2_gerar_traces.ipynb` | catalog + queries | `dados/traces_progresso.json` |
| 3 | `3_preparar_dados.ipynb` | traces + queries | `dados/treino.jsonl`, `dados/validacao.jsonl`, `dados/teste.jsonl` |
| 4 | `4_treinamento_colab.ipynb` | the three `.jsonl` files | the LoRA adapter + `resultados/saidas_*.json` |

**Step 1. Generate queries.** The Qwen and Llama models were used, via Groq, to generate
30 fictional tools (name, description, parameters) and 518 user questions, classified as
`facil`, `dificil` and `sem_tool`. The `sem_tool` ones are questions that should **not**
trigger any tool; without them the model would learn to always call one.

**Step 2. Generate traces.** For each query, the LLM produces the full conversation: the
tool call, the simulated API return, and the final answer to the user. Progress is saved
in a dictionary indexed by row, so you can stop and resume when the API quota runs out.
Result: **484 traces** out of 518 queries, covering all 30 tools.

**Step 3. Prepare the data.** Wraps each turn in the protocol tags, appends the
instruction block to the system prompt, and splits into train/validation/test in a
stratified way (80/10/10, preserving the proportion of the three categories).

**Step 4. Train.** QLoRA on Colab (T4). Produces an 80 MB adapter, not a 20 GB model.

**Step 5. Evaluate.** Compares the fine-tuned model against the base model on the 51 test
traces, plus two experiments that probe for dataset leakage.

```
484   finished traces
      ├── 386  train        (80%)  → what the model learned from
      ├──  47  validation   (10%)  → only to measure during training
      └──  51  test         (10%)  → the model NEVER saw these
```
---

### The training run

386 examples, 3 epochs, 147 steps, 34min21 on a T4. LoRA `r=8`, `lora_alpha=8`,
**12,668,928 trainable parameters, 0.25%** of 5.12B. Adapter in `float32` → 48 MB of weights.

| epoch | validation loss | Δ |
|---|---|---|
| start (step 10) | 0.939 | n/a |
| 1 (step 50) | 0.528 | **−0.41** |
| 2 (step 100) | 0.492 | −0.036 |
| 3 (step 147) | 0.488 | −0.004 |

---

## The output protocol

Every assistant turn is one of two things, always delimited:

```
<tool_call>    {"nome_tool": "...", "argumentos": {...}}    </tool_call>
<final_answer> text for the user                            </final_answer>
```

And the tool's return arrives in `<tool_result>...</tool_result>`.

It is this uniformity that lets the code know, unambiguously, whether to execute a tool or
show the answer. In the `sem_tool` traces the answer **also** goes in `<final_answer>`; the
exception, were it missing here, would become an unhandled case in the parser.

---

## Structure

```
1_gerar_queries.ipynb          step 1, local
2_gerar_traces.ipynb           step 2, local; uses the Groq API
3_preparar_dados.ipynb         step 3, local
4_treinamento_colab.ipynb      step 4, copy of the Colab notebook (needs a GPU)

dados/
  tools_catalogo.json          30 fictional tools
  queries_geradas.csv          518 classified questions
  traces_progresso.json        484 traces (dict indexed by row; null = not generated)
  treino.jsonl                 386 examples
  validacao.jsonl               47 examples
  teste.jsonl                   51 examples

resultados/
  saidas_ft.json               condition A: fine-tuned model
  saidas_shuf.json             condition B: tools in shuffled order
  saidas_base.json             condition C: base model, no adapter

```
---

## Notes

The data is **synthetic** and the 30 tools are **fictional**; none corresponds to a real
API. The resulting model is for study, not for production.
