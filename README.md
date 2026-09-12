# XboxSupport AI Agent

An AI customer support agent for **XboxSupport** built on the [Twitter Customer Support dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

Given a customer tweet, the agent:
1. **Classifies** the intent into one of 8 categories
2. **Drafts** a grounded reply using RAG over 20k historical XboxSupport threads
3. **Decides** auto-handle or escalate to a human, with a stated reason

---

## Reproduce headline results in < 15 minutes

### 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Configure LLM backend

```bash
cp .env.example .env
# Edit .env — pick ONE backend:
#   LLM_BACKEND=anthropic  →  set ANTHROPIC_API_KEY
#   LLM_BACKEND=gemini     →  set GEMINI_API_KEY
#   LLM_BACKEND=ollama     →  install Ollama, run: ollama pull llama3.2
```

### 3. Run the agent on the pre-extracted Xbox sample

```bash
# Single message (quick demo)
python3 -c "
from src.agent import run
r = run('my xbox wont turn on at all just a black screen', verbose=True)
"

# Batch (100 examples, ~5 min)
python3 src/pipeline.py --system agent --n 100
```

### 4. Evaluate all three systems

```bash
# Requires golden_labelled.csv (see below) + trained simple baseline
python3 src/baselines.py          # trains TF-IDF baseline, splits golden set
python3 src/evaluate.py --all --no-judge    # fast (no LLM judge calls)
python3 src/evaluate.py --all               # full (includes LLM judge + kappa)
```

---

## Project structure

```
src/
  data.py          — thread reconstruction from raw twcs.csv
  llm.py           — unified LLM client (anthropic / gemini / ollama)
  retriever.py     — FAISS semantic retrieval index
  agent.py         — classify + draft + escalate pipeline
  baselines.py     — trivial baseline + TF-IDF/LogReg baseline
  evaluate.py      — metrics, LLM judge, Cohen's kappa
  pipeline.py      — batch runner
  sample_golden.py — stratified sampling for golden eval set
  label.py         — interactive CLI labelling tool

data/
  xbox_threads.csv          — 20,213 reconstructed Xbox conversations

eval/
  golden_unlabelled.csv     — 178 sampled examples (pre-labelling)
  golden_labelled.csv       — hand-labelled golden set (post-labelling)

outputs/
  faiss.index               — semantic retrieval index
  golden_train/test.csv     — 80/20 split of golden set
  eval_results/             — per-system predictions + metrics JSON
```

---

## Reproduce from raw data (full pipeline)

```bash
# 1. Download dataset (requires Kaggle credentials)
kaggle datasets download -d thoughtvector/customer-support-on-twitter \
  -p data --unzip

# 2. Reconstruct Xbox threads
python3 src/data.py

# 3. Build FAISS index (one-time, ~2 min)
python3 src/retriever.py

# 4. Label golden set (interactive, ~30-40 min)
python3 src/sample_golden.py
python3 src/label.py

# 5. Train simple baseline
python3 src/baselines.py

# 6. Evaluate
python3 src/evaluate.py --all
```

---

## Intent taxonomy

| Intent | Description |
|---|---|
| `account_access` | Sign-in, password, gamertag, account settings |
| `billing_subscription` | Refunds, charges, gift cards, Xbox Live Gold |
| `hardware_device` | Controller, console, display, physical damage |
| `network_connectivity` | Xbox Live errors, NAT, server outages, ping |
| `game_content` | Game bugs, achievements, clips, installs, DLC |
| `app_feature` | Streaming apps, Mixer, feature requests |
| `enforcement_ban` | Bans, suspensions, account fraud |
| `other` | Chitchat, thanks, follow-up, unclear |

---

## Escalation logic

The agent escalates when:
- Intent is `enforcement_ban` (bans/fraud always need human review)
- Customer has a hardware damage or warranty claim
- Billing dispute with a specific amount
- Customer has already tried multiple troubleshooting steps
- LLM judge determines the reply cannot resolve the issue autonomously

Escalation proxy in the dataset: brand directed customer to DM / chat / provide gamertag (~30% of threads).

---

## LLM backends

| Backend | Set in .env | Notes |
|---|---|---|
| Anthropic Claude | `LLM_BACKEND=anthropic` + `ANTHROPIC_API_KEY` | Default: `claude-haiku-4-5-20251001` |
| Google Gemini | `LLM_BACKEND=gemini` + `GEMINI_API_KEY` | Default: `gemini-2.0-flash` |
| Ollama (local) | `LLM_BACKEND=ollama` | Default: `llama3.2` — no API key needed |

Switch models: set `LLM_MODEL=<model-id>` in `.env`.
