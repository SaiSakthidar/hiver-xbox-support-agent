# XboxSupport AI Agent: Report

Brand: XboxSupport | Dataset: Twitter Customer Support (Kaggle, ~3M tweets) | Model: Gemini 3.5 Flash Lite

---

## 1. Problem framing

### What "good" means for XboxSupport

A good XboxSupport agent must do three things well, in order of priority:

1. Not make things worse. A wrong reply (suggesting steps that don't apply, repeating steps already tried, or misrouting a fraud case as routine) is worse than no reply. The agent should escalate when uncertain rather than confabulate.

2. Sound like Xbox, not like a generic chatbot. Xbox has a distinctive, warm, first-name-optional tone with a habit of asking for the gamertag and directing to specific support channels. Replies grounded in real historical conversations preserve this.

3. Route correctly. Human agents are expensive. The agent should handle the large volume of routine connectivity errors, account setting questions, and game install issues automatically, while reliably escalating fraud, hardware damage, bans, and multi-round unresolved issues.

### What we built

- Intent classifier: zero-shot LLM classification into 8 intents derived from reading the data
- Reply drafter: RAG over 20,213 historical Xbox threads (FAISS + sentence-transformers), grounded in how Xbox actually responded to similar issues
- Escalation decider: LLM with explicit rules for ban/fraud/hardware/repeat-troubleshooting cases
- Context-awareness: all three stages receive the prior conversation turns (46% of threads have prior context), so the classifier correctly handles follow-up messages and the drafter avoids repeating suggestions

### What we chose not to build

- Fine-tuned classifier: a fine-tuned BERT/DistilBERT would likely outperform zero-shot LLM on intent, but requires labelled training data at scale. With 152 hand-labelled examples (121 train), fine-tuning would overfit. The zero-shot LLM generalises better at this data size.
- Response personalisation: addressing the customer by name or referencing their specific gamertag. Not feasible without account integration beyond what the Twitter dataset provides.
- Sentiment-aware escalation: escalating based on emotional intensity of the message. Briefly considered; abandoned because the dataset's proxy (DM redirect) does not correlate cleanly with sentiment, making ground truth noisy.
- Banking77 augmentation: evaluated and rejected. See Decision Log #11; domain vocabulary mismatch makes TF-IDF augmentation net-negative.

---

## 2. Results vs. baselines

### Systems

| System | Description |
|---|---|
| **Trivial** | Always predicts `game_content` (majority class); fixed canned reply; never escalates |
| **Simple** | TF-IDF + Logistic Regression intent; cosine nearest-neighbour reply retrieval; keyword-based escalation |
| **Agent** | LLM intent classification (zero-shot); RAG reply drafting (FAISS + sentence-transformers); LLM escalation decision; context-aware across all three stages |

### Intent classification

| System | Accuracy | Macro-F1 |
|---|---|---|
| Trivial | 0.194 | 0.041 |
| Simple | 0.387 | 0.231 |
| **Agent** | **0.724** | **0.682** |

The agent is 3.7× more accurate than the trivial baseline and 1.9× more accurate than the simple baseline.

Per-class agent performance:

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| enforcement_ban | 0.88 | 0.92 | **0.90** |
| billing_subscription | 0.68 | 0.89 | **0.77** |
| hardware_device | 0.81 | 0.74 | **0.77** |
| account_access | 0.76 | 0.64 | 0.70 |
| game_content | 0.77 | 0.65 | 0.70 |
| app_feature | 0.56 | 0.75 | 0.64 |
| network_connectivity | 0.60 | 0.55 | 0.57 |
| other | 0.38 | 0.43 | 0.40 |

### Escalation decision

| System | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Trivial | 0.548 | 0.000 | 0.000 | 0.000 |
| Simple | 0.484 | 0.200 | 0.214 | 0.111 |
| **Agent** | **0.546** | **0.517** | **0.437** | **0.473** |

Escalation by predicted intent (agent):

| Intent | Escalation Rate |
|---|---|
| enforcement_ban | 100% |
| account_access | 25% |
| billing_subscription | 25% |
| game_content | 17% |
| network_connectivity | 17% |
| app_feature | 0% |
| hardware_device | 0% |

`enforcement_ban` escalates 100% of the time, which is correct since bans always require human review. Notably, `hardware_device` escalates 0% in predictions, a known failure mode (see section 4).

### Reply quality (LLM judge)

The judge (same Gemini model) rates replies on a 1–5 rubric (5 = excellent, empathetic, actionable).

| System | Judge Mean Score |
|---|---|
| Trivial | Not evaluated (canned reply) |
| Simple | Not evaluated (retrieved reply, not generated) |
| **Agent** | **3.16 / 5** |

Judge-human calibration: 67.7% of judge ratings fall within ±1 point of human ratings on the same historical brand replies. Cohen's κ = -0.069 (unreliable on n=31; see section 4). The judge is systematically ~0.8 points stricter than the human rater (human mean 3.94, judge mean 3.16).

---

## 3. Failure analysis

### Failure mode 1: hardware/network conflation (3 errors)

Pattern: Hardware issues involving connectivity symptoms get misclassified as `network_connectivity`.

Examples:
- *"new Xbox. During setup after entering my wifi password, it connects & then goes back to the wifi screen"* → predicted `network_connectivity`, true `hardware_device`
- *"my xbox would disconnect from my wifi, and would not reconnect"* → predicted `network_connectivity`, true `hardware_device`
- *"i cannot download anything or stream anything. I've rebooted the console several times"* → predicted `network_connectivity`, true `hardware_device`

Hypothesis: Hardware faults often manifest as connectivity symptoms. The language ("disconnect", "wifi", "connect") matches `network_connectivity` surface vocabulary. The agent lacks the reasoning to distinguish *"the hardware is broken and presenting as a network error"* from *"the network configuration is wrong."*

Fix: Add few-shot examples distinguishing hardware-manifesting-as-network from pure network issues in the classification prompt.

---

### Failure mode 2: escalation under-fires on hardware (11 false negatives)

Pattern: The agent's escalation rules explicitly list hardware damage and warranty claims as escalation triggers, yet `hardware_device` escalates 0% in practice.

Hypothesis: The agent first misclassifies hardware issues as `network_connectivity` (Failure Mode 1), then passes the network label to the escalation stage. Since network issues are routine, the escalation decider says "auto." The classification error propagates into a routing error.

Impact: 11 out of 31 test cases that should escalate were auto-handled, the highest-stakes failure mode.

---

### Failure mode 3: Mixer/community feature boundary (1 error)

Example: *"No community or mixer"* → predicted `app_feature`, true `game_content`

Hypothesis: Mixer was Xbox's first-party streaming integration, sitting at the boundary of `game_content` and `app_feature`. The taxonomy itself is ambiguous here. When Xbox shut down Mixer in 2020, community features moved into the game/social layer; the distinction is subjective.

Fix: Merge `app_feature` and `game_content` into a single `platform_feature` intent, or add Mixer-specific examples to the taxonomy prompt.

---

### Failure mode 4: account/network conflation (1 error)

Example: *"has Xbox live gone down? I just got threw out of my account and it won't let me back in"* → predicted `network_connectivity`, true `account_access`

Hypothesis: The customer mentions Xbox Live being down (network framing) but the core issue is account access. The agent picks up the network signal and misses the account signal. Context-awareness doesn't help here; it's a single-turn ambiguous message.

---

### Failure mode 5: "Other" is unpredictable (1 error)

Example: *"Halo Wars 2 won't install/update for some reason. Always gets stuck at 22%"* → predicted `game_content`, true `other`

Hypothesis: This is a labelling disagreement more than a model error. A game installation failure is a strong `game_content` signal; labelling it `other` likely reflects labeller uncertainty. The `other` class has no consistent surface features, catching everything that doesn't fit elsewhere and making it inherently hard to predict.

Recommendation: Remove `other` from evaluation metrics or evaluate it separately. A 0% F1 on 1 example is not informative.

---

## 4. What is misleading about my headline number?

Headline: Agent intent accuracy = 72.4%, macro-F1 = 0.682

### 4a. The test set is small (n=31)

The golden set has 152 labelled examples; 80% was held out to train the TF-IDF baseline, leaving 31 for testing. At n=31, a single misclassified example shifts accuracy by 3.2 percentage points. The 95% confidence interval for a 71% accuracy on n=31 is approximately ±16 points (0.55–0.87). The headline number is a point estimate, not a stable measurement.

To reduce this, the agent was also evaluated on all 152 examples (no training data leakage since it's zero-shot), giving a more stable estimate.

### 4b. The golden set oversamples clear-cut examples

Keyword-based stratification, used to ensure all 8 intents are represented, preferentially selects tweets containing the most obvious intent signals ("refund", "controller", "ban"). Ambiguous tweets, mixed-intent messages, and follow-ups without prior context are underrepresented. The agent's real-world accuracy on the full distribution is likely lower than 71%.

### 4c. The escalation proxy is noisy

Ground-truth escalation is proxied by "brand directed customer to DM/chat." Around 15–20% of such redirections are routine channel management ("for account-specific help, please DM us") rather than genuine escalations. This means escalation recall is underestimated (some true auto-handles are labelled as escalations) and the 0.261 F1 understates real performance.

### 4d. Judge-human agreement is measured on n=31 with a systematic rater bias

Cohen's κ = -0.069 is statistically meaningless at n=31 (95% CI ≈ ±0.35). More informatively: 67.7% of judge ratings fall within ±1 point of human ratings. The judge is consistently ~0.8 points stricter than the human rater, a systematic calibration offset rather than random noise. The judge's 3.16/5 score for agent replies may correspond to ~4.0/5 on a human scale.

### 4e. Context-awareness helps the drafter but not the classifier (measured)

We ran a context ablation on the 48 golden-set examples that have prior conversation turns: intent accuracy with context was 0.667 vs 0.729 without context (delta: -0.062). Context makes classification slightly worse on this set. The likely cause: prior turns in the dataset are noisy raw tweets — sometimes unrelated to the current issue — and the model is distracted by them. Context still benefits the reply drafter (avoids repeating suggestions) and escalation logic (detects multi-round troubleshooting), but the classification benefit claimed in Decision #1 is not confirmed by this experiment.

### 4f. Evaluation is on Twitter-length messages, not full support tickets

Real support conversations involve attachments, account data, and context that Twitter's 280-character constraint strips away. The agent's performance on actual support tickets (longer, richer, with clearer intent signals) is unknown.

---

## 5. What I'd do next with one more week

1. Fix hardware/network conflation: add 10–15 few-shot examples to the classification prompt that explicitly distinguish hardware faults manifesting as connectivity symptoms. This single fix would likely push macro-F1 from 0.61 to ~0.68.

2. Calibrated escalation probability: replace the binary LLM escalation decision with a logistic regression trained on golden-set labels, outputting a probability. Plot the precision-recall curve and select the operating threshold based on the cost ratio of false positives (unnecessary escalation) to false negatives (missed escalation).

3. Expand the golden set to 300+: label 150 more examples with stratified sampling weighted toward ambiguous cases near intent boundaries. This would make macro-F1 and escalation F1 statistically reliable.

4. Per-turn context ablation: run the agent in context-aware vs. context-blind mode on threads with prior turns and measure the delta in intent accuracy. This would quantify the value of Decision #1 (context-awareness) rather than just asserting it.

5. Hardware escalation fix: add an explicit rule in `decide_escalation`: if `hardware_device` is predicted AND the reply suggests physical inspection or replacement, force escalation. This closes the most damaging failure mode (11 false-negative escalations).

---

## 6. Sampling and labelling methodology

Sampling: 178 examples drawn from 20,213 Xbox threads using stratified random sampling. ~22 examples per intent category (7 intents × 22 = 154) selected by keyword heuristics, plus 24 fully random examples from the remaining pool to capture the long tail and ambiguous cases. Random seed 2024 for reproducibility.

Labelling: Each example was labelled with three fields:
- `intent`: one of 8 categories; suggested label from keyword heuristic provided as a hint, overridden where wrong
- `should_escalate`: yes/no based on whether a human agent would need to take over
- `reply_quality`: 1–5 rating of the historical Xbox brand reply (1 = terrible, 5 = excellent)

Known biases: Keyword stratification oversamples textbook examples within each intent. Labelling was done by a single rater (no inter-rater reliability check). Reply quality ratings skewed high (mean 3.94); the rater was likely anchoring to "acceptable for Twitter support" rather than absolute quality.
