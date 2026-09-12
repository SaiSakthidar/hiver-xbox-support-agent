# Decision Log

Non-obvious decisions made during the build, with reasoning.

---

## 1. Context-aware agent (all three stages)

Decision: Pass prior conversation turns into `classify_intent`, `draft_reply`, and `decide_escalation`, not just the latest customer tweet.

Why:
46% of Xbox threads (9,337 / 20,213) have prior turns. A significant fraction of customer messages are follow-ups that are meaningless in isolation — "it still doesn't work", "none of that helped", "they have no info either". Classifying these without context forces the model to guess and will produce low-confidence, often wrong intent labels.

More importantly, the reply drafter without context will repeat steps already tried — the most frustrating failure mode in customer support. If the brand already said "try restarting your console", and the customer replies "still broken", a context-blind agent will say "try restarting your console" again.

Escalation also benefits: two rounds of failed troubleshooting in the context is a strong escalation signal that a single-tweet view misses.

Trade-offs considered:
- More tokens per request (~2× for threads with context) → higher latency/cost.
- Context window budget: we cap at 4 prior turns (`CONTEXT_TURNS = 4`) to bound prompt size.
- For threads without context (54%), the agent falls back gracefully to single-turn behaviour with no performance penalty.
- The `had_context` field in the output lets us measure whether context-aware classification actually changes the intent label — useful for failure analysis.

Alternatives rejected:
- Concatenate all prior text into one string: loses speaker-role information, confuses the model about who said what.
- Classify from scratch on every turn independently: correct in isolation but breaks down for follow-ups.

---

## 2. Brand: XboxSupport over SpotifyCares or AmazonHelp

Decision: Use XboxSupport (24,557 threads → 20,213 after reconstruction).

Why:
- 100% English (Spotify and Amazon have multilingual tweets that add noise).
- Clearest escalation signal: hardware faults, account bans, and fraudulent charges are unambiguous escalation cases with no equivalent in pure software brands.
- Intent variety is higher than Spotify (8 distinct intents vs ~5) without being overwhelming.
- Amazon is the largest dataset but has Japanese tweets and ~30% of threads are multi-language.

---

## 3. Thread reconstruction: walk the graph, don't filter by author_id join

Decision: Reconstruct threads by walking `response_tweet_id` / `in_response_to_tweet_id` links forward and backward from brand reply tweets.

Why:
The naive approach (filter inbound tweets whose `in_response_to_tweet_id` is in the brand's outbound tweet IDs) only finds customer messages that *follow* a brand reply — it misses the first customer message in every thread, which has `NaN` in that field. Our graph-walk correctly identifies first-turn messages and multi-turn threads.

Result: 20,213 threads with correctly linked (customer_text, brand_reply, prior_context) triples.

---

## 4. Multi-tweet reply stitching

Decision: Stitch consecutive brand tweets that form one logical reply (numbered `1`, `2`, `3` with agent initials `^EZ`, `^JL`), strip the suffix.

Why:
Xbox splits long replies across multiple numbered tweets. Without stitching, ground-truth replies are truncated fragments ("...you may need to get your  1 ^RR") and every reply-quality metric is measuring garbage.

---

## 5. Escalation proxy: DM/chat redirect as ground truth

Decision: Define escalation ground truth as "brand directed customer to DM, chat support, or provide gamertag" (~30% of threads).

Why:
There is no explicit escalation label in the dataset. This proxy captures the clearest signal: when Xbox couldn't resolve the issue on public Twitter, they moved to a private/dedicated channel. The proxy is imperfect (brands sometimes DM for routine cases too) but it is defensible and reproducible.

Known leak (documented in the "What's misleading" section of the report): ~10–15% of DM redirects are routine channel management, not genuine escalations. Our escalation recall will be underestimated as a result.

---

## 6. Intent taxonomy: 8 classes, derived from reading data

Decision: Define 8 intents from reading ~100 real customer tweets, not from clustering or an external taxonomy.

Why:
Clustering embeddings produces mathematically coherent but operationally meaningless groups (e.g., "negative sentiment tweets" rather than "billing issues"). Reading the data directly reveals the categories that matter for routing and resolution. Banking77's 77 intents were considered but are banking-specific and would require coarse-mapping that adds noise.

The 8 chosen intents (account_access, billing_subscription, hardware_device, network_connectivity, game_content, app_feature, enforcement_ban, other) cover >95% of observed tweet types with minimal overlap.

---

## 7. Golden set sampling: stratified + random, not pure random

Decision: Sample ~22 examples per intent (keyword-stratified) + 24 purely random examples, for 178 total.

Why:
Pure random sampling would under-represent rare but important intents (enforcement_ban, app_feature) since they make up <10% of threads. Stratification ensures the eval set is diagnostic across all intents.

The random slice forces exposure to long-tail and ambiguous cases that keyword heuristics miss, preventing the eval set from only containing textbook examples.

Known bias: Keyword stratification favors clear-cut examples within each bucket. The eval set is easier than the real distribution. Documented in "What's misleading."

---

## 8. LLM backend: unified client with Anthropic / Gemini / Ollama support

Decision: Abstract all LLM calls behind a single `chat()` function; backend selected via `LLM_BACKEND` env var.

Why:
Reproducibility for evaluators who may not have any specific API key. A grader with Gemini but not Anthropic can still run the full pipeline. Ollama support allows fully offline evaluation with no API cost.

Default model per backend chosen for cost-efficiency: `claude-haiku-4-5-20251001` (Anthropic), `gemini-2.0-flash` (Gemini), `llama3.2` (Ollama).

---

## 9. FAISS index capped at 15,000 threads

Decision: Build the retrieval index over 15,000 threads (not all 20,213).

Why:
`all-MiniLM-L6-v2` produces 384-dim float32 embeddings. 20k × 384 × 4 bytes ≈ 30 MB, manageable. We cap at 15k to keep index build time under 2 minutes on a laptop CPU and keep the index file under 25 MB for easy repo inclusion.

Coverage loss is minimal: at k=3, all 8 intent categories are well-represented in 15k threads.

---

## 10. Banking77: considered and rejected

Decision: Did not use the Banking77 dataset despite its availability.

Why the overlap looks appealing:
~1,270 Banking77 examples map conceptually to `billing_subscription` (request_refund, transaction_charged_twice, extra_charge_on_statement) and ~655 map to `account_access` (passcode_forgotten, unable_to_verify_identity). On the surface this looks like free training data for our two weakest classes.

Why it's actually net-negative for our pipeline:

TF-IDF learns vocabulary, not concepts. The discriminative tokens for Xbox `billing_subscription` are things like `xbox`, `gold`, `gamertag`, `redeem`, `live`, `subscription`. None of these appear in Banking77. Conversely, Banking77 introduces `card`, `pin`, `transfer`, `top-up`, `cheque`, words that almost never appear in Xbox tweets. Adding 1,270 banking examples to ~22 Xbox billing training examples would drown out the Xbox-specific signal, not amplify it. Performance on the Xbox test set would likely drop.

For the LLM agent, Banking77 is irrelevant entirely; the classifier is zero-shot and needs no labelled training data.

The one case where it would help: fine-tuning a BERT-style model with Banking77 pre-training followed by Xbox fine-tuning. That's a different architecture (neural, not TF-IDF) and outside the scope of what we defined as the "simple" baseline.

Conclusion: Domain mismatch makes it net-negative for TF-IDF and irrelevant for the LLM agent.

---

## 11. Judge calibration: Cohen's κ over Pearson r

Decision: Report Cohen's κ as the primary judge-human agreement metric, with Pearson r as secondary.

Why:
Percent agreement is inflated when one rating dominates (most replies rated 3–4). Cohen's κ corrects for chance agreement. We compute it on a 50-example sample from the golden set where both human labels and judge scores are available, following the assignment's explicit requirement for "evidence of how well your judge agrees with a human."
