# Attention — Interview Revision Progress

> One-shot read to resume studying. Context: revising ECE539 material for an interview.
> Treating topics as a beginner. Source deck: `L19_TF_Attention.pdf`.

## Study roadmap (the big picture)

The whole course arc: **Attention → Transformers → Language Models**, then two applied
branches (**Object Detection**, **Diffusion**).

Everything is built from ONE core operation (single-head attention). Once that's solid:
- **Self-attention** = same operation, Q/K/V from the same sequence (→ transformers)
- **Multi-head attention** = run the operation several times in parallel
- **Transformer block** = multi-head self-attention + feed-forward, stacked
- **LLMs** (GPT/BERT) = transformers trained at scale

Suggested prereqs (if rusty): NN basics → CNNs → RNNs/seq2seq.
Note: Object Detection is a CV side-branch (belongs near CNNs), Diffusion best studied last.

---

## What "attention" solves

Old seq2seq models compressed the ENTIRE input into a **single fixed-length context
vector**, which the decoder reused for every output word → information bottleneck.

**Attention** lets each output word **selectively focus** on the most relevant parts of the
input. Instead of one shared context vector, it computes a **fresh context vector per output
step** = a *weighted blend* of ALL input representations. The weights are recomputed
dynamically at each step.

---

## Query / Key / Value (Q, K, V)

Same input word gets cast into three different "views" via three learned matrices
(W_Q, W_K, W_V):

- **Query (Q):** what I'm looking for right now
- **Key (K):** how each input represents itself *for matching*
- **Value (V):** the actual content an input contributes *once matched*

**Why separate K and V?** They do two different jobs: Keys are optimized for *being
matched against a query*, Values for *delivering content*. It is NOT about size — K and V
are usually the same dimension. Forcing them equal would remove the model's flexibility to
optimize matching and content independently.

(Library analogy: Query = search terms, Key = index card used for matching, Value = the
book's actual content. Caveat: real index cards are smaller than books, but K is NOT a
compressed V — that's where the analogy leaks.)

---

## The engine: Score → Scale → Softmax → Sum

The 4 steps that turn a Query + set of Keys/Values into one **output vector**
(the output vector = the weighted sum of Values):

1. **Score:**  dot product of Q with each K → raw similarity scores  `a(q, kᵢ) = qᵀkᵢ`
2. **Scale:**  divide each score by **√d** (d = Q/K vector dimension)
3. **Softmax:** normalize scaled scores → attention weights (positive, sum to 1)
4. **Sum:**    weighted sum of the Values using those weights → output vector

Formula:  `Attention(q, D) = Σᵢ α(q, kᵢ) · vᵢ`
where     `α(q, kᵢ) = softmax(a(q, kᵢ)) = exp(a(q,kᵢ)) / Σⱼ exp(a(q,kⱼ))`
scaled dot-product scoring:  `a(q, kᵢ) = qᵀkᵢ / √d`

---

## Key intuitions / interview "why" answers

- **Why dot product = similarity?** It's maximal when the two vectors point the same
  direction (also grows with vector magnitude). Big dot product → strong match.

- **Why divide by √d?** The dot product sums d terms, so it grows large just from having
  many dimensions. If scores get too large, **softmax saturates** (nearly all weight on one
  token), which causes **vanishing gradients** and stalls training. Dividing by √d keeps
  scores at a stable scale.

- **What does softmax guarantee, and why does attention need it?**
  - *Sum to 1* → output stays a proper weighted average; its scale stays stable regardless
    of sequence length.
  - *All positive* → each weight is a meaningful "amount of focus," never a subtraction.

- **Two extremes of an attention distribution:**
  - Uniform e.g. `[0.33, 0.33, 0.33]` → model has no idea what's relevant, attends equally
    (maximally *unconfident*).
  - Peaky e.g. `[1.0, 0, 0]` → razor focus on one token (saturated).
  - Healthy attention lives in between.

- **Masked softmax:** restrict softmax to a subset of allowed positions (weights forced to
  0 elsewhere). Used later for causal attention.
- **Additive attention:** alternative scoring `wᵀ tanh(W_q q + W_k k)`; used when Q and K
  have different dimensions.

---

## Quiz result so far

Scored ~5.5/7 on single-head attention. Two gaps closed on re-test:
- Reciting the 4-step engine (Score → Scale → Softmax → Sum) ✅
- The full √d answer (…softmax saturates → vanishing gradients) ✅

**Status: single-head attention fully understood.**

---

## Self-attention

**Cross-attention** (what we saw first): Q comes from one sequence (decoder), K & V from a
*different* sequence (encoder). Two sequences talking to each other.

**Self-attention:** ONE sentence; every word builds its own Q, K, V (via W_Q/W_K/W_V) and
attends to ALL words in the same sentence (including itself). Same engine
(Score → Scale → Softmax → Sum), just Q/K/V all sourced from the same words.

**Why useful — contextualization:** each word rewrites itself using context from the whole
sentence. E.g. "The animal didn't cross the street because **it** was tired" — "it"'s query
matches "animal"'s key strongest, so "it" absorbs "animal"'s value → model learns the
reference. Also disambiguates "bank" (river vs. loan) by which neighbors it attends to.

|                  | Q from            | K, V from              |
|------------------|-------------------|------------------------|
| Cross-attention  | decoder sequence  | encoder (different seq)|
| Self-attention   | the sentence      | the same sentence      |

## Positional encoding (the problem self-attention creates)

**Problem:** self-attention is **permutation invariant** — reorder the words, output is just
reordered the same way; the actual values don't change (a weighted sum ignores order). So
plain self-attention can't tell "dog bites man" from "man bites dog". Fatal for language,
which is order-dependent.

**Fix:** BEFORE attention, **add** a **positional encoding** to each word embedding:
- depends ONLY on the position index (not the word)
- word embedding carries WHO, positional encoding carries WHERE → together = who + where
- added (not concatenated); deck uses **sinusoidal (SinCos)** patterns
- must be added *up front* so position flows through the dot products and shapes the
  attention weights — after attention collapses words into weighted sums, order is gone

**Interview soundbite:** *"Self-attention is permutation-invariant — it treats the input as a
set, not a sequence — so we add positional encodings to the embeddings to give the model
word-order information."*

**Status: self-attention + positional encoding fully understood.**

---

## Multi-head attention

**Limitation of one head:** a single head has one softmax distribution → can capture only
ONE type of relationship well. But words relate in many ways at once (reference, syntax,
semantics).

**Fix:** run the attention engine several times in parallel = "heads". Each head has its OWN
learned **W_Q, W_K, W_V** matrices. Same input word vectors go into every head, but the
different matrices project them into different subspaces → each head learns a different
relationship. Outputs are **concatenated** then mixed by a final linear layer **W_O**.

- The ONE thing that differs between heads = the projection matrices W_Q/W_K/W_V
  (input embeddings are shared).
- **Compute cost ~same as one head:** total dimension d is split across h heads (each head
  works in dimension d/h), e.g. 8 heads × 64 = 512. Multiple perspectives, ~same cost.

**Soundbite:** *"A single head captures one relationship type per position; multi-head runs
several attentions in parallel with their own projections, so heads specialize (reference,
syntax, semantics), then concat + mix. Splitting the dim across heads keeps cost ~constant."*

## The transformer block (the repeatable unit)

Two sub-layers, each wrapped with **residual connection + LayerNorm**:

```
input (embeddings + positional encoding)
  → Multi-Head Self-Attention  → (+ residual) → LayerNorm
  → Feed-Forward Network (MLP) → (+ residual) → LayerNorm
  → output (same shape as input → stack another block)
```

- **Attention = communication** (words exchange info) ; **FFN = computation** (each word
  "thinks" about what it gathered). FFN is applied per-position, same weights every position.
- **Residual** (`out = SubLayer(x) + x`): (1) gradient shortcut → prevents vanishing
  gradients in deep stacks; (2) layer only learns a *change* to input, not a rebuild.
- **LayerNorm:** rescales each word vector to a stable distribution → stable training.
- **Output shape = input shape** → blocks **stack** (GPT-3 = 96 blocks). Depth = where
  representational power comes from (early = surface/grammar, deep = abstract meaning).

**A stack of these blocks IS the transformer. LLMs = this stack trained at scale.**

**Status: multi-head attention + full transformer block fully understood.**

---

## Encoder vs. Decoder + masking

Same transformer block in all three; the ONLY thing that changes is **what each word may
attend to**. That choice determines what the model is good at.

| Type | Attention visibility | Training task | Example | Good for |
|------|---------------------|---------------|---------|----------|
| Encoder-only | bidirectional (whole sentence) | Masked LM (fill blanks) | BERT | understanding (classification, NER) |
| Decoder-only | causal (past words only) | predict next word | GPT | generation |
| Encoder-decoder | encoder bidirectional + decoder causal w/ cross-attention | seq2seq | T5 | translation, summarization |

**Causal masking (GPT):** during training the whole sentence is fed at once, so a word could
"cheat" by seeing the future word it's meant to predict. Fix = before softmax, set all
future-position scores to **−∞** → `exp(−∞)=0` → zero weight → future contributes nothing.
Triangular pattern; info flows forward only ("causal"). Reuses masked-softmax (L19 slide 11).
Deeper why: generation is left-to-right one word at a time, so GPT must be *trained* under the
same past-only condition it faces at generation time.

**Masked Language Modeling (BERT):** a DIFFERENT masking — hide *random* words and predict
them ("The cat [MASK] on the mat" → "sat"). Lets attention stay bidirectional (no future to
hide) while still creating a self-supervised task. BERT never generates, so it's free to look
both ways → richest understanding.

**Soundbite:** *"Same block, only the mask differs. Decoder-only (GPT) = causal mask, sees
only past → generation. Encoder-only (BERT) = bidirectional + masked LM → understanding.
Encoder-decoder = bidirectional encoder + causal decoder that cross-attends → seq2seq."*

**Status: full attention → transformer arc complete (engine → Q/K/V → √d →
self-attention → positional encoding → multi-head → transformer block →
encoder/decoder + masking).**

---

## NEXT UP: L21 Language Models

Builds directly on the arc above. Topics: pretraining objectives (next-token vs. MLM — mostly
already covered), scale / scaling laws (why bigger = better), prompting (zero-/few-shot),
and **RLHF** (the technique behind ChatGPT — the main genuinely new idea).
