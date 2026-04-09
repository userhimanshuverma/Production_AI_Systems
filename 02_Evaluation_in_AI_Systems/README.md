# Day 2 — Defining Good: Evaluation in AI Systems

> "If you can't measure it, you can't improve it. If you measure the wrong thing, you'll improve the wrong thing."

---

## Why Evaluation is Hard in AI Systems

In traditional software, evaluation is straightforward. Given input X, expect output Y. Pass or fail.

AI systems don't work that way.

**Probabilistic outputs** — the same question asked twice can return two different answers. Both might be acceptable. Neither might be perfect. There's no single correct string to compare against.

**Subjective correctness** — "Is this a good summary?" depends on who's reading it, what they needed, and what they already knew. There's no ground truth that works for everyone.

**The "looks correct but is wrong" problem** — this is the dangerous one. A model can return a fluent, confident, well-structured answer that is factually wrong. No exception is raised. No error is logged. The system thinks it succeeded. The user got misinformation.

This is why evaluation in AI systems is a first-class engineering concern, not an afterthought.

---

## What Does "Good" Mean? Core Metrics

Before you can evaluate, you need to define what you're measuring.

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| Correctness | Is the answer factually accurate? | Core quality signal |
| Relevance | Does the answer address the question? | A correct answer to the wrong question is useless |
| Consistency | Does the system give the same answer to the same question? | Unpredictability erodes trust |
| Latency | How long does it take to respond? | Slow correct answers still fail users |
| Cost | How much does each response cost? | Unsustainable systems get shut down |
| Failure rate | How often does the system error, timeout, or return nothing? | Reliability baseline |

No single metric tells the full story. A system can score high on correctness but fail on latency. High on relevance but inconsistent across runs. You need all of them, tracked together.

---

## Types of Evaluation

### a) Offline Evaluation

Run before deployment. Controlled environment. Known inputs and expected outputs.

- **Benchmark datasets** — curated question/answer pairs where you know the correct answer. Useful for regression testing: did the last change make things better or worse?
- **Controlled testing** — fixed prompts, fixed retrieval context, fixed model. Isolates one variable at a time.

Limitation: your benchmark is never the full distribution of real user inputs. A system that scores 95% on your benchmark can still fail badly in production on inputs you didn't anticipate.

### b) Online Evaluation

Runs in production. Measures real behavior with real users.

- **Real user feedback** — thumbs up/down, corrections, follow-up questions. Noisy but honest. Users tell you what your benchmark missed.
- **A/B testing** — route a percentage of traffic to a new version. Compare metrics between versions on identical user populations. The most reliable way to know if a change actually helped.

Limitation: you need volume. A/B testing on 50 users per day gives you weak signal.

### c) Human Evaluation

A human reviews outputs and scores them. Gold standard for quality — humans catch nuance that automated metrics miss.

Pros:
- catches subtle errors (tone, factual gaps, misleading framing)
- can evaluate things that are hard to automate (helpfulness, safety)

Cons:
- slow and expensive
- doesn't scale
- inter-annotator disagreement — two humans often score the same output differently

Use human evaluation for calibration and spot-checking, not as your primary production signal.

### d) LLM-as-a-Judge

Use a second LLM to evaluate the output of your primary LLM. Send the question, the generated answer, and optionally a reference answer to a judge model. Ask it to score correctness, relevance, or hallucination.

How it works:
```
Question + Generated Answer + Reference Answer
              │
              ▼
         Judge LLM
              │
              ▼
    Score (0–10) + Reasoning
```

Limitations:
- the judge inherits the same biases as the model being judged
- judge models tend to prefer longer, more confident-sounding answers
- not reliable for factual verification — the judge can be wrong too
- adds latency and cost to every evaluation call

LLM-as-a-judge is useful at scale when human evaluation isn't feasible, but it should be calibrated against human judgments first.

---

## Architecture: Evaluation in Production

Evaluation isn't a one-time step. It runs continuously alongside your system.

```
User
  │
  ▼
AI System (Retrieval + LLM + Post-processing)
  │
  ▼
Output ──────────────────────────────────────────┐
  │                                              │
  ▼                                              │
Evaluation Layer                                 │
  ├── Automated scoring (relevance, format)      │
  ├── LLM-as-a-judge (async, sampled)            │
  ├── User feedback capture                      │
  └── Failure detection (empty, timeout, error)  │
  │                                              │
  ▼                                              │
Logging & Tracing Store                          │
  │                                              │
  ▼                                              │
Metrics Dashboard ◄──────────────────────────────┘
  ├── Correctness trend over time
  ├── Latency p50 / p95 / p99
  ├── Failure rate by query type
  └── Cost per query
```

The evaluation layer runs asynchronously — it doesn't block the response to the user. Scores are computed in the background and fed into dashboards and alerting.

---

## Example: Evaluating a RAG System

A RAG system has two places where things can go wrong: retrieval and generation. You need to evaluate both independently.

**Retrieval quality**
- Did the retriever return documents relevant to the question?
- Was the correct document in the top-k results?
- Metric: recall@k — how often is the relevant document in the top k retrieved chunks?

**Answer correctness**
- Does the generated answer match the reference answer?
- Is the answer grounded in the retrieved context, or did the model go off-script?

**Hallucination detection**
- Does the answer contain claims not supported by the retrieved context?
- This is the hardest to automate — LLM-as-a-judge is commonly used here, with known limitations.

---

## Python Example: LLM-as-a-Judge

```python
import openai
import json
import logging

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """
You are an evaluation assistant. Score the following answer on two dimensions:

Question: {question}
Generated Answer: {generated}
Reference Answer: {reference}

Score each dimension from 0 to 10:
- correctness: Is the generated answer factually accurate compared to the reference?
- relevance: Does the generated answer actually address the question?

Respond in JSON only:
{{"correctness": <score>, "relevance": <score>, "reasoning": "<one sentence>"}}
"""

def evaluate_answer(question: str, generated: str, reference: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        generated=generated,
        reference=reference
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            timeout=15
        )
        raw = response.choices[0].message.content
        result = json.loads(raw)
        logger.info(f"Evaluation result: {result}")
        return result

    except json.JSONDecodeError:
        logger.error("Judge returned non-JSON output")
        return {"correctness": -1, "relevance": -1, "reasoning": "parse error"}
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"correctness": -1, "relevance": -1, "reasoning": "evaluation error"}


# Usage
if __name__ == "__main__":
    result = evaluate_answer(
        question="What is the capital of France?",
        generated="The capital of France is Paris, which is also its largest city.",
        reference="Paris is the capital of France."
    )
    print(result)
    # {"correctness": 10, "relevance": 10, "reasoning": "Answer is accurate and directly addresses the question."}
```

Key points in this implementation:
- JSON-only output makes parsing reliable
- Handles parse errors and API failures gracefully
- Logs every result for trend analysis
- Returns `-1` scores on failure so dashboards can flag broken evaluations separately from low-quality ones

---

## Best Practices

- **Evaluate continuously** — not just before deployment. Model behavior can drift as data changes even if the model doesn't.
- **Combine multiple signals** — automated scores + user feedback + sampled human review. Each catches different failure modes.
- **Don't rely on a single metric** — a system optimized for one metric will sacrifice others. Track the full picture.
- **Track failures explicitly** — log timeouts, empty responses, parse errors, and refusals as their own category. They're not the same as low-quality answers.
- **Sample for human review** — you can't review everything, but reviewing 1% of production traffic consistently is far better than reviewing nothing.
- **Calibrate your judge** — before trusting LLM-as-a-judge at scale, validate its scores against human judgments on a sample set.

---

## Common Mistakes

**Only checking accuracy**
Accuracy on a benchmark tells you almost nothing about production behavior. Real users ask questions your benchmark never covered.

**Ignoring user experience**
A technically correct answer that's too long, too vague, or poorly formatted still fails the user. Evaluation needs to include usability signals, not just factual correctness.

**No monitoring in production**
Evaluating before deployment and then going dark is the most common mistake. Systems degrade over time — data changes, usage patterns shift, edge cases accumulate. Without continuous monitoring, you won't know until users complain.

**Treating evaluation as a one-time gate**
Evaluation is not a checkbox before launch. It's an ongoing system that runs alongside your AI in production.

---

## Summary

Evaluation in AI systems is hard because there's no single correct answer, outputs are probabilistic, and failures are often silent.

Good evaluation combines:
- offline benchmarks for regression testing
- online signals from real users
- automated scoring for scale
- human review for calibration
- continuous monitoring in production

The goal isn't a perfect score on a benchmark. The goal is a system you can trust — and a system you can see clearly enough to improve.

> You don't know if your AI system is working. You only know if you're measuring it.
