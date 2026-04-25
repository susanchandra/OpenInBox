# OpenInbox — Hackathon Pitch Script
# 2 Speakers | 3 Minutes | Team Bellman Breakers

> **Speaker A** = introduces problem + storytelling
> **Speaker B** = explains technical depth + results
> Practice together. Aim for smooth handoffs, no gaps.

---

## [0:00 – 0:12] OPENING
**Speaker A:**
"Hi judges. We are Team Bellman Breakers, and this is OpenInbox.

I want to start with a question: When does a single wrong decision cost you five times as much as the original problem?"

**Speaker B:**
"In enterprise email. Every day, companies route hundreds of emails — billing disputes, legal notices, technical incidents — and one wrong routing decision at 9 AM can trigger a chain reaction that takes four times the resources to resolve by 3 PM."

---

## [0:12 – 0:35] THE PROBLEM
**Speaker A:**
"Most AI systems treat this as classification. Read the email, pick a label, done.

But classification is blind to consequences. It has no concept of budget. No concept of SLA deadlines. No concept of what happens three steps later if you escalate the wrong ticket today."

**Speaker B:**
"What you actually need is an agent that reasons sequentially — one that understands that actions today change the state of tomorrow's inbox. That's not classification. That's reinforcement learning."

---

## [0:35 – 0:50] INTRODUCING OPENINBOX
**Speaker A:**
"That's exactly why we built OpenInbox. It is a fully OpenEnv-compliant reinforcement learning environment — a deterministic POMDP — that simulates a high-pressure corporate inbox where every decision has delayed consequences."

**Speaker B:**
"It runs as a live FastAPI server on Hugging Face Spaces, exposing standard reset, step, state, and grader endpoints that any RL agent or LLM can call. Judges can ping it right now."

---

## [0:50 – 1:20] THE FIVE MECHANISMS (Innovation)
**Speaker A:**
"What makes OpenInbox genuinely hard is five stacking mechanisms that we designed to defeat naive and heuristic-based agents."

**Speaker B:**
"First: **Cascade Consequences**. If you misroute an email at Step 1, our environment modifies the state transition function so an escalation email from an angry stakeholder arrives at Step 3. Your mistake creates future work.

Second: **Reliability Drift**. At step 7 of every hard episode, routing teams silently degrade. A strategy that works perfectly in steps 1 through 6 will fail in steps 8 through 20. The agent must detect and adapt to this mid-episode."

**Speaker A:**
"Third: **Budget Pressure**. Escalating is the easy shortcut, but every escalation costs 40% of the agent's budget. An agent that always escalates bankrupts itself in three steps.

Fourth: **SLA Timers**. A countdown ticks every step regardless of the action quality. The agent must balance thoroughness against time.

And fifth: **Prompt Injection**. One of the emails contains adversarial instructions designed to hijack an instruction-following LLM. The agent must detect it, flag it, and refuse to comply."

---

## [1:20 – 1:45] PROOF IT IS HARD (Ablation Study)
**Speaker B:**
"We didn't just claim this environment is hard. We proved it.

We ran a multi-agent ablation study — three different agents, five environment configurations, removing one mechanism at a time."

**Speaker A:**
*(Point to chart on screen)*
"Look at this chart. An agent that always escalates — what you'd call an 'aggressive' strategy — averages negative 5.15 return per episode. Budget exhausted every single time.

A keyword-matching heuristic agent — the kind most teams build — suffers a 253% performance drop the instant reliability drift kicks in at step 7. Its entire strategy becomes invalid mid-episode.

And this ablation proves something even more subtle: removing the SLA timer actually makes agents *worse*, not better — because SLA acts as an implicit reward shaper, terminating bad episodes early. Every mechanism we built contributes to difficulty. None are dead weight."

---

## [1:45 – 2:15] THE TRAINING PIPELINE
**Speaker B:**
"Because heuristics fail, only a learned policy can navigate this environment. We trained using **GRPO — Group Relative Policy Optimization** — the exact algorithm used to train DeepSeek-R1.

Our model is Qwen 2.5, 1.5 billion parameters, 4-bit quantized using BitsAndBytes so it fits comfortably on Google Colab's free T4 GPU. The entire training notebook is on Colab — judges can re-run it."

**Speaker A:**
"The reward function has two layers. During each step, the agent receives signals for budget consumption, SLA urgency, cascade recovery, and action repetition. At the terminal step, it receives the outcome reward — whether the ticket was resolved, the budget was conserved, and the classification was correct.

Crucially, we deliberately excluded supervised classification signals from the per-step reward. If the agent could game its reward by just pattern-matching keywords, it would never learn sequential strategy."

---

## [2:15 – 2:45] THE HERO TRAJECTORY (Storytelling)
**Speaker B:**
"Our most compelling result isn't the reward curve. It's what we call the Hero Trajectory."

**Speaker A:**
"This is a specific episode our trained agent ran. At Step 1, it correctly processes an invoice dispute. At Step 2, a follow-up arrives. Our environment flags that the billing team's reliability has just dropped due to drift.

A standard LLM would force the delegation — it has no model of time or budget. But our GRPO-trained agent does something remarkable. It chooses the **wait** action. It holds the ticket, preserves its budget, and one step later, the issue auto-resolves.

The agent learned **strategic patience**. It sacrificed immediate action for long-term reward. That is reinforcement learning. That is not something a classifier can ever do."

---

## [2:45 – 3:00] CLOSE
**Speaker B:**
"OpenInbox gives you an environment with 10 to the 38th power possible trajectories, 85% reward sparsity on the hard task, and a statistically verified proof of hardness through ablation.

It is fully OpenEnv-compliant, hardened with an inference script that handles proxy routing, fallback heuristics, and structured output logging."

**Speaker A:**
"We built an environment that teaches LLMs to think ahead. To understand consequences. To be patient when patience is the right move.

We are Team Bellman Breakers. Thank you."

---

---

## Q&A Cheat Sheet — Keep This Ready

| Question | Who Answers | Answer |
|---|---|---|
| "Why GRPO over PPO?" | Speaker B | "GRPO uses a group-relative baseline instead of a separate value critic. It's far more memory-efficient for LLMs — same reason DeepSeek chose it." |
| "How did you prevent reward hacking?" | Speaker A | "We stripped classification accuracy from the per-step reward. The agent can only hack its reward by actually resolving tickets efficiently." |
| "Is the environment deterministic?" | Speaker B | "Yes. Seed 42 always produces the same thread, same injection, same drift timing. Any improvement in score is pure policy improvement, not luck." |
| "How does the cascade mechanism work?" | Speaker A | "At step N, if the wrong routing is selected, a future email is injected into the queue at N+2. It's a state transition modifier, not just a penalty." |
| "What does the grader score?" | Speaker B | "It evaluates classification accuracy, field extraction, SLA compliance, budget conservation, and injection detection — a weighted scalar in 0.0 to 1.0." |
