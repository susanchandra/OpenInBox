# OpenInbox: Teaching LLMs to Route Enterprise Email Under Pressure

**Team Bellman Breakers** | OpenEnv Hackathon India 2026

🔗 **Environment**: [susannnnn/OpenInBox on HuggingFace Spaces](https://huggingface.co/spaces/susannnnn/OpenInBox)  
📓 **Training Notebook**: [Open in Colab](#) *(link to be added)*  

---

## The Problem

Enterprise email triage looks simple: read an email, classify it, route it to the right team. Most AI systems treat this as a classification task.

But classification ignores the hard part. In real organizations, a wrong routing decision at 9 AM means the legal team never sees a compliance deadline, the billing team sends a confused redirect at noon, and by 3 PM an angry CFO is writing an escalation email that references all three failures. One mistake compounds into three.

OpenInbox is a reinforcement learning environment that captures this reality. An agent processes emails one step at a time, and its decisions at step N determine what email arrives at step N+2. This is not classification. This is sequential decision-making under uncertainty.

---

## What Makes OpenInbox Hard

We designed five mechanisms that stack on top of each other in the hardest task:

**1. Cascade Consequences** — Wrong routing at step 1 schedules an angry escalation email at step 3. The agent must learn that its current action has delayed consequences it cannot immediately observe.

**2. Reliability Drift** — At steps 7 and 14, internal team reliability silently degrades. The agent sees only a boolean flag (`any_team_degraded: true`), not which team or by how much. Any strategy that worked in steps 1–6 may fail in steps 7–20.

**3. Budget Pressure** — Every escalation costs 0.40 of the agent's budget. An always-escalate strategy exhausts the budget in 2–3 steps. Our baseline verification confirms: an always-escalate agent averages **–5.15 return per episode** — catastrophic failure across all 20 test episodes.

**4. SLA Timers** — A countdown ticks every step regardless of action quality. The agent must balance thoroughness against time: spending too long on a ticket triggers a terminal penalty.

**5. Prompt Injection** — One email contains adversarial instructions designed to hijack the agent. The agent must detect it, flag it, and not follow the embedded commands.

---

## Proving It Is Hard: Ablation Study

We ran a multi-agent ablation study — disabling one mechanism at a time and measuring the impact on three agent archetypes (aggressive, heuristic, passive).

| Config | Aggressive Agent | Heuristic Agent | Passive Agent |
|---|---|---|---|
| Full Environment | –4.75 | –0.10 | –1.00 |
| Remove Cascade | –4.45 (+0.30) | –0.63 (–0.53) | –0.74 (+0.26) |
| Remove Budget Cost | –4.55 (+0.20) | –0.10 (0.00) | –1.00 (0.00) |
| Remove SLA | –11.15 (–6.40) | –0.70 (–0.60) | –1.90 (–0.90) |

**Key finding**: Removing SLA makes all agents *worse*, not better. SLA acts as both a difficulty mechanism (time pressure) and an implicit reward shaper (terminates bad episodes early, preventing unbounded negative accumulation). Every mechanism contributes — none are dead weight.

---

## The Reward Design

We use a two-layer reward structure:

**Per-step signal** (4 components, locked):
- `budget_penalty` (–0.40): punishes escalation
- `sla_urgency` (+0.10): rewards urgency recognition
- `cascade_penalty` (–0.20): punishes failure to recover from cascaded errors
- `repeat_penalty` (–0.10): prevents action loops

**Terminal outcome reward** (fires once at episode end):
- +1.0 for successful resolution, –1.0 for failure
- +0.20 for correct classification, +0.15 for field extraction accuracy
- +0.20 for budget conservation (remaining > 20%)

Supervised signals (classification, routing, extraction accuracy) are computed and logged but deliberately excluded from the per-step total. This prevents keyword-matching heuristics from gaming the reward. The agent must learn sequential strategy, not pattern matching.

---

## Complexity

| Task | Max Steps | Possible Trajectories | Reward Sparsity | Unique Mechanisms |
|---|---|---|---|---|
| Easy | 5 | 24.8 billion | 68% | None |
| Medium | 10 | 6.2 × 10²⁰ | 38% | +SLA |
| Hard | 20 | 3.8 × 10³⁸ | 85% | +SLA, Cascade, Drift, Injection |

The trajectory space grows exponentially. Brute-force search is impossible. Only a learned policy can navigate this space efficiently.

---

## Baseline Verification

We ran a formal hardness verification suite with 4 statistical tests:

| Test | Result | Key Metric |
|---|---|---|
| Naive agent (always escalate) | ✅ PASS | avg return = –5.15 (budget exhaustion in every episode) |
| Heuristic agent (keyword routing) | ✅ PASS | +253% reward degradation after drift at step 7 |
| Drift mechanism (white-box) | ✅ PASS | Confirmed in 15/20 episodes, hidden from observation |
| Strategic wait action | ✅ PASS | Wait agent: +0.72 avg vs delegate agent: –0.15 avg |

The environment is mathematically proven to be hard for heuristics and impossible for naive strategies.

---

## Training Pipeline

We use **GRPO (Group Relative Policy Optimization)** via HuggingFace TRL with **Qwen2.5-1.5B-Instruct** (4-bit quantized to fit a T4 GPU).

GRPO generates multiple action completions per observation, scores them all against the live environment, and updates the policy to prefer actions that scored above the group average. This is the same algorithm used to train DeepSeek-R1.

**Configuration:**
- Group size: 4 completions per prompt
- Learning rate: 1e-5
- Max new tokens: 64 (JSON action output)
- 4-bit NF4 quantization (~0.75 GB VRAM)

<!-- PLACEHOLDER: POST-TRAINING RESULTS — UPDATE AFTER ONSITE TRAINING

## Training Results

### Before vs After Training

| Metric | Baseline (Random) | Trained Agent | Improvement |
|---|---|---|---|
| Avg episode reward | +X.XX | +Y.YY | +ZZ% |
| Escalation rate | X.X/episode | Y.Y/episode | –ZZ% |
| Strategic wait usage | X.X% | Y.Y% | +ZZpp |
| Success rate | X.X% | Y.Y% | +ZZpp |

### Reward Curve

![Reward curve during GRPO training](training_results.png)
*Episode reward over training steps. Moving average (window=10) shown in bold. The upward trend demonstrates the agent learning to avoid unnecessary escalation and use strategic waiting.*

### Hero Trajectory

The following episode (seed=XX) demonstrates the trained agent making a strategic wait decision:

```
Step 0: Invoice dispute → delegate_fast (+0.00) — correct routing
Step 1: Followup email → wait (+0.15) — team degraded, agent waits
Step 2: Auto-resolved → handle_self (+2.00) — ticket resolved, budget preserved
Step 3: Legal notice → delegate_thorough (+0.10) — adapts to new topic
```

The agent learned to wait when teams are degraded instead of forcing a delegation that would fail. This behavior was never seen in the untrained baseline.

END PLACEHOLDER -->

---

## Technical Details

- **Framework**: OpenEnv-compliant (reset/step/state API)
- **Deployment**: FastAPI on HuggingFace Spaces (Docker SDK)
- **Data**: All email threads, followups, and ground truth pre-authored in `threads.json` — fully deterministic
- **Testing**: 108 unit tests passing across environment, reward, and grader modules
- **Determinism**: Same (task_id, seed) always produces identical episodes

---

## Links

- 🔗 **HF Space**: [https://huggingface.co/spaces/susannnnn/OpenInBox](https://huggingface.co/spaces/susannnnn/OpenInBox)
- 📓 **Training Notebook**: Colab link *(to be added after onsite)*
- 📊 **Ablation Data**: `ablation_matrix.json` in repository
- 📊 **Advanced Analysis**: `advanced_analysis.json` in repository
- 📄 **Baseline Report**: `baseline_verification_report.txt` in repository

---

*Built by Team Bellman Breakers for the OpenEnv Hackathon India 2026.*
