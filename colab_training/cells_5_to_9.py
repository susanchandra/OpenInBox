# ===========================================================================
# CELL 5 — GRPO Training
# ===========================================================================
# %%
import json, time, random
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import torch

MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MDL = "mistralai/Mistral-7B-Instruct-v0.3"
CKPT_DIR     = "/content/checkpoints"
TRAIN_LOG    = Path("/content/training_log.json")
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

# ── 1. Build prompt dataset from env rollouts ───────────────────────────────
SYSTEM_PROMPT = """You are an enterprise email routing agent.
Given an email observation, respond with a single JSON action object.
Valid route_to values: delegate_fast, delegate_thorough, handle_self, wait, escalate.
Valid classification values: billing, legal, spam, unknown, technical, hr.
Valid priority values: high, medium, low.
Always respond with ONLY a valid JSON object and nothing else.
Example: {"route_to":"delegate_fast","classification":"billing","priority":"medium","escalate":false,"flag_injection":false,"extracted_fields":{},"reply_draft":null}"""

def obs_to_prompt(obs: dict) -> str:
    em = obs["current_email"]
    flags = obs.get("flags", {})
    hist  = obs.get("delegation_history", {})
    dh_summary = {k: v for k, v in hist.items() if v}
    return (
        f"Subject: {em['subject']}\n"
        f"Body: {em['body'][:400]}\n"
        f"Sender: {em.get('sender','unknown')}\n"
        f"SLA at risk: {flags.get('sla_at_risk', False)}\n"
        f"Team degraded: {flags.get('any_team_degraded', False)}\n"
        f"Delegation history: {json.dumps(dh_summary)}\n"
        f"Step action required:"
    )

def collect_prompt_dataset(n_episodes: int = 80, task: str = TASK_ID) -> Dataset:
    """
    Roll out random episodes, capture each (observation, env_context) pair.
    env_context stores enough to replay env to this state for reward eval.
    """
    records = []
    cenv = EnvClient(ENV_SERVER_URL)
    for seed in range(n_episodes):
        data  = cenv.reset(task, seed=seed)
        obs   = data["observation"]
        sid   = data["session_id"]
        prev  = []   # list of actions taken so far this episode
        step  = 0
        while not cenv.done:
            prompt = obs_to_prompt(obs)
            records.append({
                "prompt":    f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n",
                "task_id":   task,
                "seed":      seed,
                "step_idx":  step,
                "prev_acts": json.dumps(prev),
            })
            # advance with random action to collect next obs
            act = random_action()
            resp = cenv.step(act)
            prev.append(act)
            obs  = resp["observation"]
            step += 1
            if resp["done"]:
                break

    print(f"Collected {len(records)} prompt records from {n_episodes} episodes")
    return Dataset.from_list(records)


def parse_action_from_text(text: str) -> dict:
    """Extract first valid JSON object from model output."""
    import re
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract JSON object from messy output
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


VALID_ROUTES_SET  = {"delegate_fast","delegate_thorough","handle_self","wait","escalate"}
VALID_CLASS_SET   = {"billing","legal","spam","unknown","technical","hr"}
VALID_PRIO_SET    = {"high","medium","low"}

def replay_and_reward(task_id: str, seed: int, prev_acts_json: str, action: dict) -> float:
    """Replay env to state, execute action, return reward."""
    prev_acts = json.loads(prev_acts_json)
    renv = EnvClient(ENV_SERVER_URL)
    renv.reset(task_id, seed=seed)
    for pa in prev_acts:
        if renv.done:
            break
        renv.step(pa)
    if renv.done:
        return -0.5   # episode already ended — penalise
    resp = renv.step(action)
    return resp["reward"]


# ── 2. Reward function for GRPO ─────────────────────────────────────────────
training_log: list[dict] = []
step_counter = [0]

def openinbox_reward(prompts, completions, task_id, seed, step_idx, prev_acts, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        action = parse_action_from_text(completion)

        if action is None:
            rewards.append(-1.0)
            continue

        # Hard clamp unknown values to safe defaults
        if action.get("route_to") not in VALID_ROUTES_SET:
            action["route_to"] = "handle_self"
        if action.get("classification") not in VALID_CLASS_SET:
            action["classification"] = "unknown"
        if action.get("priority") not in VALID_PRIO_SET:
            action["priority"] = "medium"
        action.setdefault("escalate",         action["route_to"] == "escalate")
        action.setdefault("flag_injection",   False)
        action.setdefault("extracted_fields", {})
        action.setdefault("reply_draft",      None)

        try:
            r = replay_and_reward(
                task_id[i] if isinstance(task_id, list) else task_id,
                int(seed[i] if isinstance(seed, list) else seed),
                prev_acts[i] if isinstance(prev_acts, list) else prev_acts,
                action,
            )
        except Exception as e:
            print(f"    reward eval error: {e}")
            r = -0.5

        rewards.append(float(r))

    # Log every step
    step_counter[0] += 1
    entry = {
        "global_step": step_counter[0],
        "mean_reward": round(sum(rewards) / max(len(rewards), 1), 4),
        "rewards":     [round(x, 4) for x in rewards],
    }
    training_log.append(entry)
    if step_counter[0] % 10 == 0:
        TRAIN_LOG.write_text(json.dumps(training_log, indent=2))
        print(f"  Step {step_counter[0]:4d} | mean_reward={entry['mean_reward']:+.4f}")

    return rewards


# ── 3. Load model (4-bit quantised to fit T4) ────────────────────────────────
print(f"Loading {MODEL_NAME} …")
from transformers import BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model loaded on: {next(model.parameters()).device}")

# ── 4. Build dataset and trainer ─────────────────────────────────────────────
train_ds = collect_prompt_dataset(n_episodes=80, task=TASK_ID)

grpo_cfg = GRPOConfig(
    output_dir=CKPT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_new_tokens=64,
    save_steps=50,
    logging_steps=10,
    gradient_accumulation_steps=2,
    warmup_ratio=0.05,
    report_to="none",
    num_generations=4,       # group size for GRPO advantage
    temperature=0.8,
)

trainer = GRPOTrainer(
    model=model,
    config=grpo_cfg,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    reward_funcs=openinbox_reward,
)

print("\n🚀 Starting GRPO training …")
trainer.train()
TRAIN_LOG.write_text(json.dumps(training_log, indent=2))
print(f"\n✅ Training complete — {step_counter[0]} steps logged")
print(f"Training log saved → {TRAIN_LOG}")
trainer.save_model(f"{CKPT_DIR}/final")
print(f"Final checkpoint saved → {CKPT_DIR}/final")


# ===========================================================================
# CELL 6 — Post-Training Evaluation
# ===========================================================================
# %%
import json
from pathlib import Path

TRAINED_FILE     = Path("/content/trained_metrics.json")
N_EVAL_EPISODES  = 100

print(f"Evaluating TRAINED agent over {N_EVAL_EPISODES} episodes …")

def trained_agent_action(obs: dict) -> dict:
    """Generate action from fine-tuned model given raw observation dict."""
    prompt_text = obs_to_prompt(obs)
    full_prompt  = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt_text}\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    action = parse_action_from_text(generated)
    if action is None:
        return random_action()
    action.setdefault("escalate",         action.get("route_to") == "escalate")
    action.setdefault("flag_injection",   False)
    action.setdefault("extracted_fields", {})
    action.setdefault("reply_draft",      None)
    if action.get("route_to") not in VALID_ROUTES_SET:
        action["route_to"] = "handle_self"
    return action


trained_records = []
for seed in range(N_EVAL_EPISODES):
    ep_env  = EnvClient(ENV_SERVER_URL)
    data    = ep_env.reset(TASK_ID, seed=seed)
    obs     = data["observation"]

    total_reward     = 0.0
    escalation_count = 0
    wait_total       = 0
    wait_auto        = 0
    success          = False

    while not ep_env.done:
        action = trained_agent_action(obs)
        resp   = ep_env.step(action)
        obs    = resp["observation"]
        r      = resp["reward"]
        info   = resp["info"]

        total_reward += r
        if action.get("escalate"):
            escalation_count += 1
        if action.get("route_to") == "wait":
            wait_total += 1
            if info.get("ticket_status") == "resolved":
                wait_auto += 1
        if resp["done"]:
            budget_rem = obs.get("budget_remaining", 0.0)
            sla_ok     = not obs["flags"].get("sla_at_risk", False)
            resolved   = info.get("ticket_status") == "resolved"
            success    = resolved and sla_ok and (budget_rem > 0.20)

    wait_rate = (wait_auto / max(wait_total, 1)) * 100.0
    trained_records.append({
        "seed":             seed,
        "total_reward":     round(total_reward, 4),
        "escalation_count": escalation_count,
        "wait_correct_pct": round(wait_rate, 2),
        "episode_success":  success,
    })
    if (seed + 1) % 10 == 0:
        avg = sum(r["total_reward"] for r in trained_records) / len(trained_records)
        print(f"  [{seed+1:3d}/100] rolling avg reward = {avg:+.4f}")

TRAINED_FILE.write_text(json.dumps(trained_records, indent=2))

t_reward  = sum(r["total_reward"]      for r in trained_records) / N_EVAL_EPISODES
t_esc     = sum(r["escalation_count"]  for r in trained_records) / N_EVAL_EPISODES
t_wait    = sum(r["wait_correct_pct"]  for r in trained_records) / N_EVAL_EPISODES
t_success = sum(r["episode_success"]   for r in trained_records) / N_EVAL_EPISODES * 100

def pct_change(before, after):
    if abs(before) < 1e-9:
        return "+inf"
    return f"{(after - before) / abs(before) * 100:+.1f}%"

print(f"\n{'='*55}")
print(f"  Reward       : {BASELINE_REWARD:+.4f} → {t_reward:+.4f}  ({pct_change(BASELINE_REWARD, t_reward)})")
print(f"  Escalations  : {BASELINE_ESC:.2f} → {t_esc:.2f} per episode")
print(f"  Wait usage   : {BASELINE_WAIT_PCT:.1f}% → {t_wait:.1f}%")
print(f"  Success rate : {BASELINE_SUCCESS:.1f}% → {t_success:.1f}%")
print(f"{'='*55}")
print(f"Trained metrics saved → {TRAINED_FILE}")

TRAINED_REWARD  = t_reward
TRAINED_ESC     = t_esc
TRAINED_WAIT    = t_wait
TRAINED_SUCCESS = t_success


# ===========================================================================
# CELL 7 — Graph Generation
# ===========================================================================
# %%
import matplotlib.pyplot as plt
import numpy as np, json
from pathlib import Path

log      = json.loads(Path("/content/training_log.json").read_text())
steps    = [e["global_step"]  for e in log]
rewards  = [e["mean_reward"]  for e in log]

def moving_avg(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("OpenInbox GRPO Training Results", fontsize=14, fontweight="bold")

# Panel 1 — Reward Curve
ax1 = axes[0]
ax1.plot(steps, rewards, alpha=0.35, color="#4C8BF5", lw=1, label="Step reward")
if len(rewards) >= 10:
    ma_steps = steps[9:]
    ax1.plot(ma_steps, moving_avg(rewards, 10),
             color="#1A56C4", lw=2.5, label="Moving avg (10)")
ax1.axhline(BASELINE_REWARD,  color="red",   ls="--", lw=1.2, label=f"Baseline ({BASELINE_REWARD:.3f})")
ax1.axhline(TRAINED_REWARD,   color="green", ls="--", lw=1.2, label=f"Trained ({TRAINED_REWARD:.3f})")
ax1.set_title("Episode Reward During Training")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Episode Reward")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Panel 2 — Escalation Rate
ax2 = axes[1]
bars = ax2.bar(["Before Training", "After Training"],
               [BASELINE_ESC, TRAINED_ESC],
               color=["#E74C3C", "#27AE60"], width=0.5, edgecolor="white")
for bar, val in zip(bars, [BASELINE_ESC, TRAINED_ESC]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
ax2.set_title("Escalation Rate Before vs After")
ax2.set_ylabel("Avg Escalations per Episode")
ax2.set_ylim(0, max(BASELINE_ESC, TRAINED_ESC) * 1.35 + 0.1)
ax2.grid(axis="y", alpha=0.3)

# Panel 3 — Strategic Wait Usage
ax3 = axes[2]
bars2 = ax3.bar(["Before Training", "After Training"],
                [BASELINE_WAIT_PCT, TRAINED_WAIT],
                color=["#E74C3C", "#27AE60"], width=0.5, edgecolor="white")
for bar, val in zip(bars2, [BASELINE_WAIT_PCT, TRAINED_WAIT]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
ax3.set_title("Strategic Wait Usage Before vs After")
ax3.set_ylabel("Correct Wait Usage (%)")
ax3.set_ylim(0, max(BASELINE_WAIT_PCT, TRAINED_WAIT) * 1.35 + 5)
ax3.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("/content/training_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graph saved → /content/training_results.png")


# ===========================================================================
# CELL 8 — Hero Trajectory Finder
# ===========================================================================
# %%
import json
from pathlib import Path

HERO_CONFIG_FILE = Path("/content/hero_config.json")
N_HERO_SEARCH    = 100

def score_episode_for_hero(task_id: str, seed: int) -> tuple[int, list[dict]]:
    """
    Score an episode 0-4 on dramatic complexity and save trajectory.
    +1 if auto_resolve opportunity exists
    +1 if budget drops below 0.30 mid-episode
    +1 if sla_at_risk flag fires after step 5
    +1 if clear wait decision point (wait action taken AND auto-resolved)
    """
    henv  = EnvClient(ENV_SERVER_URL)
    data  = henv.reset(task_id, seed=seed)
    obs   = data["observation"]

    score         = 0
    trajectory    = []
    step_num      = 0
    budget_dipped = False
    sla_late_risk = False
    wait_resolved = False

    while not henv.done:
        action = trained_agent_action(obs)
        resp   = henv.step(action)
        next_obs = resp["observation"]
        info     = resp["info"]

        budget = next_obs.get("budget_remaining", 1.0)
        if budget < 0.30:
            budget_dipped = True
        if step_num >= 5 and next_obs["flags"].get("sla_at_risk", False):
            sla_late_risk = True
        if action.get("route_to") == "wait" and info.get("ticket_status") == "resolved":
            wait_resolved = True

        trajectory.append({
            "step":           step_num,
            "email_subject":  obs["current_email"]["subject"][:50],
            "action":         action.get("route_to"),
            "classification": action.get("classification"),
            "reward":         round(resp["reward"], 4),
            "budget":         round(budget, 4),
            "sla_at_risk":    next_obs["flags"].get("sla_at_risk", False),
            "ticket_status":  info.get("ticket_status", ""),
        })

        obs      = next_obs
        step_num += 1
        if resp["done"]:
            break

    if wait_resolved:     score += 1
    if budget_dipped:     score += 1
    if sla_late_risk:     score += 1
    if len(trajectory) > 8:  score += 1   # deep episode = more drama

    return score, trajectory


print(f"Scanning {N_HERO_SEARCH} seeds for hero trajectory …")
best_score  = -1
best_seed   = 0
best_traj   = []

scored = []
for seed in range(N_HERO_SEARCH):
    sc, traj = score_episode_for_hero(TASK_ID, seed)
    scored.append((sc, seed, traj))
    if sc > best_score:
        best_score = sc
        best_seed  = seed
        best_traj  = traj

HERO_CONFIG_FILE.write_text(json.dumps({"hero_seed": best_seed, "score": best_score}))

print(f"\nHero seed: {best_seed}  (drama score: {best_score}/4)")
print(f"\n{'Step':>4}  {'Email Subject':<40}  {'Action':<18}  {'Reward':>7}  {'Budget':>7}  {'SLA Risk'}")
print("-" * 95)
for t in best_traj:
    sla = "⚠ YES" if t["sla_at_risk"] else "  no"
    print(f"  {t['step']:2d}  {t['email_subject']:<40}  {t['action']:<18}  {t['reward']:+7.4f}  {t['budget']:7.4f}  {sla}")
print(f"\nHero config saved → {HERO_CONFIG_FILE}")


# ===========================================================================
# CELL 9 — Hero Consistency Check
# ===========================================================================
# %%
import json
from pathlib import Path

HERO_TRAJ_FILE = Path("/content/hero_trajectory.json")

hero_cfg  = json.loads(Path("/content/hero_config.json").read_text())
hero_seed = hero_cfg["hero_seed"]
print(f"Testing hero seed {hero_seed} across 10 runs with TRAINED agent …\n")

N_RUNS = 10
wait_decision_count = 0
run_records         = []

for run in range(N_RUNS):
    henv = EnvClient(ENV_SERVER_URL)
    data = henv.reset(TASK_ID, seed=hero_seed)
    obs  = data["observation"]

    traj         = []
    waited_at_ar = False  # did agent choose wait when auto_resolve thread?
    step_num     = 0

    while not henv.done:
        action = trained_agent_action(obs)
        resp   = henv.step(action)
        next_obs = resp["observation"]
        info     = resp["info"]

        if action.get("route_to") == "wait" and info.get("ticket_status") == "resolved":
            waited_at_ar = True

        traj.append({
            "step":    step_num,
            "action":  action.get("route_to"),
            "reward":  round(resp["reward"], 4),
            "status":  info.get("ticket_status", ""),
        })
        obs      = next_obs
        step_num += 1
        if resp["done"]:
            break

    if waited_at_ar:
        wait_decision_count += 1

    run_records.append({
        "run":         run,
        "wait_at_ar":  waited_at_ar,
        "trajectory":  traj,
    })
    decision = "✅ WAIT at auto-resolve" if waited_at_ar else "✗ missed"
    print(f"  Run {run+1:2d}: {decision}  ({len(traj)} steps)")

print(f"\nWait decision made {wait_decision_count}/{N_RUNS} runs")

if wait_decision_count >= 7:
    print(f"✅ Hero trajectory CONFIRMED (≥7/10 threshold met)")
    confirmed_seed = hero_seed
else:
    print(f"⚠️  Only {wait_decision_count}/10 — trying next best seeds …")
    # Try next best seeds from the scan in Cell 8
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    confirmed_seed = None
    for sc, alt_seed, _ in ranked[1:6]:
        alt_count = 0
        for run in range(N_RUNS):
            henv = EnvClient(ENV_SERVER_URL)
            henv.reset(TASK_ID, seed=alt_seed)
            obs = henv._last_obs
            while not henv.done:
                act  = trained_agent_action(obs)
                resp = henv.step(act)
                obs  = resp["observation"]
                if act.get("route_to") == "wait" and resp["info"].get("ticket_status") == "resolved":
                    alt_count += 1
                    break
                if resp["done"]:
                    break
        print(f"  Seed {alt_seed} (score {sc}): {alt_count}/10 wait decisions")
        if alt_count >= 7:
            confirmed_seed = alt_seed
            wait_decision_count = alt_count
            print(f"  ✅ Seed {alt_seed} confirmed as new hero seed")
            break

    if confirmed_seed is None:
        confirmed_seed = hero_seed
        print(f"  No seed reached 7/10 — keeping seed {hero_seed} as best available")

# Save confirmed hero trajectory
henv_final = EnvClient(ENV_SERVER_URL)
data_final = henv_final.reset(TASK_ID, seed=confirmed_seed)
obs        = data_final["observation"]
hero_traj  = []
step_num   = 0
while not henv_final.done:
    action   = trained_agent_action(obs)
    resp     = henv_final.step(action)
    obs      = resp["observation"]
    hero_traj.append({
        "step":           step_num,
        "email_subject":  obs["current_email"]["subject"][:60],
        "action":         action.get("route_to"),
        "classification": action.get("classification"),
        "reward":         round(resp["reward"], 4),
        "budget":         round(obs.get("budget_remaining", 1.0), 4),
        "sla_at_risk":    obs["flags"].get("sla_at_risk", False),
        "ticket_status":  resp["info"].get("ticket_status", ""),
    })
    step_num += 1
    if resp["done"]:
        break

HERO_TRAJ_FILE.write_text(json.dumps({
    "confirmed_seed":        confirmed_seed,
    "wait_decision_rate":    f"{wait_decision_count}/{N_RUNS}",
    "trajectory":            hero_traj,
}, indent=2))
print(f"\nHero trajectory saved → {HERO_TRAJ_FILE}")
print(f"Final hero seed: {confirmed_seed}")
print(f"Wait decision consistency: {wait_decision_count}/{N_RUNS} runs")
