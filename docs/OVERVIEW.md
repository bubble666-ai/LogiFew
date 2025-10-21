# Simple Guide to the LogiFew Project

LogiFew is a research project on "deductive reasoning," where the model attempts to derive conclusions from a set of facts and rules, much like a human, and provide its reasoning. In this project, we process the CLEVRER visual/textual data into a more lightweight format and train a neuro-symbolic model (a combination of a neural network and logical rules) using very few examples (few-shot learning).

## What do these scripts do?
- `scripts/build_clevrer_real_subset.py`: Reads the real CLEVRER question and answer files, cleans them, and creates a small, usable training dataset (especially for few-shot scenarios).
- `train.py`: Pre-trains the base model (NSML) on synthetic data we generated (random proofs) to learn general rules.
- `adapt_real.py`: Takes the base model and fine-tunes it on a few examples from the real CLEVRER data (textual questions); we can use a simple BOW or a T5 encoder.
- `eval_fewshot.py` and `scripts/backtest_logifew.py`: Evaluate the trained model in a few-shot setting, reporting accuracy and the quality of the reasoning (proof).

## Real-world Applications
In a more complete version, LogiFew could:
- Work on real video data (using ViT or ResNet features), not just text.
- Verify newly discovered rules with formal tools (like Prover9 or Lean) to ensure reliable output.
- Assist in simple physical analyses (collisions, motion, cause and effect) to explain why an event occurred or what will happen next.
This can be extended to fields like AI education, anomaly detection in scientific data, or even decision-support systems.

