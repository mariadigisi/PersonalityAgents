# Grounding Personality in LLM Agents: Behavioral and Cognitive Validation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TinyTroupe-orange.svg)](https://github.com/microsoft/tinytroupe)
[![Model](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-green.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"Grounding Personality in LLM Agents: Behavioral and Cognitive Validation"**  
> *Maria Di Gisi, Giuseppe Fenza, and Mariacristina Gallo* (IMT School for Advanced Studies Lucca & University of Salerno).

---

## 📌 Overview

Large Language Model (LLM) agents are increasingly used as synthetic human proxies in social science, economics, and security. However, conventional agent persona prompting suffers from **personality-label leakage** (where models pull stereotypes from explicit labels like `"INTJ"`) and lack **psychometric grounding**.

This repository implements a framework for **grounding and psychometrically validating** LLM agents. Using **TinyTroupe** and **Google Gemini 2.5 Flash**, we instantiate all **16 MBTI personality profiles** using purely observable behavioral trait descriptions—**completely avoiding explicit psychological labels**. 

To demonstrate real-world applicability, we execute an adversarial **multi-agent phishing susceptibility benchmark** where synthetic victims interact with attacker agents leveraging 7 Cialdini-based persuasion levers.

---

## 🌟 Key Contributions

1. **Blind Behavioral Grounding:** 16 MBTI persona profiles instantiated strictly via natural language descriptions of observable behaviors without explicit psychological terms.
2. **Psychometric Validation (100% Accuracy):** A rigorous manipulation check using a 60-item Likert assessment confirming that agents reproduce their target personality profiles without label leakage.
3. **Adversarial Phishing Simulation:** Multi-agent experiments ($N=560$ total agent trials across 5 independent runs) evaluating psychological susceptibility across 7 persuasion levers.
4. **Cognitive Reasoning Pattern Analysis:** Topic modeling (BERTopic) on free-text justifications reveals distinct cognitive vulnerability pathways (e.g., *Procedural Compliance* in Judging profiles vs. *Loss/Urgency Pressure* in Perceiving profiles).

---

## 📊 Benchmark Key Findings

Across 5 independent experimental runs ($N=560$ trials):

* **Overall Profile Vulnerability ($V_p$):** **ENFP** exhibited the highest susceptibility (**74.29%** click rate), followed by **ESFP** (**68.57%**) and **ESTP** (**62.86%**). **INTJ** and **INFJ** demonstrated the highest structural resistance (**8.57%** click rate).
* **Dimensional Disparities:** Extraverted ($E$) agents were over **twice as vulnerable** as Introverted ($I$) agents (**47.50% vs. 22.14%**). Sensing ($S$) profiles (**44.29%**) were more susceptible than Intuitive ($N$) profiles (**25.36%**).
* **Persuasion Levers ($E_l$):** **Social Proof** was the most effective vector (**57.50%** success rate), followed by **Curiosity** (**46.25%**). **Scarcity** was the least effective (**10.00%**).

---

## 📁 Repository Structure

```text
.
├── mbti_tinytroupe_agents.py # Definitions of 16 behaviorally grounded MBTI agents
├── mbti_phishing_attacker.py # Attacker agent generating Cialdini email vectors
├── mbti_phishing_victim.py   # Victim agent evaluation and trial isolation logic
├── phishing_simulation.py    # Main orchestration runner and analytics exporter
├── requirements.txt          # Python dependencies
└── README.md                 # Repository documentation

```

If you use this framework or codebase in your research, please cite our paper:

```text
@inproceedings{digisi2026grounding,
  author    = {Di Gisi, Maria and Fenza, Giuseppe and Gallo, Mariacristina},
  title     = {Grounding Personality in LLM Agents: Behavioral and Cognitive Validation},
  booktitle = {Proceedings of the 2nd International Workshop on LLM-Driven Agents (LLMA 2026), held in conjunction with the 27th International Conference on Web Information Systems Engineering (WISE 2026)},
  year      = {2026},
  address   = {Venice, Italy},
  month     = {November}
}
```
