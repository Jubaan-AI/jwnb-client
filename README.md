# JWNB — Jubaan Weights & Biases (Unofficial)

**JWNB** is a lightweight Python logging and experiment-tracking library inspired by `wandb`, built specifically for Jubaan’s internal back-office system.

It allows you to:
- Track **projects & runs**
- Log **scalars, images, charts, histograms, lists, booleans**
- Capture **system information** automatically
- Upload artifacts (plots, images)
- Maintain a clean experiment timeline per run

Backend: `jwnb.jubaan.com`  
Target audience: Jubaan ML & data teams

---

## ✨ Features

- 📁 Project & run management
- 🔢 Scalar logging (loss, accuracy, metrics)
- 🖼 Image logging (matplotlib, PIL, NumPy)
- 📊 Charts & histograms
- 🧠 Model / config logging (JSON-serializable)
- 🧾 Text, lists & booleans
- 💻 Automatic system info capture (Python, OS, GPU if available)
- 🔔 Notifications on run completion / failure



---
## 🔗 Using JWNB as a Git Submodule

JWNB can be added to your project as a **Git submodule**, allowing you to keep it synchronized with the upstream repository while maintaining full control over when updates are applied.

This is the **recommended integration method** for Jubaan internal projects.

---

### 1️⃣ Add the submodule

From the **root directory of your project**, run:

```bash
git submodule add https://github.com/Jubban-AI/jwnb-client.git external/jwnb


