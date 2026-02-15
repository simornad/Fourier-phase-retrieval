# Fourier Phase Retrieval using Deep Learning
פרויקט לשחזור מופע (Phase Retrieval) מבוסס על רשת U-Net ואדפטציה בזמן-אמת (Test-Time Adaptation).

## התקנה
```bash
pip install -r requirements.txt
```

## הרצה

### מצב בדיקה (ברירת מחדל)
ב-`main.py` מוגדר:
```python
TRAIN_MODE = False
```

הרצה:
```bash
python main.py
```

התוצרים נשמרים אוטומטית:
- `results_sample_{idx}.png` לכל דגימה
- `finetuning_loss_sample_{idx}.png` לכל דגימה
- `evaluation_metrics.csv`

### מצב אימון
ב-`main.py` שנה ל:
```python
TRAIN_MODE = True
```

ואז הרץ:
```bash
python main.py
```

תוצרי האימון:
- `best_model_so_far.pth`
- `pretraining_loss_curve.png`

## פלט טרמינל צפוי

אימון:
```text
Using device: cuda
Epoch 01/10 | Train: ... | Val: ... | Best Val: ... | LR: ... | Patience: 0/4
...
Saved best model to: .../best_model_so_far.pth
```

בדיקה:
```text
Using device: cuda
Loaded weights from: .../best_model_so_far.pth
Sample 123 | MSE Pre: ... | MSE FT: ... | Improvement: ...%
...
Saved metrics to: .../evaluation_metrics.csv
```
