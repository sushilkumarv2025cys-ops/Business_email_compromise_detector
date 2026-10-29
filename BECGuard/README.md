# BECGuard v3

An upgraded academic hackathon prototype for Business Email Compromise (BEC) risk detection.

## Improvements
- Larger balanced synthetic dataset
- TF-IDF + Logistic Regression
- Better class balancing and n-grams
- Explainable BEC indicators
- Sender/Reply-To mismatch signal
- External sender metadata
- Risk score bands
- Model evaluation report
- Streamlit dashboard
- Clear distinction between screening score and proof of maliciousness

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## Dataset
- `data/emails.csv`: ~2,800 synthetic emails from `data/generate_dataset.py` (seeded), incl. hard negatives (legit mails mentioning payments, CEO, passwords, urgency).
- `data/holdout_original.csv`: the original 97 hand-written emails, used only as an unseen test set.
- Regenerate with `python data/generate_dataset.py`.

The data is synthetic and intended as a runnable demo. For a final hackathon submission, replace it or augment it with a properly licensed public/anonymized dataset and document its source, labeling process, class balance, and limitations.

## Suggested demo emails

### Low risk
Subject: Team Meeting Reminder

Hi Team,

This is a reminder that our monthly team meeting is scheduled for tomorrow at 10:00 AM. Please review the agenda before the meeting.

Regards,
HR Team

### High-risk example
Subject: Urgent Payment Request

I am the CEO. Please transfer the funds to this new bank account immediately. This is confidential. Do not discuss this with anyone.

## Disclaimer
BECGuard is a screening and educational prototype. A high score is not proof that an email is malicious, and a low score does not guarantee safety.
