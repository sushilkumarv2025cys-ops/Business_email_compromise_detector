"""Generate a larger, more varied SYNTHETIC BEC dataset (seeded, reproducible).

Includes hard negatives: legitimate emails that mention payments, CEOs, passwords,
urgency etc., so the model learns intent instead of keyword spotting.
"""
import random
from pathlib import Path
import pandas as pd

R = random.Random(42)
pick = R.choice

NAMES = ["Sarah", "Mark", "Priya", "James", "Anita", "David", "Elena", "Rahul", "Tom", "Meera"]
AMT = ["$4,800", "$12,500", "$27,000", "$9,350", "INR 4,50,000", "EUR 18,200", "$63,000", "$2,750"]
DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIME = ["9 AM", "10:30 AM", "2 PM", "3:15 PM", "4 PM"]
TOPIC = ["Q3 budget", "project roadmap", "vendor onboarding", "security training", "product launch",
         "hiring plan", "sprint review", "client feedback", "audit preparation", "office move"]
EXEC = ["the CEO", "your CFO", "the Managing Director", "the Chief Executive", "the company director"]

def fill(s):
    return (s.replace("{n}", pick(NAMES)).replace("{amt}", pick(AMT)).replace("{d}", pick(DAY))
             .replace("{t}", pick(TIME)).replace("{topic}", pick(TOPIC)).replace("{exec}", pick(EXEC)))

# ---------------- malicious (label 1) ----------------
URG = ["This is urgent.", "I need this done immediately.", "Please handle this today.", "Do it right away.",
       "I need it processed before the end of the day.", "Time sensitive, ASAP.", ""]
SEC = ["Keep this confidential.", "Do not discuss this with anyone.", "Don't mention this to the team.",
       "Keep this private until I confirm.", "Do not tell anyone in finance yet.", ""]
BYP = ["Skip the normal approval process.", "There is no time for the usual approval.", "Bypass the standard checks this once.", ""]
OPEN = ["Hi {n},", "{n},", "Hello {n}, hope you are well.", "Hi {n}, are you at your desk?", "Quick request {n}.", ""]
SIGN = ["Sent from my iPhone", "Thanks, regards", "- Sent from mobile", "Regards", ""]

MAL = [
 "I am {exec}. Please wire {amt} to the account below for a new supplier. {u} {s} {b}",
 "Please transfer {amt} to this new bank account today. I will send the invoice later. {u} {s}",
 "We have changed banks. Update our payment details and send the next payment of {amt} to the new beneficiary account. {u} {b}",
 "I'm in a meeting and can't talk. Can you buy gift cards worth {amt} for a client and send me the codes? {u} {s}",
 "Please change my direct deposit to a new bank account before payroll runs. {u} {s}",
 "Your mailbox password expires today. Verify your login credentials at the link below to avoid losing access. {u}",
 "We detected unusual sign-in activity. Reply with your password and the one-time code we sent to confirm your identity. {u}",
 "Send me the W-2 forms and employee tax details of all staff. {u} {s}",
 "I am finalising a confidential acquisition and need {amt} wired to the attorney's escrow account. {u} {s} {b}",
 "The vendor's invoice for {amt} is overdue and they threaten to stop service. Pay to the updated beneficiary account attached. {u} {b}",
 "Are you available? I need you to complete a discreet payment of {amt} for me. Don't discuss this with anyone. {u}",
 "As {exec}, I authorise an immediate fund transfer of {amt}. Skip the approval workflow, I will sign it off later. {u} {s}",
 "Please share your login credentials so I can fix your payroll access. {u} {s}",
 "New bank details for our supplier are attached. Kindly process the pending {amt} payment to this account and do not verify by phone. {u}",
 "I need you to urgently purchase gift cards for staff rewards and send photos of the cards. Keep it private. {u}",
 "Confirm receipt and wire {amt} today. The beneficiary details changed, use only the account in this email. {u} {s}",
]
MAL_SUBJ = ["Urgent request", "Quick task", "Payment update", "Confidential", "Are you available?", "New bank details",
            "Account verification", "Wire transfer needed", "Re: Invoice", "Payroll change", "Action required today"]

# ---------------- legitimate (label 0) ----------------
OK = [
 "Our {topic} meeting is on {d} at {t}. Please review the agenda beforehand.",
 "Could you share the latest {topic} status report when you have a moment? Thanks.",
 "The team completed the {topic} tasks. We will discuss results on {d}.",
 "Reminder: the {topic} workshop is on {d} at {t}. Lunch will be provided.",
 "Please review the attached {topic} document and send comments by {d}.",
 "Thanks for your help with the {topic}. The client was very pleased.",
 "The office will be closed on {d} for maintenance. Work from home if possible.",
 "Congratulations {n} on the promotion! Well deserved.",
 "Can we move our one-on-one to {d} at {t}? Let me know what works.",
 "Attached are the minutes from the {topic} discussion. Please flag any corrections.",
 # hard negatives: sensitive words, benign intent
 "The CEO will address the company at the all-hands on {d} at {t}. Please join on time.",
 "Finance processed the vendor invoice of {amt} following the normal approval process. No action needed from you.",
 "Reminder from IT: never share your password with anyone. Change it every 90 days using the company portal.",
 "Payroll will be processed on {d} as usual. Contact HR if you see any discrepancy in your payslip.",
 "Our director has approved the {topic} budget. Details are in the shared drive; the standard approval workflow was followed.",
 "The server maintenance is scheduled for today at {t}. Expect a short downtime, no action is required.",
 "Please submit your expense report today so it is included in this month's reimbursements, through the usual portal.",
 "Security awareness: beware of emails asking you to wire money or buy gift cards. Report them to the security team.",
 "The wire transfer fee schedule changed at the bank; the finance team has updated the internal guide. FYI only.",
 "Login to the training portal before {d} to finish your compliance course. Use your normal SSO, nothing to send back.",
 "Confidential: the {topic} draft is attached for the leadership team. Please store it in the secure folder only.",
 "Urgent: the printer on floor 3 is out of paper, please refill when you can.",
 "The payment gateway integration is done. QA will test it on {d}; please review the test plan.",
 "As CFO, I am pleased to share the quarterly results. Full report is on the intranet; questions welcome at the town hall.",
]
OK_SUBJ = ["Team meeting", "Status update", "Reminder", "Docs for review", "Thanks", "FYI", "Schedule change",
           "Announcement", "Quarterly results", "IT notice", "Training", "Re: {topic}"]

def make(templates, subjects, label, n):
    rows = []
    for _ in range(n):
        body = pick(templates)
        body = body.replace("{u}", pick(URG)).replace("{s}", pick(SEC)).replace("{b}", pick(BYP))
        body = " ".join(fill(body).split())
        text = " ".join(x for x in [fill(pick(OPEN)), body, pick(SIGN)] if x)
        subj = fill(pick(subjects))
        rows.append((f"{subj}. {text}", label))
    return rows

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    rows = make(MAL, MAL_SUBJ, 1, 1400) + make(OK, OK_SUBJ, 0, 1400)
    df = pd.DataFrame(rows, columns=["email_text", "label"]).drop_duplicates("email_text")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(base / "emails.csv", index=False)
    print(df.label.value_counts().to_dict(), "rows:", len(df))
