import smtplib
from email.mime.text import MIMEText
import playsound

def play_alarm():
    try:
        playsound.playsound("alarm.mp3")
        print("Alarm sound played.")
    except:
        print("Alarm file missing. Add alarm.mp3 in the main folder.")

def send_email_alert():
    sender = "your_email@gmail.com"
    password = "your_app_password"
    receiver = "receiver_email@gmail.com"

    msg = MIMEText("Suspicious activity detected!")
    msg["Subject"] = "Security Alert"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("Email alert sent!")
    except Exception as e:
        print("Failed to send email:", e)

if __name__ == "__main__":
    print("Alert system ready.")
