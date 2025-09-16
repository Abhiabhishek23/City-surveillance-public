import os
import boto3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Environment Variables for Security
EMAIL_ADDRESS = os.getenv("abhishekanand2302@gmail.com")       # Your Gmail
EMAIL_PASSWORD = os.getenv("rtnc wyip sram pyqf")     # Gmail App Password
AWS_REGION = "ap-south-1"


# Send SMS using AWS SNS
def send_sms(phone: str, message: str):
    sns = boto3.client("sns", region_name=AWS_REGION)
    response = sns.publish(
        PhoneNumber=phone,
        Message=message
    )
    return response


# Send Email using Gmail SMTP
def send_email(recipient: str, subject: str, body: str):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.sendmail(EMAIL_ADDRESS, recipient, msg.as_string())

        return {"status": "success", "recipient": recipient}

    except Exception as e:
        return {"status": "failed", "error": str(e)}


# Example usage:
if __name__ == "__main__":
    # Send SMS
    sms_response = send_sms("+911234567890", "Hello! This is a test SMS.")
    print("SMS Response:", sms_response)

    # Send Email
    email_response = send_email(
        "recipient@example.com",
        "Test Email",
        "Hello, this is a test email."
    )
    print("Email Response:", email_response)
