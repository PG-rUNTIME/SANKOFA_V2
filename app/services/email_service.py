"""
Email service for password-reset notifications.
Falls back to console logging when SMTP is not configured.
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.config import settings

logger = logging.getLogger(__name__)


def _send_smtp(to_email: str, subject: str, html_body: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
        server.ehlo()
        server.starttls()
        if settings.SMTP_USER and settings.SMTP_PASSWORD:
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.EMAIL_FROM, to_email, msg.as_string())


def send_password_reset_email(to_email: str, reset_token: str, full_name: str) -> None:
    """Send a password-reset email. Falls back to logging if SMTP not configured."""
    reset_link = f"{settings.ALLOWED_ORIGINS[0]}/reset-password?token={reset_token}"

    html_body = f"""
    <html>
    <body style="font-family: 'Segoe UI', sans-serif; background:#f5f5f5; padding:30px;">
      <div style="max-width:500px;margin:auto;background:white;border-radius:12px;padding:32px;box-shadow:0 4px 20px rgba(0,0,0,0.08);">
        <div style="text-align:center;margin-bottom:24px;">
          <h1 style="color:#1A2C64;font-size:1.6rem;margin:0;">Sankofa Legal AI</h1>
          <p style="color:#6B7280;font-size:0.85rem;">Password Reset Request</p>
        </div>
        <p style="color:#374151;">Dear {full_name},</p>
        <p style="color:#374151;">
          We received a request to reset the password associated with your Sankofa account.
          Click the button below to set a new password. This link expires in
          {settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES} minutes.
        </p>
        <div style="text-align:center;margin:32px 0;">
          <a href="{reset_link}"
             style="background:#8C4FE3;color:white;padding:14px 32px;border-radius:25px;
                    text-decoration:none;font-weight:600;font-size:1rem;">
            Reset Password
          </a>
        </div>
        <p style="color:#6B7280;font-size:0.82rem;">
          If you did not request a password reset, please ignore this email.
          Your password will remain unchanged.
        </p>
        <hr style="border:none;border-top:1px solid #E5E7EB;margin:24px 0;">
        <p style="color:#9CA3AF;font-size:0.75rem;text-align:center;">
          Sankofa Legal AI — For research purposes only. Not legal advice.
        </p>
      </div>
    </body>
    </html>
    """

    if not settings.SMTP_USER:
        logger.info(
            "SMTP not configured — password reset link for %s: %s", to_email, reset_link
        )
        return

    try:
        _send_smtp(to_email, "Sankofa: Password Reset Request", html_body)
        logger.info("Password reset email sent to %s", to_email)
    except Exception as exc:
        logger.error("Failed to send password reset email to %s: %s", to_email, exc)
        # Don't raise — caller should not reveal email delivery status to client
