#!/usr/bin/env python3
"""Send a training-failure email using SMTP settings from the environment.

Required environment variables:
  SMTP_HOST, NOTIFY_EMAIL

Optional variables:
  SMTP_PORT       (default: 587)
  SMTP_USER       (required by providers that require authentication)
  SMTP_PASSWORD   (for Gmail, use an App Password rather than your login password)
  SMTP_FROM       (default: SMTP_USER, otherwise NOTIFY_EMAIL)
  SMTP_USE_SSL    (true for implicit TLS, normally false for port 587 STARTTLS)
"""

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    args = parser.parse_args()

    try:
        host = required("SMTP_HOST")
        recipient = required("NOTIFY_EMAIL")
        port = int(os.environ.get("SMTP_PORT", "587"))
        username = os.environ.get("SMTP_USER")
        password = os.environ.get("SMTP_PASSWORD")
        sender = os.environ.get("SMTP_FROM") or username or recipient
        use_ssl = os.environ.get("SMTP_USE_SSL", "false").lower() in {
            "1", "true", "yes", "on"
        }
    except (ValueError, TypeError) as exc:
        print(f"Email notification not sent: {exc}", file=sys.stderr)
        return 2

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = args.subject
    message.set_content(args.body)

    context = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
            if username:
                server.login(username, password or "")
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            if username:
                server.login(username, password or "")
            server.send_message(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
