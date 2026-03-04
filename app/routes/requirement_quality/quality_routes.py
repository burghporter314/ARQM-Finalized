from flask import Blueprint, jsonify, request, send_file
from app.util.requirement_quality import RequirementQualityAnalyzer
from email.message import EmailMessage
import smtplib
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

limiter = Limiter(
    key_func=get_remote_address
)

requirements_quality_bp = Blueprint('requirements_quality_bp', __name__)

@requirements_quality_bp.route('/analyze-quality/PDF', methods=['POST'])
@limiter.limit("25 per hour")
def analyze_requirements():

    # Get file from form-data
    file = request.files.get('files')
    email_to = request.form.get('email')  # optional

    if not file:
        return jsonify({'error': 'No file included in request'}), 400

    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'File is not a PDF'}), 400

    if not email_to:
        return jsonify({'error': 'Email is required'}), 400

    try:
        # Run your analyzer (this writes ARQM_Report.pdf to disk)
        analyzer = RequirementQualityAnalyzer(file=file)
        analyzer.get_requirement_quality()

        pdf_path = "ARQM_Report.pdf"

        # If email was provided → send email instead of returning file
        if email_to:

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            msg = EmailMessage()
            msg['Subject'] = "Your ARQM Report is Ready"
            msg['From'] = os.getenv("EMAIL_USER")
            msg['To'] = email_to
            msg.set_content("Hello,\n\nYour ARQM report is attached.\n\nBest,\nARQM Team")

            msg.add_attachment(
                pdf_bytes,
                maintype="application",
                subtype="pdf",
                filename="ARQM_Report.pdf"
            )

            # Gmail SMTP (use App Password)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
                smtp.send_message(msg)

            return jsonify({"success": f"PDF emailed to {email_to}"}), 200

        # Otherwise → return file directly (Postman download)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="ARQM_Report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500