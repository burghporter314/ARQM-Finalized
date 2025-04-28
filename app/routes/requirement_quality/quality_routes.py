from flask import Blueprint, jsonify, request, send_file

from app.util.requirement_quality import RequirementQualityAnalyzer

requirements_quality_bp = Blueprint('requirements_quality_bp', __name__)

@requirements_quality_bp.route('/analyze-quality/PDF', methods=['POST'])
def analyze_requirements():
    if request.data:
        requirement_identifier = RequirementQualityAnalyzer(content=request.data)
        requirement_identifier.get_requirement_quality()
        return jsonify({"Success"}), 200
    else:
        if not request.files:
            return jsonify({'error': 'No file included in request'}), 400

        file = next(iter(request.files.values()))
        file_name = file.filename

        if not file:
            return jsonify({'error': 'No file selected for uploading'}), 400

        if not file_name.endswith('.pdf') or file.content_type != 'application/pdf':
            return jsonify({'error': 'File is not a PDF'}), 400

        try:
            requirement_identifier = RequirementQualityAnalyzer(file=file)
            result = requirement_identifier.get_requirement_quality()
            return send_file("../ARQM_Report.pdf", as_attachment=True)
        except Exception as e:
            return jsonify({'error': e}), 500