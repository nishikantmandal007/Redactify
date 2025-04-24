#!/usr/bin/env python3
# Redactify/web/forms.py

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, SelectMultipleField, TextAreaField, BooleanField, widgets
from wtforms.validators import InputRequired, Optional, ValidationError
import re  # Import re for custom validator


# Custom validator to check if each line is a valid regex
def validate_each_line_regex(form, field):
    """Checks if each non-empty line in the TextAreaField is a valid regex."""
    if not field.data:
        return  # No validation needed if empty

    lines = field.data.splitlines()
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            re.compile(line)
        except re.error as e:
            raise ValidationError(f'Invalid regex pattern on line {line_num}: {e}')


# Custom widget for multi-checkbox rendering
class MultiCheckboxField(SelectMultipleField):
    """A multiple-select field that renders as a group of checkboxes."""
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()


class UploadForm(FlaskForm):
    """Form for uploading PDF and selecting redaction options."""

    # 1. File Upload Field - renamed from pdf_file to file
    file = FileField(
        'Upload File',
        validators=[InputRequired(message="Please select a file to upload.")]
    )

    # 2. Common PII Types Selection Field (India-specific with friendly names)
    common_pii_types = MultiCheckboxField(
        'Select Common PII Types to Redact',
        coerce=str
    )
    
    # 3. Advanced PII Types Selection Field 
    advanced_pii_types = MultiCheckboxField(
        'Select Advanced PII Types to Redact',
        coerce=str
    )

    # 4. Barcode Types Selection Field as checkboxes
    # Choices will be set dynamically in the Flask route
    barcode_types = MultiCheckboxField(
        'Select Barcode Types to Redact',
        coerce=str
    )
    
    # 5. Toggle for Barcode Redaction
    redact_barcodes = BooleanField(
        'Enable Barcode/QR Code Redaction', 
        default=True
    )

    # 6. Toggle for Document Metadata Redaction
    redact_metadata = BooleanField(
        'Enable Document Metadata Redaction',
        default=True
    )

    # 7. Keyword Filter Field (Optional)
    keyword_rules = TextAreaField(
        'Keyword Filters (Optional - one per line)',
        description="Only redact detected PII if the text also contains one of these keywords.",
        validators=[Optional()]
    )

    # 8. Regex Filter Field (Optional with Custom Validation)
    regex_rules = TextAreaField(
        'Regex Filters (Optional - one per line)',
        description="Only redact detected PII if the text also matches one of these Python regex patterns.",
        validators=[Optional(), validate_each_line_regex]
    )

    # 9. Submit Button
    submit = SubmitField('Start Redaction')