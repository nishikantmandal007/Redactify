# /home/stark007/Projects/Redactify/Redactify/forms.py - MODIFIED

from flask_wtf import FlaskForm
# Import necessary field types
from wtforms import FileField, SubmitField, SelectMultipleField, TextAreaField
# Import necessary validators
from wtforms.validators import InputRequired, Optional # Import Optional for regex field
import re # Import re for custom validator

# Custom validator to check if each line is a valid regex
def validate_each_line_regex(form, field):
    """Checks if each non-empty line in the TextAreaField is a valid regex."""
    if field.data: # Only validate if there's data
        lines = field.data.splitlines()
        invalid_lines = []
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line: # Ignore empty lines
                try:
                    re.compile(stripped_line) # Try to compile the regex
                except re.error as e:
                    invalid_lines.append(f"L{i+1}: '{stripped_line[:30]}{'...' if len(stripped_line)>30 else ''}' ({e})") # Add line number and error
        if invalid_lines:
            # Raise ValidationError if any line is invalid
            # Join the messages for display
            error_message = "Invalid regex pattern(s) found: " + "; ".join(invalid_lines)
            from wtforms.validators import ValidationError # Import here to avoid potential circular issues if needed elsewhere
            raise ValidationError(error_message)

class UploadForm(FlaskForm):
    """Form for uploading PDF and selecting redaction options."""

    # 1. File Upload Field
    pdf_file = FileField(
        'Upload PDF File',
        validators=[InputRequired(message="Please select a PDF file to upload.")]
    )

    # 2. PII Types Selection Field
    # CHOICES ARE REMOVED HERE - They will be set dynamically in the Flask route (app.py)
    # using data from utils.get_pii_types().
    # `coerce=str` ensures submitted values are treated as strings.
    pii_types = SelectMultipleField(
        'Select PII Types to Redact (Hold Ctrl/Cmd to select multiple)',
        coerce=str # Important when choices are set dynamically
        # No choices defined here
    )

    # 3. Keyword Filter Field (Optional)
    keyword_rules = TextAreaField(
        'Keyword Filters (Optional - one per line)',
        description="Only redact detected PII if the text also contains one of these keywords.",
        validators=[Optional()] # Makes this field not required
    )

    # 4. Regex Filter Field (Optional with Custom Validation)
    regex_rules = TextAreaField(
        'Regex Filters (Optional - one per line)',
        description="Only redact detected PII if the text also matches one of these Python regex patterns.",
        # Use Optional() and our custom validator
        validators=[Optional(), validate_each_line_regex]
    )

    # 5. Submit Button
    submit = SubmitField('Start Redaction')