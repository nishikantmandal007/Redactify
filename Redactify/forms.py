from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, SelectMultipleField, TextAreaField # Use TextAreaField
from wtforms.validators import InputRequired, Regexp # Use Regexp

class UploadForm(FlaskForm):
    pdf_file = FileField('Upload PDF', validators=[InputRequired()])
    # Choices should match the strings returned by utils.get_pii_types()
    pii_types = SelectMultipleField('PII Types to Redact (Ctrl+Click for multiple)', choices=[
        ('PERSON', 'Name (Person)'),
        ('PHONE_NUMBER', 'Phone Number'),
        ('EMAIL_ADDRESS', 'Email Address'),
        ('LOCATION', 'Location/Address'),
        ('CREDIT_CARD', 'Credit Card Number'),
        ('US_SSN', 'US Social Security Number'),
        ('DATE_TIME', 'Date/Time'),
        ('NRP', 'Nationality/Religion/Politics'),
        ('URL', 'URL/Website'),
        # Add more choices corresponding to utils.get_pii_types()
    ])
    keyword_rules = TextAreaField('Keyword Filter Rules (optional, one keyword per line)')
    regex_rules = TextAreaField('Regex Filter Rules (optional, one regex per line)',
                                validators=[Regexp(r'^(?:.*|\s*)$', message="One or more regex patterns might be invalid.")]) # Allow empty or valid regex lines
    submit = SubmitField('Redact PDF')