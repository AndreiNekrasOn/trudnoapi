from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired


class VacancyData(FlaskForm):
    json_data = TextAreaField('json for predicion', validators=[DataRequired()])
    namelemm = StringField('name.lemm', validators=[DataRequired()])
    city = StringField('city', validators=[DataRequired()])
    companyid = StringField('companyid', validators=[DataRequired()])
    publication_date = StringField('publication_date', validators=[DataRequired()])
    employment = StringField('employment', validators=[DataRequired()])
    schedule = StringField('schedule', validators=[DataRequired()])
    experience = StringField('experience', validators=[DataRequired()])
    key_skills = StringField('key_skills', validators=[DataRequired()])
    specializations = StringField('specializations', validators=[DataRequired()])
    descriptionlemm = StringField('descriptionlemm', validators=[DataRequired()])

    submit = SubmitField('Predict')
