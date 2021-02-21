from app.models import User
from flask_login import current_user
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, SubmitField, IntegerField, TextField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, Length


class RegistrationForm(FlaskForm):
    user_name = StringField('Username', validators=[
                            DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[
                             DataRequired(), Length(min=8)])
    confirm_password = PasswordField('Confirm Password', validators=[
                                     DataRequired(), Length(min=8), EqualTo('password')])
    age = IntegerField('Age', validators=[DataRequired()])
    address = TextField('Address', validators=[
                        DataRequired(), Length(min=4, max=40)])
    submit = SubmitField('Sign Up')

    def validate_username(self, user_name):
        '''Validation if the username entered is already present or not'''

        user = User.query.filter_by(user_name=user_name.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        '''Validation if the email entered is already present or not'''

        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[
                             DataRequired(), Length(min=8)])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class UpdateAccountForm(FlaskForm):
    user_name = StringField('Username', validators=[
                            DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    avatar = FileField('Update Profile Picture', validators=[
                       FileAllowed(['jpg', 'png', 'jpeg'])])
    submit = SubmitField('Update')

    def validate_username(self, user_name):
        '''Validation if the username entered is already present or not'''

        if user_name.data != current_user.user_name:
            user = User.query.filter_by(user_name=user_name.data).first()
            if user is not None:
                raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        '''Validation if the email entered is already present or not'''
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user is not None:
                raise ValidationError('Please use a different email address.')


class VideoUploadForm(FlaskForm):
    video_title = TextField('Title', validators=[
                            DataRequired(), Length(max=60)])
    description = TextField('Description', validators=[DataRequired()])
    category = StringField('Category', validators=[
                           DataRequired(), Length(max=20)])
    video_content = FileField('Video', validators=[FileAllowed(
        ['mp4', 'mkv', '3gp', 'mov']), FileRequired()])
    submit = SubmitField('Upload')


class UpdateVideoForm(FlaskForm):
    video_title = TextField('Title', validators=[
                            DataRequired(), Length(max=60)])
    description = TextField('Description', validators=[DataRequired()])
    category = StringField('Category', validators=[
                           DataRequired(), Length(max=20)])
    submit = SubmitField('Update')


class CommentForm(FlaskForm):
    body = StringField('', validators=[DataRequired(), Length(max=400)])
    submit = SubmitField('Comment Here')
