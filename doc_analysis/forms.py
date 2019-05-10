from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, TextField


class TargetWord(FlaskForm):

    targetword = TextField('Word: ')
    submit = SubmitField('Enter')



class ChooseId(FlaskForm):

    id = IntegerField('Document number: ')
    submit = SubmitField('Enter')
