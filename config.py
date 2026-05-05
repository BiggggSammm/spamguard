import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = 'your-secret-key-change-in-production'
    #SQLALCHEMY_DATABASE_URI = 'sqlite:///instance/spam.db'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///'+ os.path.join(basedir, "instance", "spam.db").replace("\/","/")
    SQLALCHEMY_TRACK_MODIFICATIONS = False