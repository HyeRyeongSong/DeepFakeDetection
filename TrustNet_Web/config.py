import os


base_dir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = os.environ.get(
        'SECRET_KEY') or '21bfd9e51ff7d2385b88973944a6425b'
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL') or 'sqlite:///' + os.path.join(base_dir, 'youtube.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
