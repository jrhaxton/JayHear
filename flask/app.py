from distutils.command.clean import clean
import os
import uuid
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, current_user, login_user, logout_user
from sqlalchemy import false
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from denoise import denoise

# Model import
import denoise_model.denoise

app = Flask(__name__)

############################
#       Config & Init      #
############################
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://///home/satazero/Desktop/JayHear/test.db'
app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['MAX_CONTENT_PATH'] = 128000
app.config['SECRET_KEY'] = 'dev-key'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

############################
#         User Model       #
############################
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(
        db.Integer, 
        primary_key=True
    )
    username = db.Column(
        db.String(80), 
        unique=True,
        nullable=False
    )
    email = db.Column(
        db.String(100), 
        unique=True,
        nullable=False
    )
    password = db.Column(
        db.String(1024),
        nullable=False
    )

############################
#        Sound Model       #
############################
class Sound(db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )
    file_owner = db.Column(
        db.String(80),
        nullable=False
    )
    clean_path = db.Column(
        db.String(1024),
        nullable=False 
    )
    dirty_path = db.Column(
        db.String(1024),
        nullable=False
    )

@login_manager.user_loader 
def load_user(user_id):
    return User.query.get(int(user_id))

############################
#          Routes          #
############################
@app.route('/', methods = ['GET'])
def index():
    if current_user.is_authenticated:
        return render_template('index_auth.html', name=current_user.username)
    else:
        return render_template('index.html')

@app.route('/signup-success')
def signup_success():
    return 'Success!'

@app.route('/signup')
def signup():
    print("test")
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first()

    if user:
        flash('Email address already exists')
        return redirect(url_for('index'))

    user = User.query.filter_by(username=username).first()

    if user:
        flash('Username already exists')
        return redirect(url_for('index'))

    new_user = User(email=email, username=username, password=generate_password_hash(password, method='sha256'))

    db.session.add(new_user)
    db.session.commit()

    login_user(new_user)

    os.mkdir('/home/satazero/Desktop/JayHear/flask/static/uploads/' + str(current_user.username))

    return redirect(url_for('index'))

@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False 

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Incorrect login credentials')
        return redirect(url_for('login'))

    login_user(user)

    return redirect(url_for('submit'))

@app.route('/logout')
@login_required
def logout():
    if current_user.is_authenticated:
        logout_user()
        return redirect(url_for('login'))
    return redirect(url_for('login'))

@app.route('/sounds')
def sounds():
    if current_user.is_authenticated:
        file_owner = current_user.username
        sounds = Sound.query.filter_by(file_owner=current_user.username).all()

        return render_template('sounds.html', size=len(sounds), sounds=sounds, name=current_user.username)
    return redirect(url_for('login'))

@app.route('/submit-sound')
def submit():
    if current_user.is_authenticated:
        file_owner = current_user.username
        sounds = Sound.query.filter_by(file_owner=current_user.username).all()
        return render_template('submit.html', size=len(sounds), sounds=sounds, name=current_user.username)
    return redirect(url_for('login'))

@app.route('/submit-sound', methods=['POST'])
def submit_post():
    f = request.files['file']
    file_name = str(uuid.uuid4().hex)
    dirty_file_path = "/home/satazero/Desktop/JayHear/flask/static/uploads/" + str(current_user.username) + "/dirty_" + file_name + ".wav"
    clean_file_path = "/home/satazero/Desktop/JayHear/flask/static/uploads/" + str(current_user.username) + "/clean_" + file_name + ".wav"
    f.save(dirty_file_path)

    denoise(dirty_file_path, clean_file_path)

    print(dirty_file_path)
    print(clean_file_path)

    new_sound = Sound(
        file_owner=current_user.username, 
        dirty_path=str(current_user.username + "/dirty_" + file_name + ".wav"),
        clean_path=str(current_user.username + "/clean_" + file_name + ".wav")
    )

    db.session.add(new_sound)
    db.session.commit()

    return redirect(url_for('sounds'))

if __name__ == "__main__":
    app.run(debug=True)