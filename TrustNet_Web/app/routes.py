import os
import secrets
import time
import asyncio
from PIL import Image
from app import app, db
from app.forms import RegistrationForm, LoginForm, UpdateAccountForm, VideoUploadForm, UpdateVideoForm, CommentForm
from app.models import User, Video, Comments, Likes
from flask import render_template, url_for, flash, redirect, request, abort
from flask_login import login_user, current_user, logout_user, login_required
from werkzeug.urls import url_parse

from api.TrustNetAPI import Triton
import pycuda.autoinit

triton = Triton(DEBUG = True)

# flash Options
# https://getbootstrap.com/docs/4.0/components/alerts/

@app.route('/')
@app.route('/home')
def home():
    """Home Page view"""
    videos = Video.query.all()
    return render_template('home.html', title='Home', videos=videos)


@app.route('/register', methods=['POST', 'GET'])
def register():
    """View for registration of new user"""

    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(user_name=form.user_name.data, email=form.email.data,
                    age=form.age.data, address=form.address.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['POST', 'GET'])
def login():
    """View for login of existing user to the app"""

    if current_user.is_authenticated:
        return redirect(url_for('home'))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            if not next_page or url_parse(next_page).netloc != '':
                next_page = url_for('home')
            return redirect(next_page)
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
    #return redirect(url_for('home'))


@app.route("/video/<int:id>", methods=['GET', 'POST'])
def video(id):
    video = Video.query.get_or_404(id)

    if request.method == 'GET':
        video.views_count += 1
        db.session.commit()
    form = CommentForm()
    if form.validate_on_submit():
        comment = Comments(body=form.body.data, video=video,
                           author=current_user._get_current_object())
        db.session.add(comment)
        db.session.commit()
        return redirect(url_for('video', id=video.id))

    comments = video.comments.order_by(Comments.comment_time.desc())
    return render_template('video.html', title=video.video_title, form=form, video=video, comments=comments)


def sitsave_avatar(form_picture):
    '''Function to save avatar into static/avatar directory with size optimization'''

    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/avatar', picture_fn)

    output_size = (300, 300)    # avatar resizing
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    """View for Profile Page"""

    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.avatar.data:
            picture_file = save_avatar(form.avatar.data)
            current_user.avatar = picture_file
        current_user.user_name = form.user_name.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.user_name.data = current_user.user_name
        form.email.data = current_user.email
    avatar = url_for('static', filename='avatar/' + current_user.avatar)
    return render_template('account.html', title='Account', form=form, avatar=avatar, videos=current_user.videos)


def save_video(form_video):
    '''Function to save video into static/videos directory'''

    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_video.filename)
    video_fn = random_hex + f_ext
    video_path = os.path.join(app.root_path, 'static/videos', video_fn)

    form_video.save(video_path)

    return video_fn, random_hex


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Video Upload Page view"""

    form = VideoUploadForm()

    # TODO : Check DeepFake

    if form.validate_on_submit():
        video_file, random_hex = save_video(form.video_content.data)
        
        video_path = "/".join(['./app/static/videos', video_file])

        faces = triton.getFaces(video_path)

        triton.run(faces)
        Result_DeepFake, results, CAMList = triton.getResults()
        
        print("Ensembled DeepFake Probability {:.2f} %".format(results * 100))
        
        if Result_DeepFake:
            triton.makeCAM(video_file, CAMList)
            print("DeepFake Video Upload Attempt Found - ID : ", current_user.user_name)
            flash('DeepFake Detected {:.2f} % !!!'.format(results * 100), 'danger')
            flash('Not uploaded', 'danger')
            os.remove(video_path)
            #move_to_deepfake_path = "/".join(['./app/static/videos/deepfakes', video_file])
            #os.system(f"mv {video_path} {move_to_deepfake_path}")
            imgs = []
            for i in CAMList:
                imgs.append(f"cam_{i}.png")
            return render_template('CAM.html', title='DeepFake Found', images=imgs, id=f'{random_hex}.mp4')
            #return redirect(url_for('home'))
            
        video = Video(video_title=form.video_title.data, video_content=video_file,
                      description=form.description.data, category=form.category.data, author=current_user)
        
        db.session.add(video)
        db.session.commit()
        flash('Your Video has been posted!', 'success')
        return redirect(url_for('home'))

    return render_template('upload.html', title='Upload Video', form=form, legend='Upload Your Video')


@app.route('/video/<int:id>/update', methods=['GET', 'POST'])
@login_required
def update_video(id):
    """Function to update video contents"""

    video = Video.query.get_or_404(id)
    if video.author != current_user:
        abort(403)

    form = UpdateVideoForm()
    if form.validate_on_submit():
        video.video_title = form.video_title.data
        video.description = form.description.data
        video.category = form.category.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('video', id=video.id))
    elif request.method == 'GET':
        form.video_title.data = video.video_title
        form.description.data = video.description
        form.category.data = video.category
    return render_template('update_video.html', title='Update Video', form=form, legend='Update video')


@app.route('/like/<int:video_id>/<action>')
@login_required
def like_action(video_id, action):
    video = Video.query.filter_by(id=video_id).first_or_404()
    if action == 'like':
        current_user.like_video(video)
        db.session.commit()
    if action == 'unlike':
        current_user.unlike_video(video)
        db.session.commit()
    return redirect(request.referrer)
