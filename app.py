import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, flash, request, redirect

from classify import classify

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

UPLOAD_FOLDER = r'static/uploads'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/")
def upload() :
    return render_template("index.html", pagetitle="Homepage")

@app.route("/", methods=['GET', 'POST'])
def uploading() :
    if request.method == 'POST' :
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            output = classify(image_path)

            return render_template("uploading.html",
                                    pagetitle="result",
                                    image_name=filename,
                                    result=output)
        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)

if __name__ == "__main__" :
   app.run(host="0.0.0.0", port=5000, debug=False)