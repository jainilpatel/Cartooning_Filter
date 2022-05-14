import argparse
from tools.utils import *
import os
from tqdm import tqdm
from glob import glob
import time
import numpy as np
from net import generator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import urllib.request
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import os,cv2
from flask import send_from_directory
from PIL import Image
from io import BytesIO
import base64
from flask_cors import CORS

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = "/home/AnimeGANv2/images/"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/<path:path>')
def send_report(path):
    return send_from_directory('/home/AnimeGANv2/results/Shinkai/t/', path)

@app.route('/file-upload', methods=['POST'])
def upload_file():
    data = request.json['image_bytes']
    im = Image.open(BytesIO(base64.b64decode(data)))
    filename = "image.jpg"
    im.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

    print("File Uplaoded")
    test("/home/AnimeGANv2/checkpoint/generator_Paprika_weight","Shinkai/t",app.config['UPLOAD_FOLDER'],True)
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    
    data = None
    
    with open('/home/AnimeGANv2/results/Shinkai/t/image.jpg',"rb") as f:
        data = str(base64.b64encode(f.read()))
    
    resp = jsonify({'message' : 'File successfully uploaded', 'url':data})
    resp.status_code = 201
    return resp

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    # params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {}'.format(flops.total_float_ops))

def test(checkpoint_dir, style_name, test_dir, if_adjust_brightness, img_size=[256,256]):
    # tf.reset_default_graph()
    result_dir = 'results/'+style_name
    check_folder(result_dir)
    test_files = glob('{}/*.*'.format(test_dir))

    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')

    with tf.variable_scope("generator", reuse=False):
        test_generated = generator.G_net(test_real).fake
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # tf.global_variables_initializer().run()
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else:
            print(" [*] Failed to find a checkpoint")
            return
        # stats_graph(tf.get_default_graph())
        # print('Processing image: ' + sample_file)
        for sample_file in tqdm(test_files):
            print(sample_file)
            sample_image = np.asarray(load_test_data(sample_file, img_size))
            image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file)))
            fake_img = sess.run(test_generated, feed_dict = {test_real : sample_image})
            if if_adjust_brightness:
                s = save_images(fake_img, image_path, sample_file)
                # print(str(s))
                return s
            else:
                s = save_images(fake_img, image_path, None)
                # print(str(s))
                return s

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)