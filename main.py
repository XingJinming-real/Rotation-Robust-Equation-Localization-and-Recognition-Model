from flask import Flask, render_template, Response, request, jsonify
import cv2 as cv
# import numpy, torch
import torch
import time
from process import *
from PIL import Image
# from crnn.Chinese_alphabet import alphabet
# import crnn.lib.models.crnn as crnn
from paddleocr import PaddleOCR

app = Flask(__name__)
# 电脑自带摄像头
camera0 = cv.VideoCapture(0)
crnnModel = PaddleOCR(use_angle_cls=True, lang='ch',
                      rec_char_dict_path='./crnnP/equations_dict.txt',
                      rec_model_dir='./crnnP/equations_rec',
                      det=False)
yolo = torch.hub.load('./yolov5/', 'custom', path='./yolov5/best.pt', source='local')


# def getCrnnModel():
#     model_path = 'crnn/expr/netCRNN.pth'  # 模型权重路径
#     # alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
#     model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)  # 创建模型
#     if torch.cuda.is_available():
#         model = model.cuda()
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     return model


@app.route("/")
def index():
    return render_template("index.html")


# 获得本地摄像头图像字节流传输
def gen_frames0():
    while 1:
        ret, frame = camera0.read()
        if not ret:
            break
        # 把获取到的图像格式转换(编码)成流数据，赋值到内存缓存中;
        # 主要用于图像数据格式的压缩，方便网络传输
        ret1, buffer = cv.imencode('.jpg', frame)
        # 将缓存里的流数据转成字节流
        frame = buffer.tobytes()
        # 指定字节流类型image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_frames1(modifyImg=False):
    while 1:
        ret, frame = camera0.read()
        if not ret:
            break
        # 把获取到的图像格式转换(编码)成流数据，赋值到内存缓存中;
        # 主要用于图像数据格式的压缩，方便网络传输
        frame = modify(frame, yolo, crnnModel, None, modifyImg=modifyImg, thickness=4, cvtColor=False, byte=False,
                       video=True)
        ret1, buffer = cv.imencode('.jpg', frame)
        # 将缓存里的流数据转成字节流
        frame = buffer.tobytes()
        # 指定字节流类型image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed0')
def video_feed0():
    if request.args.get('close'):
        return './static/upload.png'
    else:
        return Response(gen_frames0(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed1')
def video_feed1():
    if request.args.get('close'):
        return './static/upload.png'
    else:
        return Response(gen_frames1(True if request.args.get('fixed') == 'true' else False),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pie1')
def pie1():
    # pie1 Video
    rightCount, wrongCount = videoCount
    return jsonify({"right_count": rightCount, "wrong_count": wrongCount})


@app.route('/pie2')
def pie2():
    # pie2 Img
    rightCount, wrongCount = imgCount
    return jsonify({"right_count": rightCount, "wrong_count": wrongCount})


@app.route("/upload", methods=['POST', 'GET'])
def upload():
    if request.method == 'GET':
        uploadIdx[0] += 1
        if uploadIdx[0] > 100:
            uploadIdx[0] = 1
        new_path = './static/upload/' + str(int(time.time())) + '.png'
        old_path = request.args.get('file')
        imgModified = modify(old_path, yolo, crnnModel, new_path, modifyImg=True if uploadIdx[0] % 2 else False,
                             thickness=4, byte=False, cvtColor=True, video=False)
        plt.imsave(new_path, imgModified)
        return new_path
    elif request.method == 'POST':
        stick = str(int(time.time()))
        pathOri = './static/upload/' + stick + '.png'
        pathNew = './static/upload/' + stick + 'New.png'
        file = request.files.get('file')
        data = file.read()
        modify(data, yolo, crnnModel, pathNew, modifyImg=False, thickness=4, cvtColor=True, video=False)
        Image.open(BytesIO(data)).save(pathOri)
        return jsonify({'path1': pathOri, 'path2': pathNew})


# crnnModel = getCrnnModel()

uploadIdx = [0]
if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1')
