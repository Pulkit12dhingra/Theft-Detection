from flask import Flask, render_template,request,Response
from flask_cors import CORS,cross_origin
import cv2
import numpy as np
import sqlite3
from PIL import Image
import os
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app)
cap=cv2.VideoCapture(0)

default_dict={'sample_num':0,'signup_tag':0,'warning':[]}
recognizer=cv2.face.LBPHFaceRecognizer_create()
path=os.path.abspath('dataset')
model = tf.keras.models.load_model('my_model.h5')

def InsertOrUpdate(Id, Name, Gender, Occupation, Age):
    con=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=con.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name=%r ,Gender=%r ,Occupation=%r ,Age=%r WHERE ID=%d"%(Name,Gender,Occupation,Age,int(Id))
        con.execute(cmd)
        con.commit()
        
    else:
        cmd="INSERT INTO People Values(%r,%r,%r,%r,%r)"%(Id, Name, Gender, Occupation, Age)
        con.execute(cmd)
        con.commit()
        con.close()

def train(path):
    imgPaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    Ids=[]
    for imgPath in imgPaths:
        faceImg=Image.open(imgPath).convert("L")
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imgPath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(ID)
        
    return np.array(Ids), faces

def getProfile(id):
    con=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=con.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    con.close()
    return profile

def arms_detect(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2
    image = cv2.resize(frame,(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    #predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    #print(label[1])
    tag=label[1]
    if tag in default_dict['warning']:
        frame = cv2.putText(frame, 'Arms Detected', org, font,  
                       fontScale, color, thickness, cv2.LINE_AA)
    else:
        frame = cv2.putText(frame, 'NO Arms Detected', org, font,  
                       fontScale, color, thickness, cv2.LINE_AA)
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('login.html')

@app.route('/security',methods=['POST'])
def security():
    int_features = [str(x) for x in request.form.values()]
    final_features = list(int_features)
    default_dict['id1']=final_features[0]
    print(final_features)
    InsertOrUpdate(*final_features)
    default_dict['signup_tag']=1
    return render_template('index.html',info='Please wait for a minute to store the data make sure to have a clear backgroung')

def gen():
    while True:
        ret,frame = cap.read()
        faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray, 1.3,5)
        if default_dict["signup_tag"]==1 and default_dict['sample_num']<100:
            
            
            for(x,y,w,h) in faces:
                cv2.imwrite("Dataset/User."+str(default_dict['id1'])+"."+str(default_dict['sample_num'])+".jpg", gray[y:y+h,x:x+w])
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                default_dict['sample_num']+=1
        
        elif default_dict['sample_num']==100:
            default_dict['sample_num']=0
            default_dict['signup_tag']=0
            Ids, faces=train(path)
            recognizer.train(faces,Ids)
            recognizer.save("recognizer/trainingData.yml")
            render_template('index.html')
        else:
            #detect the unknow faces
            rec=cv2.face.LBPHFaceRecognizer_create()
            rec.read("recognizer/trainingData.yml")
            id=0
            #font = cv2.cv2.InitFont(cv2.cv2.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
            fontface = cv2.FONT_HERSHEY_SIMPLEX
            fontcolor = (0, 255, 0)
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                id, conf=rec.predict(gray[y:y+h, x:x+w])
                profile=getProfile(id)
                if conf>60:
                    cv2.putText(frame, 'Unknown', (x,y+h+30), fontface, 0.6, fontcolor, 2)
                    profile=None

                if (profile!=None):
                    #cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
                    cv2.putText(frame, "Name:"+str(profile[1]), (x,y+h+30), fontface, 0.6, fontcolor, 2)

        arms_detect(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        image= jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)