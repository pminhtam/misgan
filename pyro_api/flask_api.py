#https://code.tutsplus.com/vi/tutorials/creating-a-web-app-from-scratch-using-python-flask-and-mysql--cms-22972
from flask import Flask,request
# from misgan import *
from gain.data_gain import Data_Generate
from gain.model_gain import *

app = Flask(__name__)

column  = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14}
data_file = "data3_test.csv"
Data = np.loadtxt("./data/" + data_file, delimiter=",")



Dim,Train_No,trainX,trainM = Data_Generate(Data)
X,M,H,New_X,D_loss1,G_loss1,MSE_train_loss,MSE_test_loss,D_solver,G_solver,G_sample,prob_masks = make_model(Dim)
# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "./model/gain_data3.ckpt")
vt_list = []
vp_list = []
@app.route("/")
def main():
    # print(request)


    vt = request.args.get('vt')
    if (len(vt)<=0):
        vt = []
    else:
        vt = [column[i] for i in vt.split(".")]
    vp = column[request.args.get('vp')]
    print("11 : vt : " ,vt)
    print("11 : vp : " ,vp)

    Missing0 = np.zeros((Train_No, Dim))

    for i in vt:
        Missing0[:,i] = 1

    Z_mb = sample_Z(Train_No, Dim)
    M_mb = Missing0
    X_mb = trainX
    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce
    global G_sample
    G_sample_val = sess.run(G_sample, feed_dict={X: trainX, M: M_mb, New_X: New_X_mb})
    error = str(np.mean(np.square(np.subtract(G_sample_val*(1-M_mb),trainX*(1-M_mb))),axis=0)[vp])
    with open("log.txt",'a+') as log:
        log.writelines(str(vt)+'        '+str(vp)+'         '+error+'\n')
    return error
@app.route("/hello")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)