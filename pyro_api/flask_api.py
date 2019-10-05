#https://code.tutsplus.com/vi/tutorials/creating-a-web-app-from-scratch-using-python-flask-and-mysql--cms-22972
from flask import Flask,request
from misgan import *
app = Flask(__name__)

column  = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14}
@app.route("/")
def main():
    # print(request)
    vt = request.args.get('vt')
    if (len(vt)<=0):
        return str(1000)
    print("11 : vt : " ,vt)
    vt = [column[i] for i in vt.split(".")]
    vp = column[request.args.get('vp')]

    print(vt)
    print(vp)

    print("aaaa")
    return str(calc_loss_FD(vt,vp))
@app.route("/hello")
def hello():
    return "Hello World!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)