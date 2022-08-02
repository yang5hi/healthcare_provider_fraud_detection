from flask import Flask
app=Flask(__name__)
@app.route('/')
def hello_world():
        return "Hi"
        
app.run()


# If you have the app.run() in the app.py it will can be executed using python app.py






#****************Mac
# export FLASK_APP=app.py



#********Windows
# set FLASK_APP=app.py
# flask run