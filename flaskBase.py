# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 12:49:15 2018

@author: velmurugan.m
"""

from flask import Flask, redirect, url_for, request, render_template
from TOPS_Prediction import processData


app = Flask(__name__)
 
@app.route("/NewPage/")
def newPage():
    pass

@app.route("/TOPSPrediction/", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("loginpage.html", error=False)

    response = request.form["Response"]
    
    print ("Response from Use Field is: %s") % response
    
    predOutput = processData(response)    
        
    if response != "":
        #predOutput = "Yes this Category has been predicted"
        return render_template("outputpage.html", value=predOutput)
    else:
        predOutput = "Please enter a proper text content for Prediction !!"
        return render_template("outputpage.html", value=predOutput)   
    

if __name__=='__main__':
    app.run(debug=False, port=9001, host='0.0.0.0')
