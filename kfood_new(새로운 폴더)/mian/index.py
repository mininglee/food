# -*- coding: utf-8 -*-

# 필요한 모듈 import 
from flask import Blueprint, request, render_template, flash, redirect, url_for
from flask import current_app as app
 
main = Blueprint('main', __name__, url_prefix='/') # url_prefix : '/'로 url을 붙여라
 
@app.route('/')
def index():
      return render_template('/main/index.html')
