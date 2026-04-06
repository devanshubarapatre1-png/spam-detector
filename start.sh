#!/usr/bin/env bash
pip install -r requirements.txt
python backend/train.py
python backend/app.py