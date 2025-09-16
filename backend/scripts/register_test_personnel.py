#!/usr/bin/env python3
"""Utility: create a test personnel entry and register a fake FCM token via backend endpoint.
Usage: python register_test_personnel.py --name TestUser --token FAKE_TOKEN_123 --user-id 1
"""
import argparse
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_URL = os.getenv('DATABASE_URL', 'sqlite:///' + os.path.join(ROOT, 'test.db'))

from backend.models import Personnel  # using package import when run from repo root

def create_person(name):
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    s = Session()
    person = Personnel(name=name)
    s.add(person)
    s.commit()
    s.refresh(person)
    print('Created personnel id=', person.id)
    return person.id

def register_token(token, user_id, backend_url='http://127.0.0.1:8000'):
    url = f"{backend_url}/register_fcm_token"
    resp = requests.post(url, data={'token': token, 'user_id': user_id}, timeout=5)
    print('Register token response:', resp.status_code, resp.text)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='TestUser')
    p.add_argument('--token', default='FAKE_TOKEN_123')
    p.add_argument('--user-id', type=int, default=None)
    p.add_argument('--backend', default='http://127.0.0.1:8000')
    args = p.parse_args()

    user_id = args.user_id
    if user_id is None:
        user_id = create_person(args.name)
    register_token(args.token, user_id, args.backend)
