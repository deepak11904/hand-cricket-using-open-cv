#!/usr/bin/env python3
"""Test script to verify all imports and model loading"""

print("Testing imports...")

try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print("✗ OpenCV failed:", e)

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print("✗ NumPy failed:", e)

try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
except ImportError as e:
    print("✗ MediaPipe failed:", e)

try:
    import sklearn
    print("✓ Scikit-learn imported successfully")
except ImportError as e:
    print("✗ Scikit-learn failed:", e)

try:
    import pickle
    print("✓ Pickle imported successfully")
except ImportError as e:
    print("✗ Pickle failed:", e)

print("\nTesting model loading...")
try:
    with open('hand_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    print("✓ Model loaded successfully")
    print("Model type:", type(rf))
except Exception as e:
    print("✗ Model loading failed:", e)

print("\nTesting MediaPipe setup...")
try:
    hands = mp.solutions.hands
    mp_styles = mp.solutions.drawing_styles
    print("✓ MediaPipe hands and drawing styles loaded")
except Exception as e:
    print("✗ MediaPipe setup failed:", e)

print("\nAll tests completed!")
