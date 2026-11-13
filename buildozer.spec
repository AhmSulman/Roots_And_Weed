[app]
title = OKF - Octave Key Finder
package.name = okf
package.domain = org.okf

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,ttf
source.exclude_dirs = tests,bin,.venv,okf_output,.github,assets,.git

version = 0.1.0

# Android build deliberately excludes torch/demucs -- there is no Android
# build of PyTorch and no python-for-android recipe for it. Stem separation
# on-device uses the pure-NumPy DSP fallback in okf/separation.py instead.
# See okf/ALGORITHM_NOTES.md.
requirements = python3,kivy==2.3.0,kivymd==1.2.0,numpy,pillow

orientation = portrait
fullscreen = 0

# icon.filename = %(source.dir)s/assets/icon.png  # drop a 512x512 PNG in assets/ and uncomment

android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_AUDIO
android.api = 34
android.minapi = 24
android.ndk_api = 24
android.archs = arm64-v8a,armeabi-v7a
android.allow_backup = True

# CI runs non-interactively: auto-accept the Android SDK license, otherwise
# sdkmanager skips build-tools and the build dies with "Aidl not found".
android.accept_sdk_license = True

p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1
