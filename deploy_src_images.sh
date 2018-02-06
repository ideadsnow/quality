#!/usr/bin/bash

src_path="/Users/yypm/Documents/fr/source/quality_game_1080p"
script_path="/Users/yypm/Documents/competitor_test/video/image_process.py"

output_path="/Users/yypm/Documents/fr/source/"

# Web Default
# python $script_path --no-pre-roi --no-resize --x 12 --y 171 --width 1896 --height 807 $src_path
# mv ${src_path}_roi ${output_path}web_default

# Android Default && iOS Default. NOTE: android && iOS params are same now.
python $script_path --no-pre-roi --no-resize --x 359 --y 161 --width 1547 --height 846 $src_path
mv ${src_path}_roi ${output_path}android_default

# Assist Android
python $script_path --no-pre-roi --no-resize --x 0 --y 163 --width 1918 --height 820 $src_path
mv ${src_path}_roi ${output_path}assist_android

# Assist iOS
python $script_path --no-pre-roi --no-resize --x 1 --y 160 --width 1919 --height 752 $src_path
mv ${src_path}_roi ${output_path}assist_ios
