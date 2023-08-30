i=0
while [ "$i" -le 100 ]; do
    /common/home/gs790/blender-2-91/blender --background --python create_data.py -- --no_target --rule none --save_path $1 --bind_path $2
    i=$(( i + 1 ))
done
