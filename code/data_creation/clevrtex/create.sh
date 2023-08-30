i=0
while [ "$i" -le 100 ]; do
    /common/home/gs790/blender-2-91/blender --background --python create_data.py -- --rule $1 --save_path $2 --bind_path $3
    i=$(( i + 1 ))
done
