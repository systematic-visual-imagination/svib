i=0
while [ "$i" -lt 100 ]; do
    ./blender-2.91.2-linux64/blender --background --python-use-system-env --python create_data.py -- --rule $1 --save_path $2 --bind_path $3
    i=$(( i + 1 ))
done
