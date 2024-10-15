date="241203"
subtask="syn_scenes"
model_root="/path/to/model"

python render.py -m $model_root/$date/$subtask/Hotdog/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Lego/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/FurBall/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/AnisoMetal/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Drums/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Translucent/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Tower/ \
                --iteration 100000 \
                --skip_train \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/MaterialBalls/ \
                --iteration 100000 \
                --skip_train \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Egg/ \
                --iteration 100000 \
                --skip_train \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Fabric/ \
                --iteration 100000 \
                --skip_train \
                --use_nerual_phasefunc 

python render.py -m $model_root/$date/$subtask/Cup/ \
                --iteration 100000 \
                --skip_train \
                --use_nerual_phasefunc 
                