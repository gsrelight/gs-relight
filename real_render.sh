date="241203"
subtask="real_scenes"
model_root="/path/to/model"

# ==========================================================
# ======================Lightstage==========================
                
python render.py -m $model_root/$date/$subtask/Container/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --valid \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/LilOnes/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --valid \
                --use_nerual_phasefunc 

python render.py -m $model_root/$date/$subtask/Boot/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --valid \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Fox/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --valid \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Nefertiti/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --valid \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Zhaojun/ \
                --iteration 100000 \
                --skip_train \
                --gamma \
                --valid \
                --use_nerual_phasefunc 

## ==========================================================
## =======================NRHints============================
                
python render.py -m $model_root/$date/$subtask/Pikachu/ \
                --iteration 100000 \
                --skip_train \
                --valid \
                --use_nerual_phasefunc 

python render.py -m $model_root/$date/$subtask/Cluttered/ \
                --iteration 100000 \
                --skip_train \
                --valid \
                --use_nerual_phasefunc 

python render.py -m $model_root/$date/$subtask/Cup-Fabric/ \
                --iteration 100000 \
                --skip_train \
                --valid \
                --use_nerual_phasefunc 

python render.py -m $model_root/$date/$subtask/Fish/ \
                --iteration 70000 \
                --skip_train \
                --valid \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Cat/ \
                --iteration 100000 \
                --skip_train \
                --valid \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Pixiu/ \
                --iteration 100000 \
                --skip_train \
                --valid \
                --use_nerual_phasefunc 
                
python render.py -m $model_root/$date/$subtask/Cat_on_Decor/ \
                --iteration 100000 \
                --skip_train \
                --valid \
                --use_nerual_phasefunc 
