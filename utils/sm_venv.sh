bash
conda create -n sm_venv python=3.9.0 -y
conda activate sm_venv
export CFLAGS="-std=c++11"
pip install ipykernel 
pip install -r requirements.txt
python -m ipykernel install --user --name=sm_venv
python --version
pip install -e.

# sudo yum groupinstall 'Development Tools' -y
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --set show_channel_urls yes
# conda install paddlepaddle==2.5.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -y
# pip install paddlepaddle paddleocr