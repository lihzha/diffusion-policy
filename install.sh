conda create -n gdc python=3.9 -y
conda activate gdc
pip install -e .

cd guided_dc/maniskill
pip install -e .
cd ../..