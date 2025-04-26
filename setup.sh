# Create the Conda environment
conda env create -f environment/environment.yml
#source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the Conda environment
#conda activate ESF

# shellcheck disable=SC2164
cd cumm

conda run -n ESF python setup.py develop

conda run -n ESF python -c "import cumm"

# shellcheck disable=SC2164
cd ../spconv

conda run -n ESF python setup.py develop

conda run -n ESF python -c "import spconv"

# shellcheck disable=SC2103
cd ..

# Run OpenPCDet's setup
conda run -n ESF python setup.py develop