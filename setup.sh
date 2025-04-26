# Create the Conda environment
conda env create -f environment/environment.yml

# shellcheck disable=SC2164
cd third_party/cumm

conda run -n ESF python setup.py develop

conda run -n ESF python -c "import cumm"

# shellcheck disable=SC2164
cd ../spconv

conda run -n ESF python setup.py develop

conda run -n ESF python -c "import spconv"

# shellcheck disable=SC2103
cd ../..

# Run OpenPCDet's setup
conda run -n ESF python setup.py develop