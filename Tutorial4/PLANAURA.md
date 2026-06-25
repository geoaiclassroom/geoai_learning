## Preparing your environment

### The details to create an independant environment and install planaura as a package is explained here:

    https://github.com/NRCan/planaura/tree/main#to-prepare-the-environment-for-installing-planaura-package

### If you want to run Planaura using the "tutorial" environment suggested in this repo
We will assume we already built a "tutorial" environment using the requirements files available on the root of this repository.

- [If you have GPU available on your device](https://github.com/geoaiclassroom/geoai_learning/blob/main/requirements_gpu.txt)
- [If you only have CPU available on your device](https://github.com/geoaiclassroom/geoai_learning/blob/main/requirements_cpu.txt)

Then, clone the main branch of the Planaura package to your machine:

    git clone https://github.com/NRCan/planaura.git

Then, activate your "tutorial" environment (*conda activate tutorial*) and enter (*cd*) the clonned directory (*planaura*). Replace the existing "pyproject.toml" file with [the one available in this tutorial](/Planaura_edits/pyproject.toml). 

I have made minimal changes to this pyproject file by removing some hard dependencies given that we already have required packages installed in our tutorial environment.

Then, install the package:

    pip install -e .

To check if the installation is successful:

    conda list planaura

This should show that the recent version of planaura is available (check that it matches the version [here](https://github.com/NRCan/planaura/blob/main/AUTOVERSION). 

Note that you might get a few "warnings" when installing planaura into this tutorial's environment, it is normal due to some ignorable conflicts between the original planaura's env and this one. They are just warnings and don't affect the performance.

## Usage

For a tutorial on how to use Planaura for change detection please refer to its Github page
    https://github.com/NRCan/planaura/tree/main#to-infer-from-planaura
    https://github.com/NRCan/planaura/tree/main#sample-usage

In case, you want to test inference with a smaller dataset, I provided a smaller sample dataset in the [Data folder](/Data).

All need to be done is setting the paths correctly (in the config file and in the csv file), and then run inference:

    python infer_geotiff.py "path/to/Tutorial4/Data/sample_config.yaml"

    