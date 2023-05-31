#! /bin/bash

mkdir data
mkdir models

cd data
mkdir S-Albilora
cd S-Albilora

wget http://ceda.ic.ufmt.br/files/cobra/Oliveira2019-train-other.zip -O Oliveira2019-train-other.zip
unzip Oliveira2019-train-other.zip
rm -rf Oliveira2019-train-other.zip
mv other others

cd ../..
wget https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1 -O models/Cnn14_mAP%3D0.431.pth
