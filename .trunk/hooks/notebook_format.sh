#!/bin/sh

if command -v pdm &> /dev/null; then
	pdm run jupyter nbconvert --clear-output --inplace scripts/notebooks/*.ipynb
fi

git add scripts/notebooks/*.ipynb
