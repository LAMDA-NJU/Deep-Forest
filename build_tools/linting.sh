#!/bin/bash

set -e -x
set -o pipefail

if ! flake8 --verbose --filename=*.py deepforest/; then
  echo 'Failure on Code Quality Check.'
  exit 1
fi