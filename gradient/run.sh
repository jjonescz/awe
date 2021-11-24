#!/bin/bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root \
    --LabApp.trust_xheaders=True --LabApp.disable_check_xsrf=False \
    --LabApp.allow_remote_access=True --LabApp.allow_origin='*'
