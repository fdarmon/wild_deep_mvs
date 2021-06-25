set -e

for SCENE in 1 4 9 #10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118
do
  python reconstruction_pipeline.py --dataset dtu --scene scan$SCENE --fusion fusibile \
  --fusion_depth_thresh 0.25 --override $*
done
