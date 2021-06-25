set -e

for scene in  colosseum_exterior #grand_place_brussels hagia_sophia_interior palace_of_westminster trevi_fountain st_peters_square sacre_coeur taj_mahal temple_nara_japan prague_old_town_square pantheon_exterior notre_dame_front_facade brandenburg_gate
do
  for size_subset in 5 10 # 20 50
  do
    if [ $size_subset -eq 5 ]
    then
      numconsistent=3
      nviews=5
    elif [ $size_subset -eq 10 ]
    then
      numconsistent=3
      nviews=10
    elif [ $size_subset -eq 20 ]
    then
      numconsistent=3
      nviews=20
    elif [ $size_subset -eq 50 ]
    then
      numconsistent=5
      nviews=20
    else
      numconsistent=7
      nviews=20
    fi

  python reconstruction_pipeline.py --dataset yfcc --scene "$scene"_"$size_subset" \
          --nviews $nviews --fusion colmap --filter_num_views $nviews --filter \
          --fusion_num_consistent $numconsistent $*
~

  done
done
