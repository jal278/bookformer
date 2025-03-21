#!/bin/bash

# only do following lines if False
if [ "True" = "False" ]; then
./do_transformation.sh vec-clf-m3-cml4.pkl cml 
./do_transformation.sh vec-clf-m3-gift.pkl gift
./do_transformation.sh vec-clf-m3-assigned.pkl assigned
./do_transformation.sh vec-clf-m3-disgust.pkl disgust
./do_transformation.sh vec-clf-m3-perma-p.pkl permap
./do_transformation.sh vec-clf-m3-perma-e.pkl permae
./do_transformation.sh vec-clf-m3-surprise.pkl surprise
./do_transformation.sh vec-clf-m3-wbe.pkl wbe
fi

#./do_transformation.sh "str:The weirdest book I have ever read." weird
#./do_transformation.sh "str:A literary page-turned." lpt
#./do_transformation.sh "str:The best book I have ever read." best
#./do_transformation.sh "str:The most erotic book I have ever read." erotic

#./do_transformation.sh vec-clf-m3-cml5.pkl cml 

 

if [ "True" = "False" ]; then
./do_transformation.sh vec-clf-m3-gift.pkl gift
./do_transformation.sh vec-clf-m3-assigned.pkl assigned
./do_transformation.sh vec-clf-m3-disgust.pkl disgust
./do_transformation.sh vec-clf-m3-perma-p.pkl permap
./do_transformation.sh vec-clf-m3-perma-e.pkl permae
./do_transformation.sh vec-clf-m3-surprise.pkl surprise
./do_transformation.sh vec-clf-m3-wbe.pkl wbe
./do_transformation.sh vec-clf-m3-erotic.pkl erotic
fi

./do_transformation.sh vec-clf-m3-weird.pkl weird
./do_transformation.sh vec-clf-m3-best.pkl best
./do_transformation.sh vec-clf-m3-erotic.pkl erotic
