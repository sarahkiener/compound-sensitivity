#!/bin/bash


ext=txt
metric=metric_name


echo $ext

#cd ./wmt20/wmt20metrics/newstest2020/$ext 
#cd ~/$ext 
testset=newstest2020
#score newstest2020
for lp in `ls -d system-outputs/* | cut -d/ -f2`; do
    # echo  ${lp:0:2} 
    src=sources/${testset}-${lp:0:2} #${lp:3:5}-src.${lp:0:2}.${ext}
    src=sources/${testset}-${lp:0:2}${lp:3:5}-src.${lp:0:2}.${ext}
    for ref in references/*-${lp:0:2}${lp:3:5}-ref.${lp:3:5}.${ext}; do
        basename=$(basename $ref);
        refset=$(echo $basename | cut -d- -f1); 
        for hyp in system-outputs/$lp/*; do
            basename=$(basename $hyp); 
            systemname=$(echo $basename | cut -d. -f3- | rev | cut -d. -f2- | rev ); 
            if [ $refset = newstest2020 ] && [ $systemname = 'Human-A.0' ]; then
                continue 
            fi
            if [ $refset = newstestB2020 ] && [ $systemname = 'Human-B.0' ]; then
                continue
            fi

            if [ $refset = newstestP2020 ] && [ $systemname = 'Human-P.0' ]; then
                continue
            fi
            echo "$metric $lp $testset $refset $systemname"
            # &lt;YOUR EVALUATION TOOL&gt; --hypothesis=$hyp --source=$src  --reference=$ref              
        done  
    done
done



# #Evaluate MT systems with multirefs, testset will be newstestM2020
# refset=newstestM2020
# for lp in de-en en-de en-zh ru-en zh-en ; do 
    # ref=`ls -d references/*-${lp:0:2}${lp:3:5}*`
    # src=sources/$testset-${lp:0:2}${lp:3:5}-src.${lp:0:2}.$ext
    # for hyp in system-outputs/$lp/*; do
        # basename=$(basename $hyp)
        # systemname=$(echo $basename | cut -d. -f3- | rev | cut -d. -f2- | rev ); 
        # if [ $systemname =  'Human-A.0' ] || [ $systemname = 'Human-B.0' ]  || [ $systemname = 'Human-P.0' ]; then
                # continue 
        # fi
        # echo "$metric $lp $testset $refset $systemname"   
        # # &lt;YOUR EVALUATION TOOL&gt; --hypothesis=$hyp --source=$src  --reference=$ref 
    # done
# done
 
# #eval each en-de ref against the other two
# lp=en-de
# testset=newstest2020
# refset=newstestM2020
# systemname='Human-B.0'
# ref="references/newstest2020-ende-ref.de.${ext} references/newstestP2020-ende-ref.de.${ext} "
# hyp="system-outputs/en-de/$testset.en-de.Human-B.0.$ext"
# # &lt;YOUR EVALUATION TOOL&gt; --hypothesis=$hyp --source=$src  --reference=$ref 
# echo "$metric $lp $testset $refset $systemname"   

# systemname='Human-A.0'
# ref="references/newstestB2020-ende-ref.de.${ext} references/newstestP2020-ende-ref.de.${ext} "
# hyp="system-outputs/en-de/$testset.en-de.Human-A.0.$ext"
# # &lt;YOUR EVALUATION TOOL&gt; --hypothesis=$hyp --source=$src  --reference=$ref 
# echo "$metric $lp $testset $refset $systemname" 

# systemname='Human-P.0'
# ref="references/newstest2020-ende-ref.de.${ext} references/newstestB2020-ende-ref.de.${ext} "
# hyp="system-outputs/en-de/$testset.en-de.Human-P.0.$ext"
# # &lt;YOUR EVALUATION TOOL&gt; --hypothesis=$hyp --source=$src  --reference=$ref 
# echo "$metric $lp $testset $refset $systemname"   



# #score testsuites.  
# cd ~/wmt20/wmt20-news-task-primary-submissions/testsuites2020/$ext  
# testset=testsuites2020
# refset=testsuites2020
# for lp in `ls -d system-outputs/* | cut -d/ -f2`; do
    # src=sources/$testset-${lp:0:2}${lp:3:5}-src-ts.${lp:0:2}.${ext}
    # for ref in references/*-${lp:0:2}${lp:3:5}-ref-ts.${lp:3:5}.${ext}; do
        # for hyp in system-outputs/$lp/*; do
            # basename=$(basename $hyp);   
            # systemname=$(echo $basename | cut -d. -f3- | rev | cut -d. -f2- | rev );  
            # echo "$metric $lp $testset $refset $systemname"         
            # # &lt;YOUR EVALUATION TOOL&gt; --hypothesis=$hyp --source=$src  --reference=$ref 
        # done  
    # done
# done   