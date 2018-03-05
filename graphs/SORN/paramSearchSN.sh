#!/bin/bash

# Script to manage local parameter searches.

##########################
# ### Parameters ###
##########################
numMPI=1
numThreads=20
seed=10000
trials=$(seq 0 1 0)
changeFile="naze_4params.gsl"

# If 1D search - use ERROR_is_2D when is 2D or ERROR_not_in_use when not in use.
variables=ERROR_is_2D
additionalVariables_1=ERROR_is_2D
additionalVariables_2=ERROR_is_2D
additionalVariables_3=ERROR_is_2D

# If 4D search - use ERROR_is_1D when is 1D or ERROR_not_in_use when not in use.
variable_1=(0.8)      	# EIratio
variable_2=(0.9)    # tauSTDP
variable_3=(0.5)      # stimCfrac
variable_4=(0.1)      # ratioDelay

# If 2D search - use ERROR_is_1D when is 1D or ERROR_not_in_use when not in use.
#variables_X=`echo $(seq 1.0 2.0 3.0 4.0)`
#variables_Y=`echo $(seq 4.0 5.0 6.0 7.0)`
additionalVariables_1_X=(ERROR_not_in_use) # These should be of same size as above. E.g. (`echo $(seq 0.0 0.08334 0.834)`)
additionalVariables_2_X=(ERROR_not_in_use)
additionalVariables_3_X=(ERROR_not_in_use)
additionalVariables_1_Y=(ERROR_not_in_use)
additionalVariables_2_Y=(ERROR_not_in_use)
additionalVariables_3_Y=(ERROR_not_in_use)

if [ "$1" = "" ]; then
    echo "Please select a task:"
    echo "1  -- Run the parameter search"
    echo "2  -- Create index"
    exit 1
fi

if [ "$1" != "1" ] && [ "$1" != "2" ]; then
    echo "Please choose from one of the choices."
    exit 1
fi

# Get the number of dimensions
echo "1D or 2D or 4D? 1/2/4"
read dimensions
if [ "$dimensions" != "1" ]; then
    if [ "$dimensions" != "2" ]; then
        if [ "$dimensions" != "4" ]; then
	    echo "Number of dimensions can only be 1 or 2 or 4."
	    exit 1
        fi

    fi
fi

##################################
# ### Run the parameter search ###
##################################
if [ "$1" = "1" ]; then
    # Ask questions determining what to change etc.
    echo "Would you like to change the source file? y/n"
    read change
    if [ "$change" != "y" ]; then
        if [ "$change" != "n" ]; then
            echo "Input has to be y or n."
            exit 1
        fi
    fi
    if [ "$change" = "y" ]; then
	echo "Are there any additional variables to change? y/n"
	read additional
	if [ "$additional" != "y" ]; then
	    if [ "$additional" != "n" ]; then
		echo "Input has to be y or n."
		exit 1
	    fi
	fi
    fi
    # 4 dimension of variables
    if [ "$dimensions" = "4" ]; then
        # Ask more questions determining what to change etc.                    
        if [ "$change" = "y" ]; then
            if [ "$additional" = "y" ]; then
                echo "How many additional variables for the X dimension? 1/2/3"
                read additionalNum_X
                if [ "$additionalNum_X" != "0" ]; then                
                    if [ "$additionalNum_X" != "1" ]; then
                        if [ "$additionalNum_X" != "2" ]; then
                            if [ "$additionalNum_X" != "3" ]; then
                                echo "Input has to be 1, 2 or 3."
                                exit 1
                            fi
                        fi
                    fi
                fi
                echo "How many additional variables for the Y dimension? 1/2/3"
                read additionalNum_Y
                if [ "$additionalNum_Y" != "0" ]; then                
                    if [ "$additionalNum_Y" != "1" ]; then
                        if [ "$additionalNum_Y" != "2" ]; then
                            if [ "$additionalNum_Y" != "3" ]; then
                                echo "Input has to be 1, 2 or 3."
                                exit 1
                            fi
                        fi
                    fi
                fi                
            fi
        fi            
echo $variable_1 $variable_2 $variable_3 $variable_4
        # Run the parameter search
        ri=0;
        for f in ${variable_1[@]}; do
            gi=0;
            for g in ${variable_2[@]}; do
                hi=0;
                for h in ${variable_3[@]}; do
                    ii=0;
                    for i in ${variable_4[@]}; do
			for t in $trials; do            
			    # Make a copy of the gsl file to edit and run
			    cp $changeFile $changeFile.run;
			    # Edit it with the parameter values
			    sed -i 's/!XX_DIR_1_XX!/'"`echo $f | sed -r 's/\./_/'`"'/g' ./$changeFile.run
			    sed -i 's/!XX_DIR_2_XX!/'"`echo $g | sed -r 's/\./_/'`"'/g' ./$changeFile.run
			    sed -i 's/!XX_DIR_3_XX!/'"`echo $h | sed -r 's/\./_/'`"'/g' ./$changeFile.run
			    sed -i 's/!XX_DIR_4_XX!/'"`echo $i | sed -r 's/\./_/'`\/`echo $t`"'/g' ./$changeFile.run
			    sed -i 's/!XX_VAR_1_XX!/'"$f"'/g' ./$changeFile.run
			    sed -i 's/!XX_VAR_2_XX!/'"$g"'/g' ./$changeFile.run                    
			    sed -i 's/!XX_VAR_3_XX!/'"$h"'/g' ./$changeFile.run                    
			    sed -i 's/!XX_VAR_4_XX!/'"$i"'/g' ./$changeFile.run                    
			    # Edit with additional changes if requested
			    if [ "$additional" = "y" ]; then
				case "$additionalNum_X" in
				    "1") # one additional variable
					sed -i 's/!XX_ADD1_X_XX!/'"${additionalVariables_1_X[$fi]}"'/g' ./$changeFile.run
					;;
				    "2") # two additional variables
					sed -i 's/!XX_ADD1_X_XX!/'"${additionalVariables_1_X[$fi]}"'/g' ./$changeFile.run
					sed -i 's/!XX_ADD2_X_XX!/'"${additionalVariables_2_X[$fi]}"'/g' ./$changeFile.run
					;;
				    "3") # three additional variables
					sed -i 's/!XX_ADD1_X_XX!/'"${additionalVariables_1_X[$fi]}"'/g' ./$changeFile.run
					sed -i 's/!XX_ADD2_X_XX!/'"${additionalVariables_2_X[$fi]}"'/g' ./$changeFile.run
					sed -i 's/!XX_ADD3_X_XX!/'"${additionalVariables_3_X[$fi]}"'/g' ./$changeFile.run
					;;
				esac
				case "$additionalNum_Y" in
				    "1") # one additional variable
					sed -i 's/!XX_ADD1_Y_XX!/'"${additionalVariables_1_Y[$gi]}"'/g' ./$changeFile.run
					;;
				    "2") # two additional variables
					sed -i 's/!XX_ADD1_Y_XX!/'"${additionalVariables_1_Y[$gi]}"'/g' ./$changeFile.run
					sed -i 's/!XX_ADD2_Y_XX!/'"${additionalVariables_2_Y[$gi]}"'/g' ./$changeFile.run
					;;
				    "3") # three additional variables
					sed -i 's/!XX_ADD1_Y_XX!/'"${additionalVariables_1_Y[$gi]}"'/g' ./$changeFile.run
					sed -i 's/!XX_ADD2_Y_XX!/'"${additionalVariables_2_Y[$gi]}"'/g' ./$changeFile.run
					sed -i 's/!XX_ADD3_Y_XX!/'"${additionalVariables_3_Y[$gi]}"'/g' ./$changeFile.run
					;;
				esac                        
			    fi
			    # Run the simulation
			    ../../gsl/bin/gslparser -f $changeFile.run -t $numThreads -s $seed
			    # Remove the temporary file
			    rm $changeFile.run;
			    # Increment the seed
			    seed=$((seed+1))
			done
		        # Increment the counter
		        ii=$((ii+1))
		    done
		    # Increment the counter
		    hi=$((hi+1))
                done
                # Increment the counter
                gi=$((gi+1))
            done
            # Increment the counter
            ri=$((ri+1))
        done        
    fi
fi
        
######################
# ### Create index ###
######################
if [ "$1" = "2" ]; then
    # Ask additional information and create temporary index directory
    echo "Which image would you like indexed?"
    read file
    mkdir index;
    # 1 dimension of variables
    if [ "$dimensions" = "1" ]; then
	echo "Not coded yet, sorry"
    fi
    # 2 dimensions of variables
    if [ "$dimensions" = "2" ]; then
        for f in $variables_X; do
            for g in $variables_Y; do
                for t in $trials; do
                    if [ -f "`echo $f | sed -r 's/\./_/'`/`echo $g | sed -r 's/\./_/'`/$t/$file.png" ]; then
		        # Copy it to an index directory
		        cp "`echo $f | sed -r 's/\./_/'`/`echo $g | sed -r 's/\./_/'`/$t/$file.png" "index/`echo $f | sed -r 's/\./_/'`-`echo $g | sed -r 's/\./_/'`.png.$t"
                    else
                        # Doesn't exist; create a blank one
                        echo "Warning: file `echo $f | sed -r 's/\./_/'`/`echo $g | sed -r 's/\./_/'`/$t/$file.png doesn't exist; creating a blank one."
                        convert -size 640x480 xc:white "index/temp.png"
                        mv "index/temp.png" "index/`echo $f | sed -r 's/\./_/'`-`echo $g | sed -r 's/\./_/'`.png.$t"
                    fi
                done
	    done
	done
        # Index them together with eacha trial getting a different index image
        for t in $trials; do
            montage -density 640x480 -geometry 640x480 `ls -v index/*.png.$t` -tile 24x30 $file""_Index_t$t.png
        done
        # Remove these temporary files
        rm -r index
    fi
fi
