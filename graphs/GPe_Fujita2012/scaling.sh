#!/bin/bash

SAMPLE=3
FILE=model.gsl

CONSTANT_THREAD=28
CONSTANT_SIM_TIME=10000
CONSTANT_PHASE_1=run1
CONSTANT_PHASE_2=run2
CONSTANT_PHASE_3=run3
CONSTANT_PHASE_4=run3
CONSTANT_PHASE_5=run1
CONSTANT_PHASE_6=run1
CONSTANT_PHASE_7=run2
CONSTANT_PHASE_8=run3
CONSTANT_MPI=1

THREAD=(1 2 4 8 16 28 56)
SIM_TIME=(10000 20000 50000 100000 250000 500000)
PHASE_1=(run1 run1 run1 run1)
PHASE_2=(run2 run2 run2 run2)
PHASE_3=(run1 run3 run1 run3)
PHASE_4=(run1 run3 run3 run1)
PHASE_5=(run3 run1 run1 run3)
PHASE_6=(run3 run1 run1 run3)
PHASE_7=(run1 run2 run2 run1)
PHASE_8=(run2 run3 run3 run2)
MPI=(1 2 3 4 5 6 7 8 9 10 11 12)

if [ "$1" = "" ]; then
    echo "Please select a task:"
    echo "0 -- All of the below"
    echo "1 -- Scale threads"
    echo "2 -- Scale simulation time"
    echo "3 -- Scale phases"
    echo "4 -- Scale MPI processes"
    echo "5 -- Scale threads and MPI processes"
    exit 1
fi

if [ "$1" != "0" ] && [ "$1" != "1" ] && [ "$1" != "2" ] && [ "$1" != "3" ] \
       && [ "$1" != "4" ] && [ "$1" != "5" ]; then
    echo "Please choose from one of the choices."
    exit 1
fi

replace_constant() {
    if [ ! $1 = 1 ]; then
        sed -i 's/!XX_SIM_TIME_XX!/'"$CONSTANT_SIM_TIME"'/g' ./$FILE.run
    fi
    if [ ! $1 = 2 ]; then
        sed -i 's/!XX_PHASE_1_XX!/'"$CONSTANT_PHASE_1"'/g' ./$FILE.run
        sed -i 's/!XX_PHASE_2_XX!/'"$CONSTANT_PHASE_2"'/g' ./$FILE.run
        sed -i 's/!XX_PHASE_3_XX!/'"$CONSTANT_PHASE_3"'/g' ./$FILE.run
        sed -i 's/!XX_PHASE_4_XX!/'"$CONSTANT_PHASE_4"'/g' ./$FILE.run
        sed -i 's/!XX_PHASE_5_XX!/'"$CONSTANT_PHASE_5"'/g' ./$FILE.run
        sed -i 's/!XX_PHASE_6_XX!/'"$CONSTANT_PHASE_6"'/g' ./$FILE.run
        sed -i 's/!XX_PHASE_7_XX!/'"$CONSTANT_PHASE_7"'/g' ./$FILE.run
        sed -i 's/!XX_PHASE_8_XX!/'"$CONSTANT_PHASE_8"'/g' ./$FILE.run
    fi
}

threads() {
    rm times_threads.txt 2> /dev/null
    echo "#threads "${THREAD[@]} >> times_threads.txt
    for N in $(seq 1 $SAMPLE); do
        for T in "${THREAD[@]}"; do
            # Create temp file
            cp $FILE $FILE.run
            replace_constant 0;
            # Run temp file
            (/usr/bin/time -f "real %e" \
                           ../../gsl/bin/gslparser -t $T -f $FILE.run -s `date +%s`) \
                2>&1 | tee /dev/tty | grep "real" | awk '{printf $(2)" "}' >> times_threads.txt
        done
        echo >> times_threads.txt
    done
}

sim_times() {
    rm times_sim_times.txt 2> /dev/null
    echo "#sim_times "${SIM_TIME[@]} >> times_sim_times.txt
    for N in $(seq 1 $SAMPLE); do
        for ST in "${SIM_TIME[@]}"; do
            # Create temp file
            cp $FILE $FILE.run
            replace_constant 1;
            sed -i 's/!XX_SIM_TIME_XX!/'"$ST"'/g' ./$FILE.run
            # Run temp file
            (/usr/bin/time -f "real %e" \
                           ../../gsl/bin/gslparser -t $CONSTANT_THREAD -f $FILE.run -s `date +%s`) \
                2>&1 | tee /dev/tty | grep "real" | awk '{printf $(2)" "}' >> times_sim_times.txt
        done
        echo >> times_sim_times.txt
    done
}

phases() {
    rm times_phases.txt 2> /dev/null
    echo "#phases "${PHASE_1[@]} >> times_phases.txt
    for N in $(seq 1 $SAMPLE); do
        for ((p=0;p<${#PHASE_1[@]};++p)); do
            # Create temp file
            cp $FILE $FILE.run
            replace_constant 2;
            sed -i 's/!XX_PHASE_1_XX!/'"${PHASE_1[p]}"'/g' ./$FILE.run
            sed -i 's/!XX_PHASE_2_XX!/'"${PHASE_2[p]}"'/g' ./$FILE.run
            sed -i 's/!XX_PHASE_3_XX!/'"${PHASE_3[p]}"'/g' ./$FILE.run
            sed -i 's/!XX_PHASE_4_XX!/'"${PHASE_4[p]}"'/g' ./$FILE.run
            sed -i 's/!XX_PHASE_5_XX!/'"${PHASE_5[p]}"'/g' ./$FILE.run
            sed -i 's/!XX_PHASE_6_XX!/'"${PHASE_6[p]}"'/g' ./$FILE.run
            sed -i 's/!XX_PHASE_7_XX!/'"${PHASE_7[p]}"'/g' ./$FILE.run
            sed -i 's/!XX_PHASE_8_XX!/'"${PHASE_8[p]}"'/g' ./$FILE.run
            # Run temp file
            (/usr/bin/time -f "real %e" \
                           ../../gsl/bin/gslparser -t $CONSTANT_THREAD -f $FILE.run -s `date +%s`) \
                2>&1 | tee /dev/tty | grep "real" | awk '{printf $(2)" "}' >> times_phases.txt
        done
        echo >> times_phases.txt
    done
}

MPIs() {
    rm times_MPIs.txt 2> /dev/null
    echo "#MPIs "${MPI[@]} >> times_MPIs.txt
    for N in $(seq 1 $SAMPLE); do
        for M in "${MPI[@]}"; do
            # Create temp file
            cp $FILE $FILE.run
            replace_constant 0;
            # Run temp file
            (/usr/bin/time -f "real %e" \
                           mpiexec --mca plm_rsh_no_tree_spawn 1 --hostfile ../../../my_hosts \
                           --mca btl ^openib -n $M \
                           ../../gsl/bin/gslparser -t $CONSTANT_THREAD -f $FILE.run -s `date +%s`) \
                2>&1 | tee /dev/tty | grep "real" | awk '{printf $(2)" "}' >> times_MPIs.txt
        done
        echo >> times_MPIs.txt
    done
}

threads_and_MPIs() {
    rm times_threads_and_MPIs.txt 2> /dev/null
    echo "#rows threads "${THREAD[@]} >> times_threads_and_MPIs.txt
    echo "#cols MPIs "${MPI[@]} >> times_threads_and_MPIs.txt
    echo >> times_threads_and_MPIs.txt
    for N in $(seq 1 $SAMPLE); do
        for T in "${THREAD[@]}"; do
            for M in "${MPI[@]}"; do
                # Create temp file
                cp $FILE $FILE.run
                replace_constant 0;
                # Run temp file
                (/usr/bin/time -f "real %e" \
                               mpiexec -n $M ../../gsl/bin/gslparser -t $T -f $FILE.run -s `date +%s`) \
                    2>&1 | tee /dev/tty | grep "real" | awk '{printf $(2)" "}' >> times_threads_and_MPIs.txt
            done
            echo >> times_threads_and_MPIs.txt
        done
        echo >> times_threads_and_MPIs.txt
        echo >> times_threads_and_MPIs.txt
    done
}

case $1 in
    "0")
        threads;
        sim_times;
        phases;
        MPIs;
        threads_and_MPIs;
        ;;
    "1")
        threads;
        ;;
    "2")
        sim_times;
        ;;
    "3")
        phases;
        ;;
    "4")
        MPIs;
        ;;
    "5")
        threads_and_MPIs;
        ;;
esac
rm $FILE.run 2> /dev/null

