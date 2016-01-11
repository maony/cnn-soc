if [ "$1" == "gpu" ]
then
    make -f Makefile.gpu    
    ./matrix_mult
fi

if [ "$1" == "sim" ]
then
    rm -f matrix_mult matrix_mult.aocx channelizer channelizer.aocx
    echo "-------------------------------------------------------------------"
    echo "-----------------------build $1------------------------------------"
    echo "-------------------------------------------------------------------"
    # build opencl kernel emulator
    echo "----------build opencl kernel emulator----------"
    # aoc -march=emulator -v --board de1soc_sharedonly device/channel_conv_8x8.cl -o bin/matrix_mult_sim.aocx --report
    aoc -march=emulator -v --board de1soc_sharedonly device/channel_conv_16x16.cl -o bin/matrix_mult_sim.aocx --report
    # aoc -march=emulator -v --board de1soc_sharedonly vadd.cl -o bin/matrix_mult_sim.aocx --report
    echo "-------------------------------------------------------------------"
    echo "----------build host---------"
    echo "-------------------------------------------------------------------"
    make -f Makefile.emulate
    # make -f Makefile.add
    cp bin/matrix_mult_sim.aocx ./channelizer.aocx
    echo "-------------------------------------------------------------------"
    echo "----------run---------------"
    echo "-------------------------------------------------------------------"
    env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 LD_LIBRARY_PATH=/home/maoningyuan/altera/15.1/hld/linux64/hld LD_LIBRARY_PATH=/home/maoningyuan/altera/15.1/hld/host/linux64/lib ./channelizer
fi

if [ "$1" == "arm" ]
then
    rm -f matrix_mult matrix_mult.aocx
    echo "-------------------------------------------------------------------"
    echo "----------build $1----------"
    echo "-------------------------------------------------------------------"
    echo "----------build opencl kernel----------"
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/channelizer.cl -o bin/channelizer.aocx --report
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/channel_conv_8x8.cl -o bin/conv_8x8_profile.aocx --report --profile
    # aoc -v --fp-relaxed --fpc -O2 --board de1soc_sharedonly device/channel_conv_8x8x1.cl -o bin/conv_8x8x1_profile.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/channel_conv_8x8.cl -o bin/conv_8x8_opt_profile.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/channel_conv_8x8.cl -o bin/conv_8x8_opt_row1_profile.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/channel_conv_8x8.cl -o bin/conv_8x8_opt_novec_profile.aocx --report --profile
    aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/channel_conv_16x16.cl -o bin/conv_16x16_profile.aocx --report --profile
    # aoc -v --board --sw-dimm-partition --fp-relaxed --board de1soc_sharedonly vadd.cl -o bin/matrix_mult.aocx --report
    # aoc -v --sw-dimm-partition --fp-relaxed --util 95 -O3 --board de1soc_sharedonly matrix_mult.cl -o bin/matrix_mult_8x8_default.aocx --report
    echo "-------------------------------------------------------------------"
    echo "----------build host---------"
    echo "-------------------------------------------------------------------"
    make -f Makefile
    cp bin/conv_8x8_opt_row1_profile.aocx ./channelizer.aocx
    echo "-------------------------------------------------------------------"
    echo "----------scp transfer---------------"
    echo "-------------------------------------------------------------------"
    scp channelizer.aocx root@192.168.2.143:/home/root/mny
    scp channelizer      root@192.168.2.143:/home/root/mny
    # ssh root@192.168.2.143
fi

if [ "$1" == "host" ]
then
    rm -f channelizer channelizer.aocx 
    echo "-------------------------------------------------------------------"
    echo "----------build $1 only----------"
    echo "----------build host---------"
    echo "-------------------------------------------------------------------"
    make -f Makefile
    # cp bin/matrix_mult_simd.aocx ./matrix_mult.aocx
    # cp bin/matrix_mult_16x8.aocx ./matrix_mult.aocx
    cp bin/conv_8x8_opt_profile.aocx ./channelizer.aocx
    echo "-------------------------------------------------------------------"
    echo "----------scp transfer---------------"
    echo "-------------------------------------------------------------------"
    scp channelizer.aocx root@192.168.2.143:/home/root/mny
    scp channelizer      root@192.168.2.143:/home/root/mny
    # ssh root@192.168.2.143
fi

if [ "$1" == "report" ]
then
    echo "-------------------------------------------------------------------"
    echo "----------build $1----------"
    echo "-------------------------------------------------------------------"
    aoc -c -v --board --sw-dimm-partition --fp-relaxed --board de1soc_sharedonly matrix_mult.cl -o bin/matrix_mult.aoco --report
fi
