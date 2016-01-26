if [ "$1" == "gpu" ]
then
    make -f Makefile.gpu    
    ./matrix_mult
fi

if [ "$1" == "sim" ]
then
    rm -f cnn cnn.aocx
    echo "-------------------------------------------------------------------"
    echo "-----------------------build $1------------------------------------"
    echo "-------------------------------------------------------------------"
    # build opencl kernel emulator
    echo "----------build opencl kernel emulator----------"
    aoc -march=emulator --fp-relaxed --fpc -v --board de1soc_sharedonly device/cnn_define.cl -o bin/im2col_sim.aocx --report
    echo "-------------------------------------------------------------------"
    echo "----------build host---------"
    echo "-------------------------------------------------------------------"
    make -f Makefile.emulate
    cp bin/im2col_sim.aocx ./cnn.aocx
    echo "-------------------------------------------------------------------"
    echo "----------run---------------"
    echo "-------------------------------------------------------------------"
    env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 LD_LIBRARY_PATH=/home/maoningyuan/altera/15.1/hld/linux64/hld LD_LIBRARY_PATH=/home/maoningyuan/altera/15.1/hld/host/linux64/lib ./cnn
fi

if [ "$1" == "arm" ]
then
    rm -f cnn.aocx cnn
    echo "-------------------------------------------------------------------"
    echo "----------build $1----------"
    echo "-------------------------------------------------------------------"
    echo "----------build opencl kernel----------"
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/im2col.cl -o bin/im2col_1x1.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/im2col.cl -o bin/im2col_task.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/cnn_define.cl -o bin/max_pooling.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/cnn_define.cl -o bin/relu.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/cnn_define.cl -o bin/softmax.aocx --report --profile
    aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/cnn_define.cl -o bin/cnn2inner.aocx --report
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/cnn_define.cl -o bin/inner_product.aocx --report --profile
    # aoc -v --fp-relaxed --fpc --board de1soc_sharedonly device/cnn_define.cl -o bin/conv.aocx --report
    # aoc -v --sw-dimm-partition --fp-relaxed --util 95 -O3 --board de1soc_sharedonly matrix_mult.cl -o bin/matrix_mult_8x8_default.aocx --report
    echo "-------------------------------------------------------------------"
    echo "----------build host---------"
    echo "-------------------------------------------------------------------"
    make -f Makefile
    cp bin/im2col_1x1.aocx ./cnn.aocx
    echo "-------------------------------------------------------------------"
    echo "----------scp transfer---------------"
    echo "-------------------------------------------------------------------"
    scp cnn.aocx root@192.168.2.143:/home/root/mny
    scp cnn      root@192.168.2.143:/home/root/mny
    # ssh root@192.168.2.143
fi

if [ "$1" == "host" ]
then
    rm -f cnn cnn.aocx 
    echo "-------------------------------------------------------------------"
    echo "----------build $1 only----------"
    echo "----------build host---------"
    echo "-------------------------------------------------------------------"
    make -f Makefile
    # cp bin/matrix_mult_simd.aocx ./matrix_mult.aocx
    # cp bin/matrix_mult_16x8.aocx ./matrix_mult.aocx
    # cp bin/im2col_1x1.aocx ./cnn.aocx
    cp bin/cnn2inner.aocx ./cnn.aocx
    echo "-------------------------------------------------------------------"
    echo "----------scp transfer---------------"
    echo "-------------------------------------------------------------------"
    scp cnn.aocx root@192.168.2.143:/home/root/mny
    scp cnn      root@192.168.2.143:/home/root/mny
    # ssh root@192.168.2.143
fi

if [ "$1" == "report" ]
then
    echo "-------------------------------------------------------------------"
    echo "----------build $1----------"
    echo "-------------------------------------------------------------------"
    aoc -c -v --board --sw-dimm-partition --fp-relaxed --board de1soc_sharedonly matrix_mult.cl -o bin/matrix_mult.aoco --report
fi
