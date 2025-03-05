sudo chmod a+rw /dev/dma_heap/reserved
./cam-kms-tf.py -m ./ssdlite_mobiledet_coco_qat_postprocess.tflite -l labelmap.txt -d ~/mesa/build/src/gallium/targets/teflon/libteflon.so
