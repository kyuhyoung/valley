#python main2.py -P param.txt -c Ch1 Ch2 -p B01 B03 B05 B07 ./3rd_data_validation_2
#python pso_opt.py -C max -k thres -s 20 -b bounds.txt -c Ch1 Ch2 -p B01 B03 F03 D02 -g ./RawData/thres_gt.txt ./RawData
#python pso_opt.py -C max -k thres -s 20 -b bounds.txt -c Ch1 Ch2 -p pc.txt -g ./RawData/thres_gt.txt ./RawData
#python pso_opt.py -C avg -k thres -s 20 -b bounds.txt -c Ch1 Ch2 -p pc.txt -g ./RawData/thres_gt.txt ./RawData
#python pso_opt.py -C max -k mean -s 20 -b bounds.txt -c Ch1 Ch2 -p pc.txt -g ./RawData/thres_gt.txt ./RawData
python pso_opt.py -C avg -k mean -s 20 -b bounds.txt -c Ch1 Ch2 -g ./RawData/thres_gt.txt ./RawData

    
