USERNAME := $(shell whoami)
IMG_NUM := 100
START_IDX := 0


clean:
	make delete-labels
	make delete-mini

run-all:
	python ./convert_json_to_yolo.py --img_num $(IMG_NUM) --start_idx $(START_IDX)
	python ./create_thermal_8_bit_mini.py --img_num $(IMG_NUM) --start_idx $(START_IDX)

delete-labels:
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/val/yolo_labels_$(USERNAME)/
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/train/yolo_labels_$(USERNAME)/

delete-mini:
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit_mini_$(USERNAME)/
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit_mini_$(USERNAME)/
