

add-yaml:
	cp thermal_image_dataset.yaml yolov5/data/

setup-run:
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit_mini/labels
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit_mini/labels
	mkdir /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit_mini/labels
	mkdir /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit_mini/labels
	cp /scratch/sfberrio/FLIR_ADAS_1_3/train/yolo_labels/* /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit_mini/labels
	cp /scratch/sfberrio/FLIR_ADAS_1_3/val/yolo_labels/* /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit_mini/labels

full-setup-run:
	cp /scratch/sfberrio/FLIR_ADAS_1_3/train/yolo_labels/* /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit
	cp /scratch/sfberrio/FLIR_ADAS_1_3/val/yolo_labels/* /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit

