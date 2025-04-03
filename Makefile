USERNAME := $(shell whoami)

add-yaml:
	cp thermal_image_dataset.yaml yolov5/data/

setup-run:
	python gen_yaml.py
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit_mini_$(USERNAME)/labels
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit_mini_$(USERNAME)/labels
	mkdir /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit_mini_$(USERNAME)/labels
	mkdir /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit_mini_$(USERNAME)/labels
	cp /scratch/sfberrio/FLIR_ADAS_1_3/train/yolo_labels_$(USERNAME)/* /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit_mini_$(USERNAME)/labels
	cp /scratch/sfberrio/FLIR_ADAS_1_3/val/yolo_labels_$(USERNAME)/* /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit_mini_$(USERNAME)/labels

full-setup-run:
	cp /scratch/sfberrio/FLIR_ADAS_1_3/train/yolo_labels/* /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit
	cp /scratch/sfberrio/FLIR_ADAS_1_3/val/yolo_labels/* /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit

clean-full:
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/train/thermal_8_bit/*txt
	rm -rf /scratch/sfberrio/FLIR_ADAS_1_3/val/thermal_8_bit/*txt

