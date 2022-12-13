
TraditionalKmeans:
	python3 TraditionalFeatureExtraction.py Image_Data/ Test12 640 350
	python3 KMeansClustering.py Results/Test12/Test12.csv Results/Test12 Normalize 0.8 10 20
	python3 Evaluation.py Results/Test12/KMeansResults.csv Results/Test12/ 15

kmeans2:
	python3 ImageClustering.py /Image_Data/ normalize 244 0.90 1 5 20 1

clean:
	rm *.out && rm *.log

TraditionalHDBSCAN:
	python3 TraditionalFeatureExtraction.py Image_Data/ Test13 640 350
	python3 HDBSCANCluster.py Results/Test13/Test13.csv Results/Test13 Standardizing 0.8 3 10
	python3 Evaluation.py Results/Test13/HDBSCANResults.csv Results/Test13/ 15
	
