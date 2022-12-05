
TraditionalKmeans:
	python3 TraditionalFeatureExtraction.py Image_Data/ Test12 244
	python3 KMeansClustering.py Results/Traditional/Test12/Test12.csv Results/Traditional/Test12 Standardizing 0.95 2 4
	python3 Evaluation.py Results/Traditional/Test12/ClusterResults.csv Results/Traditional/Test12/

kmeans2:
	python3 ImageClustering.py /Image_Data/ normalize 244 0.90 1 5 20 1

clean:
	rm *.out && rm *.log

traditionalHDBSCAN:
	python3 TraditionalFeatureExtraction.py Image_Data/ Test13 244
	python3 HDBSCANCluster.py Results/Traditional/Test13/Test13.csv Results/Traditional/Test13 Standardizing 0.9 2 4
	python3 Evaluation.py Results/Traditional/Test13/HDBSCANResults.csv Results/Traditional/Test13/
	
