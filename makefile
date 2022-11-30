
TraditionalKmeans:
	python3 TraditionalFeatureExtraction.py Image_Data/ Test10 244
	python3 KMeansClustering.py Results/Traditional/Test10/Test10.csv Results/Traditional/Test10 stand 0.9 4 8
	python3 Evaluation.py Results/Traditional/Test10/ClusterResults.csv

kmeans2:
	python3 ImageClustering.py /Image_Data/ normalize 244 0.90 1 5 20 1

clean:
	rm *.out && rm *.log
