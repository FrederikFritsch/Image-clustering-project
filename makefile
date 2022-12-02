
TraditionalKmeans:
	python3 TraditionalFeatureExtraction.py Image_Data/ Test12 244
	python3 KMeansClustering.py Results/Traditional/Test12/Test12.csv Results/Traditional/Test12 Standardizing 0.95 10 25
	python3 Evaluation.py Results/Traditional/Test12/ClusterResults.csv Results/Traditional/Test12/

kmeans2:
	python3 ImageClustering.py /Image_Data/ normalize 244 0.90 1 5 20 1

clean:
	rm *.out && rm *.log
