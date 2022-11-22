
kmeans1:
	python3 ImageClustering.py /Image_Data/ normalize 244 0.95 1 10 20 1

kmeans2:
	python3 ImageClustering.py /Image_Data/ normalize 244 0.90 1 5 20 1

clean:
	rm *.out && rm *.log
