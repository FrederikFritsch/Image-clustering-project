import os


if __name__ == "__main__":
    print("Hello World")
    data_dir = "/cephyr/NOBACKUP/groups/uu-it-gov/top20/frames"
    base_dir = os.getcwd()
    print(base_dir)
    tmp = 0
    for index, directories in enumerate(os.walk(data_dir)):
        for sample in directories[2]:
            if sample.endswith('.png'):
                #full_path = directories[0] + "/" + sample
                #all_paths.append(full_path)
                tmp += 1
    print(tmp)
    try: 
        os.mkdir("Results/Test")
    except:
        pass
    results_dir = base_dir + "/Results/"
    print(results_dir)
    # Store all_paths in file or something
