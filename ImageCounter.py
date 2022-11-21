import os


if __name__ == "__main__":
    
    data_dir = "/cephyr/NOBACKUP/groups/uu-it-gov"
    os.chdir(data_dir)
    base_dir = os.getcwd()
    print(base_dir)
    # for index, directories in enumerate(os.walk(full_data_dir_path)):
    #    for sample in directories[2]:
    #        if sample.endswith('.png'):
    #            full_path = directories[0] + "/" + sample
    #            all_paths.append(full_path)

    # Store all_paths in file or something