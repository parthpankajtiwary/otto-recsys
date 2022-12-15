import glob, os


def submit_file(message, path):
    list_of_files = glob.glob(path + "*.csv")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    command = f'kaggle competitions submit -f {latest_file} -m "{message}" -q otto-recommender-system'
    return os.system(command)


if __name__ == "__main__":
    submit_file(message="without word2vec")
