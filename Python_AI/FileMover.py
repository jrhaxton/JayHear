import os
from natsort import natsorted
import sys
import shutil
from tqdm import tqdm


def tempMoveFiles(source, destination, amount):
    listOfFiles = natsorted(os.listdir(source))
    count = 0
    for file in tqdm(listOfFiles):
        if (count == amount):
            break
        print(count)
        count += 1
        shutil.move(source + '/' + file, destination)  # source,destination


def main(arguments):
    # """Main func."""

    source = "/home/braden/Documents/mozilla_MP3/clips"
    destination = "/home/braden/Environments/Research/Audio/Research(Refactored)/Mozilla_MP3"
    amount = 2000
    tempMoveFiles(source, destination, amount)


if __name__ == "__main__":
    main(sys.argv[1:])
