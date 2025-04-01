import os

from .. import settings

from PIL import Image


def make_chunks(
    quality="hq",
    input_file_location="../data/images",
    output_dir="../data/chunks",
    chunk_size=settings.IMAGE_SIZE[0],
    delta=settings.DELTA,
):
    """makes chunks out of a given image

    args:
        quality: name of the image to be chunked, usually either "hq" or "lq"
        input_file_location: the location of the file, complete with everything
        output_dir: default is "../data/chunks", but can be set if using test data. will make the dir
        chunk_size: the size of the chunks
        delta: the offset for each chunk
    """
    image = Image.open(input_file_location)
    width, height = image.size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for x in range(0, width, delta):
        for y in range(0, height, delta):
            crop = image.crop((x, y, x + chunk_size, y + chunk_size))
            crop.save(
                os.path.join(
                    output_dir,
                    quality + "_" + str(x) + "_" + str(y) + ".jpg",
                )
            )


def chunk_data_images_dir():
    print("chunking ../data/images")

    data_dir = "../data/images"

    for subdirectory in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, subdirectory)):
            print(subdirectory + "/" + filename)
            if "hq" in filename:
                make_chunks(
                    quality="hq",
                    input_file_location=os.path.join(data_dir, subdirectory, filename),
                    output_dir=os.path.join("../data/chunks", subdirectory),
                )
            if "lq" in filename:
                make_chunks(
                    quality="lq",
                    input_file_location=os.path.join(data_dir, subdirectory, filename),
                    output_dir=os.path.join("../data/chunks", subdirectory),
                )


if __name__ == "__main__":
    chunk_data_images_dir()
