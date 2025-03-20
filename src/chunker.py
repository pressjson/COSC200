import os

from PIL import Image


def make_chunks(
    quality="hq",
    output_dir_number=None,
    output_dir="../data/chunks",
    chunk_size=256,
    delta=256,
):
    """makes chunks out of a given image

    args:
        quality: name of the image to be chunked, usually either "hq" or "lq"
        output_dir_number: the number of the subdirectory to be written to, which is output_dir/%number%
        output_dir: default is "../data/chunks", but can be set if using test data
        chunk_size: the size of the chunks
        delta: the offset for each chunk
        @TODO: make these set in a settings file/class
        @TODO: clean this up so it works generally
    """
    image = Image.open(
        "../data/images/" + str(output_dir_number) + "/" + quality + ".jpg"
    )
    width, height = image.size

    if not os.path.exists(os.path.join(output_dir, str(output_dir_number))):
        os.makedirs(os.path.join(output_dir, str(output_dir_number)))

    for x in range(0, width, delta):
        for y in range(0, height, delta):
            crop = image.crop((x, y, x + chunk_size, y + chunk_size))
            crop.save(
                os.path.join(
                    output_dir,
                    str(output_dir_number),
                    quality + "_" + str(x) + "_" + str(y) + ".jpg",
                )
            )


if __name__ == "__main__":
    print("chunking ../data/images")

    data_dir = "../data/images"

    for subdirectory in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, subdirectory)):
            print(subdirectory + "/" + filename)
            if "hq" in filename:
                make_chunks("hq", subdirectory)
            if "lq" in filename:
                make_chunks("lq", subdirectory)
