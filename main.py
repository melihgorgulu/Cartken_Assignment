from morphop.mop import MorphOps
import utils
import os


# Ref: https://www.geeksforgeeks.org/boundary-extraction-of-image-using-matlab/

def main():
    binary_map = utils.get_binary_map(os.getenv("DATA_PATH"))
    morph_op = MorphOps()
    boundry_map = morph_op.find_boundries(binary_map)
    utils.vis_binary_map(boundry_map)
    segments = morph_op.extract_boundry_segments(boundry_map)
    # some important segments index = 11, 33 , 46
    # utils.vis_segments(segments, start_idx=11, end_index=33)
    utils.vis_important_segments(segments)


if __name__ == "__main__":
    main()
