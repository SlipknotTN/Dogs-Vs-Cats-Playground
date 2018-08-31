import glob
import os
from torch.utils.data import Dataset


class StdFileSystemDataset(Dataset):
    """Dog Cat dataset, works with generic dataset where each subdirectory contain a class images"""

    def __init__(self, root_dir, transform):
        """
        Args:
            root_dir (string): Directory with classes subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        # Read file system and save num classes, num of images, tuples (image path, gt)

        self.classes = next(os.walk(self.root_dir))[1]

        for class_index, class_name in enumerate(self.classes):

            for file in sorted(glob.glob(os.path.join(self.root_dir, class_name) + "/*.jpg")):

                self.samples.append((file, class_index))

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        raise NotImplementedError

        # image_name = os.path.join(self.root_dir,
        #                           self.key_pts_frame.iloc[idx, 0])
        #
        # image = mpimg.imread(image_name)
        #
        # # if image has an alpha color channel, get rid of it
        # if (image.shape[2] == 4):
        #     image = image[:, :, 0:3]
        #
        # key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        # key_pts = key_pts.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'keypoints': key_pts}
        #
        # if self.transform:
        #     sample = self.transform(sample)
        #
        # return sample