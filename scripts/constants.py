# Work space directory
HOME_DIR = '/root/salgan/workspace/'

# Path to raw data
pathToImages = '/root/salgan/workspace/dataset/raw_data/images'
pathToMaps = '/root/salgan/workspace/dataset/raw_data/salmaps'
pathToFixationMaps = '/root/salgan/workspace/dataset/raw_data/fixation'

# Path to processed data
pathOutputImages = '/root/salgan/workspace/dataset/processed_data/images'
pathOutputMaps = '/root/salgan/workspace/dataset/processed_data/salmaps'
pathToPickle = '/root/salgan/workspace/dataset/processed_data/pickle'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/users/jpang/scratch-local/salicon_data/320x240/fix_trainData.pickle'
VAL_DATA_DIR = '/home/users/jpang/scratch-local/salicon_data/320x240/fix_validationData.pickle'
TEST_DATA_DIR = '/home/users/jpang/scratch-local/salicon_data/256x192/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/root/salgan/workspace/weights/vgg16.pkl'

# Input image and saliency map size
# this is not the size of the test image, do not change after training..
#INPUT_SIZE = (256, 192)
INPUT_SIZE = (1392, 512)

# Directory to keep snapshots
DIR_TO_SAVE = 'test'
