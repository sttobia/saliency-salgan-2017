# Work space directory
HOME_DIR = '/root/workspace/'

# Path to raw data
pathToImages = '/root/workspace/dataset/raw_data/images/'
pathToMaps = '/root/workspace/dataset/raw_data/salmaps/'
pathToFixationMaps = '/root/workspace/dataset/raw_data/fixation/'

# Path to processed data
pathOutputImages = '/root/workspace/dataset/processed_data/images/'
pathOutputMaps = '/root/workspace/dataset/processed_data/salmaps/'
pathToPickle = '/root/workspace/dataset/processed_data/pickle/'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/root/workspace/pickle/fix_trainData.pickle'
VALIDATION_DATA_DIR = '/root/workspace/pickle/fix_validationData.pickle'
TEST_DATA_DIR = '/root/workspace/pickle/fix_testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/root/workspace/weights/vgg16.pkl'

# Input image and saliency map size
# this is not the size of the test image, do not change after training..
INPUT_SIZE = (256, 192)
#INPUT_SIZE = (1392, 512)

# Directory to keep snapshots of current training
DIR_TO_SAVE = '/root/workspace/output'
