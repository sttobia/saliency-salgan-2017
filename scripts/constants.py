# Work space directory
HOME_DIR = '/root/workspace/'

# Path to raw data -> USED IN PREPROCESSING
pathToImages = '/root/workspace/dataset/90_1_9/raw_data/images/'
pathToMaps = '/root/workspace/dataset/90_1_9/raw_data/salmaps/'
pathToFixationMaps = '/root/workspace/dataset/90_1_9/raw_data/fixation/'

# Path to processed data -> USED IN PREPROCESSING
pathOutputImages = '/root/workspace/dataset/90_1_9/processed_data/images/'
pathOutputMaps = '/root/workspace/dataset/90_1_9/processed_data/salmaps/'
pathToPickle = '/root/workspace/dataset/90_1_9/processed_data/pickle/'

# Path to pickles which contains processed data -> USED IN TRAINING
TRAIN_DATA_DIR = '/root/workspace/dataset/50_25_25/processed_data/pickle/trainData.pickle'
VALIDATION_DATA_DIR = '/root/workspace/dataset/50_25_25/processed_data/pickle/validationData.pickle'
TEST_DATA_DIR = '/root/workspace/dataset/50_25_25/processed_data/pickle/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/root/workspace/weights/vgg16.pkl'

# Input image and saliency map size
# this is not the size of the test image, do not change after training..
INPUT_SIZE = (256, 192)
#INPUT_SIZE = (1392, 512)

# Directory to keep snapshots of current training
DIR_TO_SAVE = '/root/workspace/output/'
