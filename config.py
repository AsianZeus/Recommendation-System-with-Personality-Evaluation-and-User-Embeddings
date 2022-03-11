DESCRIPTION = "Recommendation engine for a dating apps to match users based on their personality traits like Social, Emotional, Creative, Athletic, Intellectual, Spiritual."

CONTACT_INFO = {
    "name": "Akshat Surolia",
    "url": "https://www.akshatsurolia.com",
    "email": "mail@akshatsurolia.com",
}

TITLE = "Recommendation System with Personality Evaluation and User Embeddings"

VERSION = "0.0.1"

SCORE_THRESHOLD = 10.0
MIN_QUESTIONS_THRESHOLD = 10

USERNAME = "username"
PASSWORD = "password"
DB = "database"
COLLECTION = "users"

INTEREST_SCORE_DISTRIBUTION = {'Cooking': {'Creative': 0.949806, 'Social': 0.761699, 'Intellectual': 0.523341, 'Spiritual': 0.491432, 'Emotional': 0.000828, 'Athletic': 0.000132}, 'Travelling': {'Creative': 0.74666, 'Intellectual': 0.619141, 'Athletic': 0.614277, 'Spiritual': 0.586522, 'Social': 0.549081, 'Emotional': 0.348753}, 'Sports': {'Athletic': 0.9999, 'Creative': 0.398264, 'Emotional': 0.113864, 'Intellectual': 0.110371, 'Spiritual': 0.076452, 'Social': 0.032297}, 'Video Games': {'Creative': 0.816865, 'Athletic': 0.61707, 'Intellectual': 0.194692, 'Emotional': 0.047866, 'Social': 0.002403, 'Spiritual': 0.000251}, 'Singing': {'Creative': 0.88479, 'Social': 0.577905, 'Spiritual': 0.574165, 'Emotional': 0.129772, 'Intellectual': 0.000411, 'Athletic': 0.000129}, 'Dancing': {'Creative': 0.927237, 'Social': 0.833174, 'Athletic': 0.29001, 'Spiritual': 0.003073, 'Intellectual': 0.000352, 'Emotional': 0.000258}, 'Artwork': {'Creative': 0.961688, 'Social': 0.55184, 'Athletic': 0.345682, 'Spiritual': 0.218566, 'Emotional': 0.17355, 'Intellectual': 0.100602}, 'Reading': {'Intellectual': 0.724705, 'Creative': 0.681769, 'Social': 0.572699, 'Spiritual': 0.373994, 'Emotional': 0.098602, 'Athletic': 5e-05}, 'Writing': {'Creative': 0.957597, 'Intellectual': 0.750557, 'Social': 0.604312, 'Spiritual': 0.444471, 'Athletic': 0.327705, 'Emotional': 0.238211}, 'Binge-Watch': {'Social': 0.43275, 'Creative': 0.069575, 'Emotional': 0.000791, 'Athletic': 0.000242, 'Intellectual': 0.000137, 'Spiritual': 0.000122},
                               'Astrology': {'Intellectual': 0.665472, 'Creative': 0.56237, 'Spiritual': 0.252457, 'Social': 0.00015, 'Emotional': 5e-05, 'Athletic': 2.8e-05}, 'Stand-up Comedy': {'Creative': 0.53885, 'Social': 0.334904, 'Emotional': 0.011266, 'Athletic': 0.000203, 'Intellectual': 0.000105, 'Spiritual': 5e-05}, 'Photography': {'Creative': 0.985727, 'Social': 0.006371, 'Athletic': 0.00239, 'Emotional': 0.001682, 'Spiritual': 8.9e-05, 'Intellectual': 8.4e-05}, 'Music': {'Creative': 0.828033, 'Intellectual': 0.620501, 'Spiritual': 0.462553, 'Social': 0.435509, 'Emotional': 0.34586, 'Athletic': 0.06108}, 'Party': {'Social': 0.925684, 'Creative': 0.890612, 'Intellectual': 0.811586, 'Athletic': 0.519235, 'Spiritual': 0.491298, 'Emotional': 0.106208}, 'Workout': {'Athletic': 0.967679, 'Intellectual': 0.809979, 'Creative': 0.786341, 'Social': 0.767252, 'Spiritual': 0.593423, 'Emotional': 0.27002}, 'Technology': {'Intellectual': 0.985549, 'Athletic': 0.391094, 'Creative': 0.356991, 'Emotional': 0.057442, 'Social': 0.003928, 'Spiritual': 0.002048}, 'Shopping': {'Creative': 0.842639, 'Intellectual': 0.020884, 'Social': 0.007005, 'Emotional': 0.000319, 'Spiritual': 0.000262, 'Athletic': 5.4e-05}, 'Politics': {'Creative': 0.425403, 'Social': 0.311772, 'Emotional': 0.12724, 'Intellectual': 0.035602, 'Athletic': 0.03288, 'Spiritual': 0.000228}, 'Content Creation': {'Creative': 0.952628, 'Intellectual': 0.924935, 'Social': 0.61761, 'Spiritual': 0.451271, 'Emotional': 0.00041, 'Athletic': 0.000342}}
STOP_WORDS = {'down', 'other', 'more', "you'll", 'same', 'that', 'been', 'you', 'from', 'yours', 'by', 't', 'has', 'he', 'on', 'shan', 'each', "weren't", 'few', 'her', 'if', 'again', 'all', 'further', "wasn't", 'in', 'no', 'than', 'will', "doesn't", 'didn', 'own', 'nor', "hasn't", "you'd", "isn't", 'mightn', 'most', 'before', "shouldn't", 'and', 'about', 'having', 'me', "mustn't", 'doesn', 'does', 'him', 'had', "you've", 'was', 'an', 'its', 'those', 'too', 'don', 'against', 'or', 'through', 'over', 'such', 'doing', 'a', 'yourself', 'both', 'above', 've', 'so', 'this', 'but', "should've", "needn't", "won't", "haven't", 'into', 'm', "that'll", 'isn', 'hers', 'did', 'at', 'whom', 'wouldn', 'are', 'how', 'theirs', 'your', 'his', 'themselves',
              'we', 'be', 'aren', 'very', "mightn't", 'they', 'not', 'won', 'ours', 'my', 'when', 'their', 'here', 'who', 'once', 'only', 'where', 'until', 'yourselves', 'then', "wouldn't", 'some', "couldn't", 'after', 'myself', 'for', 'under', 'd', 'o', 'being', "it's", 'just', 'these', "aren't", "didn't", 'hadn', 'i', 'is', 'while', "hadn't", 'our', 'should', "don't", "you're", 'needn', 'mustn', 'she', 'up', 'below', 'now', "she's", 'it', 'why', 're', 'off', 'as', 'haven', 'have', 'them', 'any', 'am', "shan't", 'with', 'do', 'couldn', 'because', 'the', 'can', 'there', 'between', 'itself', 'hasn', 's', 'ain', 'during', 'to', 'shouldn', 'll', 'weren', 'himself', 'out', 'ourselves', 'what', 'wasn', 'of', 'herself', 'y', 'ma', 'were', 'which'}
CANDIDATE_LABELS = ['Social', 'Emotional', 'Creative',
                    'Athletic', 'Intellectual', 'Spiritual']

TAGS_METADATA = [
    {
        "name": "Get Interests",
        "description": "Returns a list of all the interests that the user can select user"
    },
    {
        "name": "Get Initial Personality",
        "description": "Takes user's ID and returns the personality score and personality type of the user. Read the schema for more details"
    },
    {
        "name": "Generate Question",
        "description": "Takes user's ID and returns an automated/predefined question and answer and type pair, and personality score and type of the user. Read the schema for more details"
    },
    {
        "name": "Matching Probability",
        "description": "Takes swiper ID and a list of swipee IDs and returns a probability of chances of getting matched, based on thier swiping history if the users has any, else on personality scores. Read the schema for more details"
    },

]

RECOMMENDATION_MODEL_PATH = "RecommendationEngine/data/prod/models/keras/classifier_not_trainable.h5"
IDS_PATH = "RecommendationEngine/data/prod/hex2int-IDs.pkl"
