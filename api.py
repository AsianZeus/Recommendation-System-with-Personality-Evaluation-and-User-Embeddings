from fastapi import FastAPI, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import *
from config import *
from resources import Generic_Questions, Question_Generator, Zeroshot_Classifier, Sentiment_Analysis, User_Database

app = FastAPI(openapi_tags=TAGS_METADATA,
              title=TITLE,
              description=DESCRIPTION,
              version=VERSION,
              contact=CONTACT_INFO,
              license_info={
                  "name": "Apache 2.0",
                  "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
              },
              )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return 'Recommendation System with Personality Evaluation and User Embeddings'


@ app.get('/getInterests', tags=["Get Interests"])
def getInterests():
    return {"interests": ["Cooking", "Travelling", "Sports", "Video Games", "Singing", "Dancing", "Artwork", "Reading", "Writing", "Binge-Watch", "Astrology", "Stand-up Comedy", "Photography", "Music", "Party", "Workout", "Technology", "Shopping", "Politics", "Content Creation"]}


@ app.post('/getInitialPersonality', tags=["Get Initial Personality"])
def getInitialPersonality(res: Response):
    id = res.id
    if User_Database.verify_user(id):
        interests = User_Database.get_interests(id)
        try:
            score = calculate_initial_personality(
                interests, INTEREST_SCORE_DISTRIBUTION)
            sub_personality_type = get_subPersonality(score)
            return {"score": score, "personality_type": sub_personality_type}
        except KeyError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid interests!")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User ID does not exist!")


@ app.post("/generateQuestions", tags=["Generate Question"])
def generateQuestions(res: Response):
    id = res.id
    if User_Database.verify_user(id):
        try:
            previous_questions = User_Database.get_qna(id)
        except:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Key 'qna' does not exist!")
        try:
            interests = User_Database.get_interests(id)
        except:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Key 'interests' does not exist!")
        try:
            interests_score = calculate_initial_personality(
                interests, INTEREST_SCORE_DISTRIBUTION)
        except KeyError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid Interests!")
        is_empty = not previous_questions
        if not res.automated:
            gen_q = generateQuestion(Generic_Questions)
        else:
            last_q = None if is_empty else previous_questions[-1]
            gen_q = generate_automated_questions(Question_Generator, last_q)
        if is_empty:
            return {"generated": gen_q}
        else:
            generated_score = rerank_personality(
                interests_score, previous_questions, Sentiment_Analysis, Zeroshot_Classifier)
            sub_personality_type = get_subPersonality(generated_score)
            return {"score": generated_score, "personality_type": sub_personality_type, "generated": gen_q}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User ID does not exist!")


@ app.post("/getMatchingProbability", tags=["Matching Probability"])
async def getMatchingProbability(match: Match):
    swipees = match.swipees
    swiper = match.swiper
    for_recommendation_engine = []
    scores_for_potential_couples = {}
    filtered_swipers = []
    filtered_swipees = []

    if not User_Database.verify_user(swiper):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Swiper ID {swiper} does not exist!")
    for swipee in swipees:
        if not User_Database.verify_user(swipee):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Swipee ID {swipee} does not exist!")
        if(User_Database.count_liked_users(swiper) > 10 and User_Database.count_disliked_users(swiper) > 5 and User_Database.count_liked_users(swipee) > 10 and User_Database.count_disliked_users(swipee) > 5):
            filtered_swipers.append(swiper)
            filtered_swipees.append(swipee)
            for_recommendation_engine.append((swiper, swipee))
        else:
            try:
                score1 = User_Database.get_personality_scores(swiper)
                score2 = User_Database.get_personality_scores(swipee)
            except:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Key 'personality_scores' does not exist!")
            scores_for_potential_couples[swipee] = compute_compatibility(
                score1, score2)
    if(filtered_swipees):
        pred = predict_matching_probability(
            filtered_swipers, filtered_swipees)
        for swiper, swipee, pred in zip(filtered_swipers, filtered_swipees, pred):
            scores_for_potential_couples[swipee] = pred.item()*100
    return {"scores": scores_for_potential_couples}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)