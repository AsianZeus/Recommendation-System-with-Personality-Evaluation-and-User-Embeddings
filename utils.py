from config import SCORE_THRESHOLD, STOP_WORDS
from pydantic import BaseModel
from typing import Optional, List
from config import CANDIDATE_LABELS, MIN_QUESTIONS_THRESHOLD, RECOMMENDATION_MODEL_PATH, IDS_PATH
import numpy as np
import pickle
from tensorflow import keras


class Response(BaseModel):
    id: str
    automated: Optional[bool] = True


class Match(BaseModel):
    swiper: str
    swipees: List[str]


def generateQuestion(df):
    dx = df.sample(1)
    q = dx['Question'].item()
    a = dx['Answer'].item()
    t = dx['Type'].item()
    return {'question': q, 'answer': a, 'type': t}


def preprocess(i):
    exception = ['does talking', 'does thinking']
    sp = i.split(" ")
    if("prefer" in sp or "rather" in sp):
        temp = ' '.join([x for x in sp[3:] if x not in STOP_WORDS])[:-1]
    elif(' '.join(sp[:2]) in exception):
        temp = ' '.join([x for x in sp[1:] if x not in STOP_WORDS])[:-1]
    else:
        temp = ' '.join([x for x in sp[2:] if x not in STOP_WORDS])[:-1]
    return temp


def getType(q, a):
    tx = ['Yes', 'No', 'Maybe']
    if(a[0] in tx):
        return 'Option'
    if(q.lower().split()[0] == 'how'):
        return 'Range'
    return 'ThisThat'


def getScoreDistribution(x, zeroshot_classifier):
    sc = zeroshot_classifier(x, CANDIDATE_LABELS, multi_label=True)
    return {sc['labels'][i]: sc['scores'][i] for i in range(len(sc['labels']))}


def get_subPersonality(stats):
    num_personality = 4
    initials = {'Creative': 'C', 'Intellectual': 'I', 'Social': 'S',
                'Emotional': 'E', 'Athletic': 'A', 'Spiritual': 'M'}
    top_personality = sorted(stats, key=stats.get, reverse=True)[
        :num_personality]
    new_personality = ''.join([initials[x] for x in top_personality])
    return ''.join(sorted(new_personality))


def answer2numeric(response, typex):
    if(typex == 'Option'):
        if response == "Yes":
            response = 1
        elif response == "No":
            response = -1
        else:
            response = 0.5
        return response
    else:
        response = (((int(response) - 0) * 2) / 1) + -1
        return response


def calculate_interest_question_weight(tq):
    q_w = 0.9
    i_w = 0.1
    if(tq < MIN_QUESTIONS_THRESHOLD):
        q_w = tq/MIN_QUESTIONS_THRESHOLD
        i_w = 1-q_w
    return (q_w, i_w)


def flip_response(response, typex):
    return response*-1


def get_sentiment(text, sentiment_analysis):
    tokenizer, model = sentiment_analysis
    label = [0, 1]
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    return label[scores.argmax()]


def get_scores(question, answer, typex, sentiment_analysis, zeroshot_classifier):
    total_question_per_personality = {'Social': 0, 'Creative': 0, 'Emotional': 0,
                                      'Athletic': 0, 'Spiritual': 0, 'Intellectual': 0}
    if(typex == 'Range'):
        question = question.replace('On a Scale of 1-10, ', "")
    if(typex == 'ThisThat'):
        # fetching the score distribution of the question
        new_score = getScoreDistribution(answer, zeroshot_classifier)
        if(not get_sentiment(answer, sentiment_analysis)):
            answer = -1
        else:
            answer = 1
    else:
        answer = answer2numeric(answer, typex)
        if(not get_sentiment(question, sentiment_analysis)):
            answer = flip_response(answer, typex)
        # fetching the score distribution of the question
        new_score = getScoreDistribution(
            preprocess(question), zeroshot_classifier)

    for label in new_score:
        # checking if the question is intended for a personality type or not by checking if the score has crossed the threshold
        if abs(new_score[label]*100) <= SCORE_THRESHOLD:
            new_score[label] = 0
        else:
            new_score[label] = answer*100
            total_question_per_personality[label] = 1
    return new_score, total_question_per_personality


def rerank_personality(interests_score, previous_questions, sentiment_analysis, zeroshot_classifier):
    total_questions_per_personality = {'Social': 0, 'Creative': 0, 'Emotional': 0,
                                       'Athletic': 0, 'Spiritual': 0, 'Intellectual': 0}
    new_score = {'Social': 0, 'Creative': 0, 'Emotional': 0,
                 'Athletic': 0, 'Spiritual': 0, 'Intellectual': 0}
    scores = []

    for qna in previous_questions:
        question = qna['question']
        answer = qna['answer']
        typex = qna['type']
        score, total_question_per_personality = get_scores(question, answer, typex,
                                                           sentiment_analysis, zeroshot_classifier)

        # calculating the number of questions per personality
        for label in score:
            total_questions_per_personality[label] += total_question_per_personality[label]
        # print(f"question {question} | answer {answer} | score {score}")
        scores.append(score)

    for label in scores[0]:
        for score in scores:
            new_score[label] += score[label]
            if(new_score[label] < 0):
                new_score[label] = 0
        # average the score based on the number of questions per personality
        try:
            new_score[label] /= total_questions_per_personality[label]
        except ZeroDivisionError:
            new_score[label] = 0

        # calculating the weight of the question and interest
        q_w, i_w = calculate_interest_question_weight(
            total_questions_per_personality[label])

        # calculating the final score based on the weight of the question and interest
        new_score[label] = (interests_score[label]*i_w) + \
            (new_score[label]*q_w)

        # sanity check
        if(new_score[label] < 0):
            new_score[label] = 0
        elif(new_score[label] > 100):
            new_score[label] = 100
    # print(total_questions_per_personality)
    return new_score


def generate_automated_questions(Question_Generator, previous_questions):
    if not previous_questions:
        input_q = "Q:"
        idx = 1
    else:
        input_q = f"Q: {previous_questions['question']} A: {previous_questions['answer']} Q:"
        idx = 2
    output = Question_Generator(input_q, max_length=45, num_return_sequences=1)
    temp = []
    for i in output:
        try:
            sent = i['generated_text'].split('Q:')[idx]
            q, a = sent.split('A: ')
        except:
            sent = Question_Generator("Q:", max_length=45, num_return_sequences=1)[
                'generated_text'].split('Q:')[1]
            q, a = sent.split('A: ')
        q = q.strip()
        a = a.strip().split('/')
        t = getType(q, a)
        if(t == 'Range'):
            q = 'On a Scale of 1-10, '+q
        temp.append({'question': q,
                     'answer': a,
                     'type': t})
    return temp[0]


def calculate_initial_personality(interests, score_distribution):
    score = {'Social': 0, 'Emotional': 0, 'Creative': 0,
             'Athletic': 0, 'Intellectual': 0, 'Spiritual': 0}
    total_interest = len(interests)
    for interest in interests:
        for label in score:
            score[label] += ((score_distribution[interest]
                             [label] / total_interest) * 100)
    return score


def compute_compatibility(sc1, sc2):
    score = 0
    for x in sc1.keys():
        score += abs(sc1[x] - sc2[x])
    return 100-(score/6)


def predict_matching_probability(swiper, swipee):
    with open(IDS_PATH, 'rb') as f:
        ids = pickle.load(f)
    model = keras.models.load_model(RECOMMENDATION_MODEL_PATH)
    swiper = [ids[i] for i in swiper]
    swipee = [ids[i] for i in swipee]
    pred = model.predict([np.array(swiper), np.array(swipee)])
    return pred
