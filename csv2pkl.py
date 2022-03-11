import pandas as pd
import ast
questions = pd.read_csv('Documents/GenericQuestions.csv', index_col=0)
cols = questions.columns.tolist()

for i in range(len(questions)):
    for x in cols:
        try:
            questions[x][i] = ast.literal_eval(questions[x][i])
        except:
            pass
questions.to_pickle('GenericQuestions.pkl')
