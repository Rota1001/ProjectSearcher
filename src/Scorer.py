import AskQuestion

def Score(comment, targets):
	k = len(targets)
	score = 0
	for target in targets :
		score +=  AskQuestion.AskGpt(comment, target)#ask chatgpt question
	return score / k
