
def make_assertion(expr, message):
	if not expr:
		print('about to fail assertion')
		raise Exception(message)
	assert expr, message