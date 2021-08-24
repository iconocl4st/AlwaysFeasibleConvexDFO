
class Formatting:
	NUM_DECIMAL_PLACES = 2

	@staticmethod
	def format_strings(strings):
		sep = ' & '
		newl = '\\\\\n'
		string_lengths = {}
		for row in strings:
			for idx, obj in enumerate(row):
				l = len(str(obj))
				if idx not in string_lengths or string_lengths[idx] < l:
					string_lengths[idx] = l
		return newl.join([
			sep.join([
				str(obj).rjust(string_lengths[idx], ' ')
				for idx, obj in enumerate(row)
			])
			for row in strings
		]).replace('_', ' ').replace('#', 'N')

	@staticmethod
	def format_vector(vec):
		strs = [Formatting.format_float(xi) for xi in vec]
		max_length = max(len(s) for s in strs)
		return '[' + ','.join(s.rjust(max_length) for s in strs) + ']'

	@staticmethod
	def format_float(value):
		if value is None:
			return 'none'
		if abs(value) < 1e-12:
			return '0.0'
		if abs(value) < 0.001:
			return ('{:.' + str(Formatting.NUM_DECIMAL_PLACES) + 'e}').format(value)
		return str(round(value, Formatting.NUM_DECIMAL_PLACES))
