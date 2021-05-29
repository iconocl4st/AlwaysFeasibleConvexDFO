
# TODO: Use this in more spots
class TrialValue:
	def __init__(self):
		self.x = None
		self.value = None

	def accept(self, alt_x, alt_value, feasible=True):
		if not feasible:
			return
		if self.x is not None and self.value <= alt_value:
			return
		self.x = alt_x
		self.value = alt_value

