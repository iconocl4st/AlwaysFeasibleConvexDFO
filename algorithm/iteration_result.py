

class IterationResult:
	def __init__(self):
		self.description = None
		self.update_radius = None
		self.radius_update = None
		self.infeasible_sample_region = None
		self.update_iterate = None
		self.new_iterate = None
		self.completed = None
		self.converged = None
		self.successful = None

	def repair_sample_region(self):
		self.infeasible_sample_region = True
		return self

	def accept(self, new_iterate):
		self.update_iterate = True
		self.new_iterate = new_iterate
		return self

	def multiply_radius(self, factor):
		self.update_radius = True
		self.radius_update = factor
		return self

	def set_successful(self):
		self.successful = True
		return self

	def set_completed(self):
		self.completed = True
		return self

	def set_converged(self):
		self.converged = True
		self.completed = True
		return self

	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return self.description

	def to_json(self):
		return {
			'description': self.description,
			'should-update-radius': self.update_radius,
			'update-radius-factor': self.radius_update,
			'sample-region-is-infeasible': self.infeasible_sample_region,
			'should-update-iterate': self.update_iterate,
			'new-iterate': self.new_iterate,
			'successful': self.successful,
			'terminated': self.completed,
			'converged': self.converged,
		}

	@staticmethod
	def parse_json(json_object):
		if json_object is None:
			return None
		result = IterationResult()
		result.description = json_object['description']
		result.update_radius = json_object['should-update-radius']
		result.radius_update = json_object['update-radius-factor']
		result.infeasible_sample_region = json_object['sample-region-is-infeasible']
		result.update_iterate = json_object['should-update-iterate']
		result.new_iterate = json_object['new-iterate']
		result.completed = json_object['terminated']
		result.converged = json_object['converged']
		result.successful = json_object['successful']
		return result

	@staticmethod
	def create(description):
		result = IterationResult()
		result.description = description
		result.update_radius = False
		result.radius_update = 1.0
		result.infeasible_sample_region = False
		result.update_iterate = False
		result.new_iterate = None
		result.completed = False
		result.converged = False
		result.successful = False
		return result

