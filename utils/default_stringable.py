
from abc import *
from utils.json_utils import JsonUtils


class DefaultStringable(ABC):
	def __repr__(self):
		return self.__str__()

	def __str__(self):
		return JsonUtils.dumps(self.to_json())

	@abstractmethod
	def to_json(self):
		pass
