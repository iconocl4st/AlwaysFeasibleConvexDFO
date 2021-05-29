import json
import numpy as np

# Could use:
# https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module


class ComplexEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(ComplexEncoder, self).__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs['indent']
        self._replacement_map = {}

    def default(self, obj):
        # if isinstance(obj, list) or isinstance(obj, tuple) and len(obj) < 3:
        if isinstance(obj, np.bool_) or isinstance(obj, np.bool):
            return bool(obj)
        if isinstance(obj, np.float) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            # num_dimension = len(obj.shape)
            # if num_dimension == 1:
            #     return '[' + ', '.join([str(xi) for xi in obj]) + ']'
            # if num_dimension == 2:
            #     return '[' + '; '.join([', '.join([str(xij) for xij in xi]) for xi in obj]) + ']'
            return [xi for xi in obj]

        to_json_method = getattr(obj, "to_json", None)
        if to_json_method is not None and callable(to_json_method):
            return to_json_method()

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class JsonUtils:
    @staticmethod
    def dumps(obj):
        return json.dumps(obj, cls=ComplexEncoder, indent=2)

    @staticmethod
    def dump(obj, out):
        return json.dump(obj, out, cls=ComplexEncoder, indent=2)

    @staticmethod
    def to_json(matrix):
        pass
