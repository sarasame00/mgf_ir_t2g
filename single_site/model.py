from .simulation import run_single_site
from .result import SingleSiteResult

class SingleSiteModel:
    def __init__(self):
        self._params = {}

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def run_simulation(self):
        # Run the OOP simulation using current parameters
        data = run_single_site(**self._params)
        return SingleSiteResult(data)
