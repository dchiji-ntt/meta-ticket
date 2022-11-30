
class CombinedOptimizer(object):
    def __init__(self, opts_dict):
        self.opts_dict = opts_dict
        self.opts = [opts_dict[k] for k in opts_dict]

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        state_dict = dict()
        for k in self.opts_dict:
            state_dict[k] = self.opts_dict[k].state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for k in state_dict:
            self.opts_dict[k].load_state_dict(state_dict[k])

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()

