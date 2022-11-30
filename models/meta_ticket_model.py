
import traceback
from torch.autograd import grad

from models.sparse_module import SparseModule
from utils.l2l_utils import clone_module, update_module
from models.maml_model import MAMLModel

class MetaTicketModel(MAMLModel):
    def __init__(self, model, lrs, **kwargs):
        super(MAMLModel, self).__init__()   # avoid to call super(MetaTicketModel, self).__init__()

        self.lrs = lrs
        self.kwargs = kwargs
        self.already_sparse = self.kwargs.get('already_sparse', False)
        self.init_mode = self.kwargs.get('init_mode', None)
        self.ignore_params = self.kwargs.get('ignore_params', [])
        self.first_order = self.kwargs.get('first_order', False)
        self.allow_nograd = self.kwargs.get('allow_nograd', False)
        self.allow_unused = self.kwargs.get('allow_unused', self.allow_nograd)
        self.init_sparsity = self.kwargs.get('init_sparsity', None)
        self.learnable_scale = self.kwargs['learnable_scale']
        self.scale_delta_coeff = self.kwargs['scale_delta_coeff']
        self.rerand_freq = self.kwargs['rerand_freq']
        self.rerand_rate = self.kwargs['rerand_rate']

        if self.already_sparse:
            self.module = model
        else:
            self.module = SparseModule(model,
                                       self.init_sparsity,
                                       init_mode=self.init_mode,
                                       meta_ticket_mode=True,
                                       ignore_params_keywords=self.ignore_params,
                                       rerand_freq=self.rerand_freq,
                                       rerand_rate=self.rerand_rate,
                                       scale_delta_coeff=self.scale_delta_coeff,
                                       learnable_scale=self.learnable_scale)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self,
              loss,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):

        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.inner_parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            param_names = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for name, param in self.module.named_inner_parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
                param_names.append(name)
        else:
            try:
                gradients = grad(loss,
                                 self.module.inner_parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=allow_unused)
                param_names = [name for name, _ in self.module.named_inner_parameters()]
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # Update the module
        self.module = self._update(self.module, self.lrs, gradients)
        return zip(param_names, gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        kwargs = self.kwargs.copy()
        if first_order is not None:
            kwargs['first_order'] = first_order
        if allow_unused is not None:
            kwargs['allow_unused'] = allow_unused
        if allow_nograd is not None:
            kwargs['allow_nograd'] = allow_nograd
        return MetaTicketModel(clone_module(self.module), self.lrs,
                               already_sparse=True, **kwargs)

    def to(self, device):
        self.module = self.module.to(device)
        return super().to(device)

    def _update(self, model, lrs, grads=None):
        if grads is not None:
            named_params = list(model.named_inner_parameters())
            if not len(grads) == len(list(named_params)):
                msg = 'WARNING:MetaTicketModel._update(): Parameters and gradients have different length. ('
                msg += str(len(named_params)) + ' vs ' + str(len(grads)) + ')'
                print(msg)
            for (name, p), g in zip(named_params, grads):
                lr = lrs[name]
                if g is not None:
                    p.update = -1 * g
                    p.update.mul_(lr)
        return update_module(model)

    def named_meta_parameters(self):
        return self.module.named_meta_parameters()

    def named_inner_parameters(self):
        return self.module.named_inner_parameters()
