import sys
sys.path.append('/home/youzhiji/workload/paddle_project/utils')
import paddle_aux
import paddle
import logging
from collections import OrderedDict
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.customize_loss import tanh_L1Loss, tanh_L2Loss
logger = logging.getLogger('base')


class GenerationModel(BaseModel):

    def __init__(self, opt):
        super(GenerationModel, self).__init__(opt)
        if opt['dist']:
            self.rank = paddle.distributed.get_rank()
        else:
            self.rank = -1
        train_opt = opt['train']
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = paddle.DataParallel(layers=self.netG)
        else:
            self.netG = paddle.DataParallel(self.netG)
        self.print_network()
        self.load()
        if self.is_train:
            self.netG.train()
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = paddle.nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = paddle.nn.MSELoss().to(self.device)
            elif loss_type == 'tanh_l1':
                self.cri_pix = tanh_L1Loss().to(self.device)
            elif loss_type == 'tanh_l2':
                self.cri_pix = tanh_L2Loss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'
                    .format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'
                ] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if not v.stop_gradient:
                    optim_params.append(v)
                elif self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k)
                        )
            self.optimizer_G = paddle.optimizer.Adam(parameters=
                optim_params, learning_rate=train_opt['lr_G'], weight_decay
                =wd_G, beta1=(train_opt['beta1'], train_opt['beta2'])[0],
                beta2=(train_opt['beta1'], train_opt['beta2'])[1])
            self.optimizers.append(self.optimizer_G)
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR_Restart
                        (optimizer, train_opt['lr_steps'], restarts=
                        train_opt['restarts'], weights=train_opt[
                        'restart_weights'], gamma=train_opt['lr_gamma'],
                        clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.
                        CosineAnnealingLR_Restart(optimizer, train_opt[
                        'T_period'], eta_min=train_opt['eta_min'], restarts
                        =train_opt['restarts'], weights=train_opt[
                        'restart_weights']))
            else:
                raise NotImplementedError(
                    'MultiStepLR learning rate scheme is enough.')
            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ']#.to(self.device)
        self.var_cond = data['cond']#.to(self.device)
        if need_GT:
            self.real_H = data['GT']#.to(self.device)

    def optimize_parameters(self, step):
        """Class Method: *.zero_grad, can not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*/torch.distributions.Distribution.*/torch.autograd.function.FunctionCtx.*/torch.profiler.profile.*/torch.autograd.profiler.profile.*, and convert manually"""
        self.optimizer_G.clear_grad()
        self.fake_H = self.netG((self.var_L, self.var_cond))
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with paddle.no_grad():
            self.fake_H = self.netG((self.var_L, self.var_cond))
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].astype(dtype='float32').cpu()
        out_dict['SR'] = self.fake_H.detach()[0].astype(dtype='float32').cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].astype(dtype='float32'
                ).cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, paddle.DataParallel) or isinstance(self.
            netG, paddle.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                self.netG.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.
                format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'][
                'strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
