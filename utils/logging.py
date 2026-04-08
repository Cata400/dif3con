import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from .utils import postprocess
from PIL import Image
import logging
import sys
import warnings
from discord_webhook import DiscordWebhook, DiscordEmbed
from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET",
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)]
)

# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
warnings.simplefilter(action='ignore', category=FutureWarning)


class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, cfg):
        log_dir = os.path.join(cfg['paths']["experiments_root"], cfg['paths']["experiment_name"], cfg['paths']['tb_logger'])
        self.result_dir = os.path.join(cfg['paths']["experiments_root"], cfg['paths']["experiment_name"], cfg['paths']['results'])
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.writer = None
        self.selected_module = ""

        log_dir = str(log_dir)

        # Retrieve vizualization writer.
        succeeded = False
        self.writer = SummaryWriter(log_dir)

        self.epoch = 0
        self.iter = 0
        self.phase = ''
        self.epsilon_phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

        self.label_to_name = {
            -1: "forget", 1: "retain"
        }

    def set_iter(self, epoch, iter, phase='train', epsilon_phase=''):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter
        self.epsilon_phase = epsilon_phase

    def save_images(self, results, save_dir=None):
        """
        Save the current results at result_path.
        If label field is not None, save in the corresponding retain / forget folder.
        """

        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        
        if self.epsilon_phase:
            result_path = os.path.join(result_path, f'epsilon_phase_{self.epsilon_phase}_epoch_{self.epoch}')
            os.makedirs(result_path, exist_ok=True)
        else:
            if save_dir is not None:
                result_path = os.path.join(result_path, save_dir)
            else:
                result_path = os.path.join(result_path, str(self.epoch))
            os.makedirs(result_path, exist_ok=True)

        ''' get names and corresponding images from results[OrderedDict] '''
        try:
            names = results['name']
            outputs = postprocess(results['result'])
            labels = results.get('label', [None for _ in range(len(names))])

            for i in range(len(names)): 
                if labels[i]:
                    class_path = self.label_to_name[labels[i].item()]

                    if not os.path.exists(os.path.join(result_path, class_path)):
                        os.makedirs(os.path.join(result_path, class_path))
                else:
                    class_path = ''

                Image.fromarray(outputs[i]).save(
                    os.path.join(result_path, class_path, names[i])
                )
        except:
            raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')

    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
        
        
class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        # self.rank = cfg['global_rank']
        self.rank = 0
        self.phase = phase

        self.setup_logger(None, os.path.join(cfg['paths']["experiments_root"], cfg['paths']["experiment_name"]), self.phase, level=logging.INFO, screen=False)
        self.logger = logging.getLogger(self.phase)
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0: # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper
    
    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)


class LogTracker:
    """
    record training numerical indicators.
    """
    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
    
    
class DiscordLogger:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_message(self, content, type=None):
        webhook = DiscordWebhook(url=self.webhook_url)
        if type is None:
            webhook.content = content
        elif type == 'error':
            embed = DiscordEmbed(title="Error", description=content, color="ff0000")
            embed.set_timestamp()
            webhook.add_embed(embed)
        elif type == 'success':
            embed = DiscordEmbed(title="Success", description=content, color="00ff00")
            embed.set_timestamp()
            webhook.add_embed(embed)
        webhook.execute()
