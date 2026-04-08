from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

from metrics.inception import FeatureExtractorInceptionV3, FeatureExtractorBase
import torchvision
from datasets.datasets import ImagesPathDataset, TransformPILtoRGBTensor
from torch.utils.data import Dataset, DataLoader
from rich.progress import track
from rich.console import Console
import multiprocessing


"""
Adapted from:
https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_isc.py
"""

import numpy as np
import torch


KEY_METRIC_ISC_MEAN = "inception_score_mean"
KEY_METRIC_ISC_STD = "inception_score_std"
DEFAULT_FEATURE_EXTRACTOR = "inception-v3-compat"


def isc_features_to_metric(feature, splits=10, shuffle=True, rng_seed=2020):
    assert torch.is_tensor(feature) and feature.dim() == 2
    N, C = feature.shape
    if shuffle:
        rng = np.random.RandomState(rng_seed)
        feature = feature[rng.permutation(N), :]
    feature = feature.double()

    p = feature.softmax(dim=1)
    log_p = feature.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits) : ((i + 1) * N // splits), :]
        log_p_chunk = log_p[(i * N // splits) : ((i + 1) * N // splits), :]
        q_chunk = p_chunk.mean(dim=0, keepdim=True)
        kl = p_chunk * (log_p_chunk - q_chunk.log())
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)

    return {
        KEY_METRIC_ISC_MEAN: float(np.mean(scores)),
        KEY_METRIC_ISC_STD: float(np.std(scores)),
    }


def isc_featuresdict_to_metric(featuresdict, feat_layer_name, **kwargs):
    features = featuresdict[feat_layer_name]

    out = isc_features_to_metric(
        features,
        kwargs.get("splits", 10),
        kwargs.get("shuffle", True),
        kwargs.get("rng_seed", 2020)
    )

    verbose = kwargs.get("verbose", True)
    if verbose:
        print(f"Inception Score: {out[KEY_METRIC_ISC_MEAN]:.7g} ± {out[KEY_METRIC_ISC_STD]:.7g}")

    return out


"""
Adapted from:
https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/utils.py
"""
def glob_samples_paths(path, samples_find_deep, samples_find_ext, samples_ext_lossy=None, verbose=True):
    assert type(samples_find_ext) is str and samples_find_ext != "", "Sample extensions not specified"
    assert \
        samples_ext_lossy is None or type(samples_ext_lossy) is str, "Lossy sample extensions can be None or string"

    samples_find_ext = [a.strip() for a in samples_find_ext.split(",") if a.strip() != ""]
    if samples_ext_lossy is not None:
        samples_ext_lossy = [a.strip() for a in samples_ext_lossy.split(",") if a.strip() != ""]
    have_lossy = False
    files = []
    for r, d, ff in os.walk(path):
        if not samples_find_deep and os.path.realpath(r) != os.path.realpath(path):
            continue
        for f in ff:
            ext = os.path.splitext(f)[1].lower()
            if len(ext) > 0 and ext[0] == ".":
                ext = ext[1:]
            if ext not in samples_find_ext:
                continue
            if samples_ext_lossy is not None and ext in samples_ext_lossy:
                have_lossy = True
            files.append(os.path.realpath(os.path.join(r, f)))
    files = sorted(files)
    return files


def prepare_input_from_descriptor(input_desc, **kwargs):
    bad_input = False
    input = input_desc["input"]
    if type(input) is str:
        if os.path.isdir(input):
            samples_find_deep = True
            samples_find_ext = "png,jpg,jpeg"
            samples_ext_lossy = "jpg,jpeg"
            samples_resize_and_crop = 0
            verbose = True
            input = glob_samples_paths(input, samples_find_deep, samples_find_ext, samples_ext_lossy, verbose)
            assert len(input) > 0, f"No samples found in {input} with samples_find_deep={samples_find_deep}"
            transforms = []
            if samples_resize_and_crop > 0:
                transforms += [
                    torchvision.transforms.Resize(samples_resize_and_crop),
                    torchvision.transforms.CenterCrop(samples_resize_and_crop),
                ]
            transforms.append(TransformPILtoRGBTensor())
            transforms = torchvision.transforms.Compose(transforms)
            input = ImagesPathDataset(input, transforms)
        else:
            bad_input = True
    else:
        bad_input = True
    assert \
        not bad_input, \
        f'Input descriptor "input" field must be a string with a path to a directory with images'
        
    num_samples = kwargs.get("num_samples", -1)
    if num_samples > 0:
        input = torch.utils.data.Subset(input, range(num_samples))

    return input


def create_feature_extractor(name, list_features, cuda=True, **kwargs):
    if kwargs.get("verbose", True):
        print(f'Creating feature extractor "{name}" with features {list_features}')
    cls = FeatureExtractorInceptionV3
    feat_extractor = cls(name, list_features, **kwargs)
    feat_extractor.requires_grad_(False)
    feat_extractor.eval()
    if cuda:
        feat_extractor.cuda()
        
    return feat_extractor


def make_input_descriptor_from_int(input_int, path, **kwargs):
    assert input_int in (1, 2), "Supported input slots: 1, 2"
    input = path
    input_desc = {
        "input": input
    }

    return input_desc


def prepare_input_descriptor_from_input_id(input_id, **kwargs):
    assert \
        type(input_id) is int, \
        "Input can be either integer (1 or 2) specifying the first or the second set of kwargs, or a string as a shortcut for registered datasets"
    if type(input_id) is int:
        input_desc = make_input_descriptor_from_int(input_id, **kwargs)
    return input_desc


def resolve_feature_layer_for_metric(metric):
    out = FeatureExtractorInceptionV3.get_default_feature_layer_for_metric(metric)
    return out


def prepare_input_from_id(input_id, **kwargs):
    input_desc = prepare_input_descriptor_from_input_id(input_id, **kwargs)
    return prepare_input_from_descriptor(input_desc, **kwargs)


def get_featuresdict_from_dataset(input, feat_extractor, batch_size, cuda, save_cpu_ram, verbose):
    assert isinstance(input, Dataset), "Input can only be a Dataset instance"
    assert torch.is_tensor(input[0]), "Input Dataset should return torch.Tensor"
    assert isinstance(feat_extractor, FeatureExtractorBase), "Feature extractor is not a subclass of FeatureExtractorBase"

    if batch_size > len(input):
        batch_size = len(input)

    num_workers = 0 if save_cpu_ram else min(4, 2 * multiprocessing.cpu_count())

    dataloader = DataLoader(
        input,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=cuda,
    )

    out = None

    for bid, batch in track(enumerate(dataloader), disable=not verbose, total=len(input), description="Processing samples", console=Console(stderr=True)):
        if cuda:
            batch = batch.cuda(non_blocking=True)

        features = feat_extractor(batch)
        featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    return out

def get_featuresdict_for_batch(batch, feat_extractor, **kwargs):
    cuda = kwargs.get("cuda", True)
    save_cpu_ram = False
    verbose = kwargs.get("verbose", True)
    
    features = feat_extractor(batch)
    featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
    featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}
    
    return featuresdict


def extract_featuresdict_from_input_id(input_id, feat_extractor, **kwargs):
    batch_size = kwargs.get("batch_size", 50)
    cuda = kwargs.get("cuda", True)
    verbose = kwargs.get("verbose", True)
    input = prepare_input_from_id(input_id, **kwargs)
    save_cpu_ram = False
    
    featuresdict = get_featuresdict_from_dataset(input, feat_extractor, batch_size, cuda, save_cpu_ram, verbose)
        
    return featuresdict


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "path",
        type=str,
        help=("Paths to the generated images or " "to .npz statistic files"),
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument(
        "--num-workers",
        type=int,
        help=(
            "Number of processes to use for data loading. " "Defaults to `min(8, num_cpus)`"
        ),
    )
    parser.add_argument(
        "--device", type=str, default='cpu', help="Device to use. Like cuda, cuda:0 or cpu"
    )
    
    parser.add_argument(
        "--splits", type=int, default=10, help="Number of splits when computing the inception score. Default is 10"
    )
    
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbosity"
    )

    
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples to use, -1 means all samples"
    )
    
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["cuda"] = 'cuda' in args.device

    
    feature_extractor = DEFAULT_FEATURE_EXTRACTOR
    feature_layer_isc = resolve_feature_layer_for_metric("isc")
    feature_layers = [feature_layer_isc]
    feat_extractor = create_feature_extractor(feature_extractor, list(feature_layers), **kwargs)
    featuresdict_1 = extract_featuresdict_from_input_id(1, feat_extractor, **kwargs)
    metric_isc = isc_featuresdict_to_metric(featuresdict_1, feature_layer_isc, **kwargs)


if __name__ == '__main__':
    main()