# %%
import math
import typing as ty
from typing import Optional
from pathlib import Path
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import zero
from torch import Tensor

from einops import rearrange, repeat

from stripedhyena.test import Hyena
from stripedhyena.utils import dotdict, print_rank_0

# from mamba_ssm import Mamba
import lib
import pdb
import time as timeqy
from dropout import dropout_features0
    

# %%
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # nn_init.kaiming_normal_(self.category_embeddings.weight, nonlinearity='relu')
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        # self.weight = nn.Parameter(Tensor(d_numerical , d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn_init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
            # nn_init.kaiming_normal_(self.bias, nonlinearity='relu')

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class FT_Hyena(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        activation: str,
        prenormalization: bool,
        initialization: str,
        #
        d_out: int,
    ) -> None:

        super().__init__()
        # self.drop_path_prob = 0.0
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens
        self.LayerNorm = nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        config_hyena = dotdict(yaml.load(open("./bin/test.yml"), Loader=yaml.FullLoader))
        config_hyena['hidden_size'] = d_token
        config_hyena['num_filters'] = d_token
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'hyena': Hyena(config_hyena)
                }
            )
            # if not prenormalization or layer_idx:
            #     layer['norm0'] = make_normalization()
            self.layers.append(layer)

        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        # self.prenormalization = prenormalization
        # self.last_normalization = make_normalization() if prenormalization else None
        self.head = nn.Linear(d_token, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            x ,_ = layer['hyena'](x)
        # x = x[:, 0 ,:]
        x = torch.mean(x, axis=1)
        x = self.LayerNorm(x)
        # if self.last_normalization is not None:
        #     x = self.last_normalization(x)
        # x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


# %%
if __name__ == "__main__":
    args, output = lib.load_config()
    args['model'].setdefault('token_bias', True)
    # %%
    zero.set_randomness(args['seed'])
    dataset_dir = lib.get_path(args['data']['path'])
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **lib.load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

    D = lib.Dataset.from_dir(dataset_dir)
    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)
    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    lib.dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat = X
    del X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']
    chunk_size = None

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    model = FT_Hyena(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories(X_cat),
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **args['model'],
    ).to(device)
    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = lib.get_n_parameters(model)
    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    # for x in ['tokenizer', '.norm', '.bias']:
    #     assert any(x in a for a in (b[0] for b in model.named_parameters()))
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        (
            [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
            ]
        ),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pt'

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'chunk_size': chunk_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )

    def apply_model(part, idx):
        return model(
            None if X_num is None else X_num[part][idx],
            None if X_cat is None else X_cat[part][idx],
        )

    def dropout_train(Dataset, p_train, p_val, p_test, seed):
        D_drop = dropout_features0(Dataset, p_train, p_val, p_test, seed)
        X = D_drop.build_X(
            normalization=args['data'].get('normalization'),
            num_nan_policy='mean',
            cat_nan_policy='new',
            cat_policy=args['data'].get('cat_policy', 'indices'),
            cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
            seed=args['seed'],
        )
        if not isinstance(X, tuple):
            X = (X, None)
        zero.set_randomness(args['seed'])
        X = tuple(None if x is None else lib.to_tensors(x) for x in X)
        device = lib.get_device()
        if device.type != 'cpu':
            X = tuple(
                None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
            )
            Y_device = {k: v.to(device) for k, v in Y.items()}
        else:
            Y_device = Y
        X_num, X_cat = X
        del X
        return X_num, X_cat

    @torch.no_grad()
    def evaluate(parts):
        global eval_batch_size
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx)
                                for idx in lib.IndexLoader(
                                    min(D.size(part),1000), eval_batch_size, False, device
                                )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    print('New eval batch size:', eval_batch_size)
                    stats['eval_batch_size'] = eval_batch_size
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part][0:1000].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                'logits',
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
        return metrics, predictions

    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        lib.dump_stats(stats, output, final)
        lib.backup_output(output)

    timer.run()
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()
        # X_num, X_cat = dropout_train(D, 0.1, 0.1, 0.3, 7)
        model.train()
        epoch_losses = []
        count = 0
        for batch_idx in epoch:
            loss, new_chunk_size = lib.train_with_auto_virtual_batch(
                optimizer,
                loss_fn,
                lambda x: (apply_model(lib.TRAIN, x), Y_device[lib.TRAIN][x]),
                batch_idx,
                chunk_size or batch_size,
            )
            epoch_losses.append(loss.detach())
            count += 1

            if count >= 10:  # 如果计数器达到10，退出循环
                break
            if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
                stats['chunk_size'] = chunk_size = new_chunk_size
                print('New chunk size:', chunk_size)
        # for name, param in  model.named_parameters():
        #     print(f"  {name}:")
        #     print("    Weight:", param.data)
        #     if param.grad is not None:
        #         print("    Gradient:", param.grad.data)
        #     else:
        #         print("    Gradient: None")
        # pdb.set_trace()
        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')
        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[lib.VAL]['score'])

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail:
            break
    # %%
    print('\nRunning the final evaluation...')
    # model.load_state_dict(torch.load("./output/jannis/ft_hyena/d1/checkpoint.pt")['model'])
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    start_time = timeqy.time()
    # X_num, X_cat = dropout_train(D, 0.1, 0.1, 0.7, 7)
    stats['metrics'], predictions = evaluate([lib.TEST])
    end_time = timeqy.time()
    # 计算推理时间
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.6f} seconds")
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
    stats['time'] = lib.format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
