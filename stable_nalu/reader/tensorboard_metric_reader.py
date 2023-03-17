
import re
import ast
import pandas
import collections
import multiprocessing

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from .tensorboard_reader import TensorboardReader

def _parse_numpy_str(array_string):
    pattern = r'''# Match (mandatory) whitespace between...
                (?<=\]) # ] and
                \s+
                (?= \[) # [, or
                |
                (?<=[^\[\]\s])
                \s+
                (?= [^\[\]\s]) # two non-bracket non-whitespace characters
            '''

    # Replace such whitespace with a comma
    fixed_string = re.sub(pattern, ',', array_string, flags=re.VERBOSE)
    return np.array(ast.literal_eval(fixed_string))

def _csv_format_column_name(column_name):
    return column_name.replace('/', '.')

def _everything_default_matcher(tag):
    return True

class TensorboardMetricReader:
    def __init__(self, dirname,
                 metric_matcher=_everything_default_matcher,
                 step_start=0,
                 recursive_weight=False,
                 processes=None,
                 progress_bar=True,
                 weights_only=False):
        self.dirname = dirname
        self.metric_matcher = metric_matcher
        self.step_start = step_start
        self.recursive_weight = recursive_weight
        self.weights_only = weights_only

        self.processes = processes
        self.progress_bar = progress_bar

    def _parse_tensorboard_data(self, inputs):
        (dirname, filename, reader) = inputs

        columns = collections.defaultdict(list)
        columns['name'] = dirname

        current_epoch = None
        current_logged_step = None

        for e in tf.compat.v1.train.summary_iterator(filename):
            step = e.step - self.step_start

            for v in e.summary.value:
                if v.tag == 'epoch':
                    current_epoch = v.simple_value

                elif self.metric_matcher(v.tag):
                    columns[v.tag].append(v.simple_value)
                    current_logged_step = step

                    # Syncronize the step count with the loss metrics
                    if len(columns['step']) != len(columns[v.tag]):
                        columns['step'].append(step)

                    # Syncronize the wall.time with the loss metrics
                    if len(columns['wall.time']) != len(columns[v.tag]):
                        columns['wall.time'].append(e.wall_time)

                    # Syncronize the epoch with the loss metrics
                    if current_epoch is not None and len(columns['epoch']) != len(columns[v.tag]):
                        columns['epoch'].append(current_epoch)

                elif (v.tag.endswith('W/text_summary') or v.tag.endswith('W_real/text_summary')) and \
                        current_logged_step == step:
                    # recursive_weight means verbose flag was set
                    if self.recursive_weight:
                        W = _parse_numpy_str(v.tensor.string_val[0][5:-6].decode('ascii'))
                        #if len(columns['step']) != len(columns['recursive.weight']):
                        #    columns['recursive.weight'].append(W[0, -1])

                        # assuming single layer task (i.e. 2 weights in matrix only)
                        W = W.reshape(-1)  # flatten first since modules save weights differently e.g. [i,o] or [o,i]
                        if len(columns['step']) != len(columns['weights.w0']):
                            columns['weights.w0'].append(W[0])
                        if len(columns['step']) != len(columns['weights.w1']):
                            columns['weights.w1'].append(W[1])

                elif v.tag.endswith('gate/text_summary') and current_logged_step == step and self.recursive_weight:
                    g = _parse_numpy_str(v.tensor.string_val[0][5:-6].decode('ascii'))
                    # assuming single layer task (i.e. 2 weights in matrix only)
                    g = g.reshape(-1)
                    if len(columns['step']) != len(columns['gate.g0']):
                        columns['gate.g0'].append(g[0])
                    if len(columns['step']) != len(columns['gate.g1']):
                        columns['gate.g1'].append(g[1])
                            
                elif v.tag.endswith('W/sparsity_error') and current_logged_step == step:
                    # Step changed, update sparse error
                    if len(columns['step']) != len(columns['sparse.error.max']):
                        columns['sparse.error.max'].append(v.simple_value)
                    else:
                        columns['sparse.error.max'][-1] = max(
                            columns['sparse.error.max'][-1],
                            v.simple_value
                        )

                #######################################################################################################
                # Process sparsity error for the NPU and RealNPU real weight matrix
                elif v.tag.endswith('W_real/sparsity_error') and current_logged_step == step:
                    # Step changed, update sparse error
                    if len(columns['step']) != len(columns['sparse.error.max']):
                        columns['sparse.error.max'].append(v.simple_value)
                        # columns['W.re'].append(v.simple_value)

                    else:
                        columns['sparse.error.max'][-1] = max(
                            columns['sparse.error.max'][-1],
                            v.simple_value
                        )
                        # columns['W.re'][-1] = v.simple_value

                # Process sparsity error for the NPU and RealNPU imaginary weight matrix
                elif v.tag.endswith('W_im/sparsity_error') and current_logged_step == step:
                    if len(columns['step']) != len(columns['sparse.error.max']):
                        assert "NPU should have already processed a W_real value. " \
                               "There should already exist a sparisty error in this row"
                    else:
                        # calc avg sparsity err between the W_re and W_im
                        columns['sparse.error.max'][-1] = (columns['sparse.error.max'][-1] + v.simple_value)/2.

                #######################################################################################################
                # Process sparsity error for the (mul)MCFC params -1) junction, 2) out_gate (and 3) bias for mulMCFC)
                # Process  sparsity error for the MCFC junction
                elif v.tag.endswith('mcfc_junction/sparsity_error') and current_logged_step == step:
                    # Step changed, update sparse error
                    if len(columns['step']) != len(columns['sparse.error.max']):
                        columns['sparse.error.max'].append(v.simple_value)
                    else:
                        columns['sparse.error.max'][-1] = max(
                            columns['sparse.error.max'][-1],
                            v.simple_value
                        )
                # Process sparsity error for the MCFC out_gate
                elif v.tag.endswith('mcfc_out_gate/sparsity_error') and current_logged_step == step:
                    if len(columns['step']) != len(columns['sparse.error.max']):
                        assert "MCFC should have already processed a mcfc_junction value. " \
                               "There should already exist a sparsity error in this row"
                    else:
                        # calc avg sparsity err between the junction and out_gate
                        columns['sparse.error.max'][-1] = (columns['sparse.error.max'][-1] + v.simple_value)/2.

                # Process sparsity error for the (mul)MCFC gate
                elif v.tag.endswith('mulmcfc_bias/sparsity_error') and current_logged_step == step:
                    if len(columns['step']) != len(columns['sparse.error.max']):
                        assert "MCFC should have already processed a mcfc_junction and mcfc_out_gate value. " \
                               "There should already exist a sparsity error in this row"
                else:
                    # calc avg sparsity err between the junction, out_gate and bias.
                    # requires reverting the normalisation from the MFCF and reapplying a norm of cardinality 3
                    columns['sparse.error.max'][-1] = ((columns['sparse.error.max'][-1] * 2) + v.simple_value) / 3.
                #######################################################################################################

        if len(columns['sparse.error.max']) == 0:
            columns['sparse.error.max'] = [None] * len(columns['step'])
        if self.recursive_weight:
            if len(columns['recursive.weight']) == 0:
                columns['recursive.weight'] = [None] * len(columns['step'])

        return dirname, columns

    def _parse_tensorboard_data_for_weights(self, inputs):
        # only parses the text_summaries, saving each parameter element in it's own column
        (dirname, filename, reader) = inputs

        columns = collections.defaultdict(list)
        columns['name'] = dirname

        for e in tf.compat.v1.train.summary_iterator(filename):
            step = e.step - self.step_start

            for v in e.summary.value:
                if v.tag.endswith('/text_summary'):
                    param_name = v.tag.split('/')[-2]
                    param = _parse_numpy_str(v.tensor.string_val[0][5:-6].decode('ascii'))
                    param = param.reshape(-1)   # flatten to 1D

                    # plot each element of the parameter in its own column using the index as the element reference
                    for i, weight in enumerate(param):
                        col_name = "param." + param_name + "." + str(i)
                        columns[col_name].append(param[i])

                    # Syncronize the step count with the loss metrics
                    if len(columns['step']) != len(columns[col_name]):
                        columns['step'].append(step)

        return dirname, columns

    def __iter__(self):
        reader = TensorboardReader(self.dirname, auto_open=False)
        with tqdm(total=len(reader), disable=not self.progress_bar) as pbar, \
             multiprocessing.Pool(self.processes) as pool:

            columns_order = None
            for dirname, data in pool.imap_unordered(
                    self._parse_tensorboard_data_for_weights if self.weights_only else self._parse_tensorboard_data,
                    reader):
                pbar.update()

                # Check that some data is present
                # if len(data['step']) == 0:
                #     print(f'missing data in: {dirname}')
                #     continue

                # Fix flushing issue
                for column_name, column_data in data.items():
                    if len(data['step']) - len(column_data) == 1:
                        data[column_name].append(None)

                # Convert to dataframe
                df = pandas.DataFrame(data)
                if len(df) == 0:
                    print(f'Warning: No data for {dirname}')
                    continue

                # Ensure the columns are always order the same
                if columns_order is None:
                    columns_order = df.columns.tolist()
                else:
                    df = df[columns_order]

                df.rename(_csv_format_column_name, axis='columns', inplace=True)
                yield df

