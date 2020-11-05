
# Random Seed

SEED = 1234

def reset_random_seed(seed: int) -> None:
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    return

# Logging

import logging
import sys

LOGGER_NAME = 'mutag_logger'
LOGGER = logging.getLogger(LOGGER_NAME)
LOGGER_OUTPUT_FILE = './logs.txt'
LOGGER_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)

def _initialize_logger() -> None:
    LOGGER.setLevel(logging.INFO)
    logging_formatter = logging.Formatter('{asctime} - pid: {process} - threadid: {thread} - func: {funcName} - {levelname}: {message}', style='{')
    logging_file_handler = logging.FileHandler(LOGGER_OUTPUT_FILE)
    logging_file_handler.setFormatter(logging_formatter)
    LOGGER.addHandler(logging_file_handler)
    LOGGER.addHandler(LOGGER_STREAM_HANDLER)
    return

_initialize_logger()

import os
from typing import Generator
from contextlib import contextmanager
@contextmanager
def training_logging_suppressed() -> Generator:
    logger_to_original_level = {}
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and 'lightning' == name:
            logger_to_original_level[logger] = logger.level
            logger.setLevel(logging.ERROR)
    with open(os.devnull, 'w') as dev_null:
        orignal_stream = LOGGER_STREAM_HANDLER.stream
        LOGGER_STREAM_HANDLER.setStream(dev_null)
        yield
        LOGGER_STREAM_HANDLER.setStream(orignal_stream)
    for logger, original_level in logger_to_original_level.items():
        logger.setLevel(original_level)
    return

# Manual tqdm

from typing import Generator
from contextlib import contextmanager
@contextmanager
def manual_tqdm(*args, **kwargs) -> Generator:
    import tqdm
    progress_bar = tqdm.tqdm(*args, **kwargs)
    yield progress_bar
    progress_bar.close()
    return

# Debugging Utilities

from typing import Generator
from contextlib import contextmanager
@contextmanager
def warnings_suppressed() -> Generator:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    return

from contextlib import contextmanager
@contextmanager
def exceptions_suppressed() -> Generator:
    try:
        yield
    except Exception:
        pass
    return

import io
from typing import Generator
from contextlib import contextmanager
@contextmanager
def std_out(stream: io.TextIOWrapper) -> Generator:
    import sys
    original_std_out = sys.stdout
    sys.stdout = stream
    yield
    sys.stdout = original_std_out
    return

from typing import Generator
from contextlib import contextmanager
@contextmanager
def suppressed_output() -> Generator:
    import os
    import sys
    with open(os.devnull, 'w') as dev_null:
        with std_out(dev_null):
            yield
    return

from typing import Callable, Union, Generator
from contextlib import contextmanager
@contextmanager
def redirected_output(exitCallback: Union[None, Callable[[str], None]] = None) -> Generator:
    import sys
    from io import StringIO
    temporary_std_out = StringIO()
    with std_out(temporary_std_out):
        yield
    printed_output: str = temporary_std_out.getvalue()
    if exitCallback is not None:
        exitCallback(printed_output)
    return

def shell(shell_command: str) -> str:
    import subprocess
    return subprocess.check_output(shell_command, shell=True).decode('utf-8')

def pid() -> int:
    import os
    return os.getpid()

from typing import Optional
def file(obj) -> Optional[str]:
    import inspect
    file_location: Optional[str] = None
    try:
        file_location = inspect.getfile(obj)
        source_file_location = inspect.getsourcefile(obj)
    except TypeError as err:
        module = inspect.getmodule(obj)
        if module:
            file_location = inspect.getfile(module)
            source_file_location = inspect.getsourcefile(module)
        else:
            file_location = source_file_location = None
    if None in {file_location, source_file_location} or file_location != source_file_location:
        print("Consider using inspect.getsourcefile as well.")
    return file_location

def source(obj) -> None:
    import inspect
    try:
        source_code = inspect.getsource(obj)
    except TypeError as err:
        obj_type = type(obj)
        source_code = inspect.getsource(obj_type)
    print(source_code)
    return

def module(obj):
    return getmodule(obj)

def doc(obj) -> None:
    import inspect
    print(inspect.getdoc(obj))
    return

from typing import Tuple
def parent_classes(obj) -> Tuple[type, ...]:
    import inspect
    cls = obj if inspect.isclass(obj) else type(obj)
    return inspect.getmro(cls)

from typing import Iterable
def p1(iterable: Iterable) -> None:
    for e in iterable:
        print(e)
    return

def pdir(arbitrary_object: object) -> None:
    for e in dir(arbitrary_object):
        print(e)
    return

from typing import List
def current_tensors() -> List:
    import torch
    import gc
    return [e for e in gc.get_objects() if isinstance(e, torch.Tensor)]

def _dummy_tqdm_message_func(index: int):
    return ''
def tqdm_with_message(iterable,
                      pre_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      post_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      *args, **kwargs):
    import tqdm
    if 'bar_format' not in kwargs:
        kwargs['bar_format']='{l_bar}{bar:50}{r_bar}'
    progress_bar_iterator = tqdm.tqdm(iterable, *args, **kwargs)
    for index, element in enumerate(progress_bar_iterator):
        if pre_yield_message_func != _dummy_tqdm_message_func:
            pre_yield_message = pre_yield_message_func(index)
            progress_bar_iterator.set_description(pre_yield_message)
            progress_bar_iterator.refresh()
        yield element
        if post_yield_message_func != _dummy_tqdm_message_func:
            post_yield_message = post_yield_message_func(index)
            progress_bar_iterator.set_description(post_yield_message)
            progress_bar_iterator.refresh()

from typing import Callable
def debug_on_error(func: Callable) -> Callable:
    import pdb
    import traceback
    import sys
    def decorating_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return decorating_function

from typing import Callable
def raise_on_warn(func: Callable) -> Callable:
    import warnings
    def decorating_function(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = func(*args, **kwargs)
        return result
    return decorating_function

TRACE_INDENT_LEVEL = 0
TRACE_INDENTATION = '    '
TRACE_VALUE_SIZE_LIMIT = 200
from typing import Callable
def trace(func: Callable) -> Callable:
    from inspect import signature
    import sys
    import random
    def human_readable_value(value) -> str:
        readable_value = repr(value)
        if len(readable_value) > TRACE_VALUE_SIZE_LIMIT:
            readable_value = readable_value[:TRACE_VALUE_SIZE_LIMIT]+'...'
        return readable_value
    def decorating_function(*args, **kwargs):
        arg_values_string = ', '.join((f'{param_name}={human_readable_value(value)}' for param_name, value in signature(func).bind(*args, **kwargs).arguments.items()))
        probably_unique_id = random.randint(10,99)
        global TRACE_INDENT_LEVEL, TRACE_INDENTATION
        entry_line = f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] {func.__name__}({arg_values_string})'
        with std_out(sys.__stdout__):
            print()
            print(entry_line)
            print()
        TRACE_INDENT_LEVEL += 1
        result = func(*args, **kwargs)
        TRACE_INDENT_LEVEL -= 1
        with std_out(sys.__stdout__):
            print()
            print(f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] returned {human_readable_value(result)}')
            print()
        return result
    return decorating_function

# Timing Utilities

from typing import Callable, Generator
from contextlib import contextmanager
@contextmanager
def timeout(time: int, functionToExecuteOnTimeout: Callable[[], None] = None) -> Generator:
    """NB: This cannot be nested."""
    import signal
    def _raise_timeout(*args, **kwargs):
        raise TimeoutError
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        if functionToExecuteOnTimeout is not None:
            functionToExecuteOnTimeout()
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
    return

from typing import Callable, Generator
from contextlib import contextmanager
@contextmanager
def timer(section_name: str = None, exitCallback: Callable[[float], None] = None) -> Generator:
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        print(f'{section_name} took {elapsed_time} seconds.')
    else:
        print(f'Execution took {elapsed_time} seconds.')
    return

# General Utilities

from collections import defaultdict
def recursive_defaultdict() -> defaultdict:
    return defaultdict(recursive_defaultdict)

def is_ascii(input_string: str) -> bool:
    return all(ord(character) < 128 for character in input_string)

from typing import Generator
from contextlib import contextmanager
@contextmanager
def temp_plt_figure(*args, **kwargs) -> Generator:
    import matplotlib.pyplot as plt
    figure = plt.figure(*args, **kwargs)
    yield figure
    plt.close(figure)
    return

from typing import Iterable
def only_one(items: Iterable):
    assert isinstance(items, Iterable)
    items = [e for e in items]
    assert len(items) == 1
    return items[0]

from typing import List
def at_most_one(items: List):
    return only_one(items) if items else None

from typing import List
def parallel_map(*args, **kwargs) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.map(*args, **kwargs)
    p.close()
    p.join()
    return result

from typing import List
def parallel_mapcar(func, *args) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.starmap(func, zip(*args))
    p.close()
    p.join()
    return result

from typing import Iterable, Callable,  List
def eager_map(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

from typing import Iterable, Callable,  List
def eager_map_reduce(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

from typing import Iterable, Callable, List
def eager_filter(func: Callable, iterable: Iterable) -> List:
    return list(filter(func, iterable))

from typing import List
def eager_zip(*args) -> List:
    args = list(map(tuple, args))
    assert len(set(map(len, args))) == 1
    return list(zip(*args))

def identity(input):
    return input

def xor(disjunct_a: bool, disjunct_b: bool) -> bool:
    return bool(disjunct_a) ^ bool(disjunct_b)

def implies(antecedent: bool, consequent: bool) -> bool:
    return not antecedent or consequent

def iff(antecedent: bool, consequent: bool) -> bool:
    return bool(antecedent) == bool(consequent)

UNIQUE_BOGUS_RESULT_IDENTIFIER = object()

from typing import Generator
def uniq(iterator: Iterable) -> Generator:
    previous = UNIQUE_BOGUS_RESULT_IDENTIFIER
    for value in iterator:
        if previous != value:
            yield value
            previous = value

from itertools import cycle, islice
from typing import Iterable, Generator
def roundrobin(*iterables: Iterable) -> Generator:
    number_of_active_iterables = len(iterables)
    nexts = cycle(iter(iterable).__next__ for iterable in iterables)
    while number_of_active_iterables > 0:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            number_of_active_iterables -= 1
            nexts = cycle(islice(nexts, number_of_active_iterables))

from typing import Iterable 
def powerset(iterable: Iterable) -> Iterable:
    from itertools import chain, combinations
    items = list(iterable)
    number_of_items = len(items)
    subset_iterable = chain.from_iterable(combinations(items, length) for length in range(1, number_of_items+1))
    return subset_iterable

def n_choose_k(n: int, k: int):
    from functools import reduce
    k = min(k, n-k)
    numerator = reduce(int.__mul__, range(n, n-k, -1), 1)
    denominator = reduce(int.__mul__, range(1, k+1), 1)
    return numerator // denominator

def lerp(start: float, end: float, floatValue: float) -> float:
    return start + floatValue * (end - start);

def false(*args, **kwargs) -> bool:
    return False

def current_timestamp_string() -> str:
    import time
    return time.strftime("%Y_%m_%d_%H_%M_%S")

from typing import Iterable 
def unzip(zipped_item: Iterable) -> Iterable:
    return zip(*zipped_item)

from collections import Counter
def histogram(iterator: Iterable) -> Counter:
    counter: Counter = Counter()
    for element in iterator:
        counter[element]+=1
    return counter

