from typing import Callable


def get_printer(
    verbosity_level: int, verbosity_threshold: int
) -> Callable[[str], None]:
    if verbosity_level >= verbosity_threshold:

        def optprint(message: str, **kwargs):
            print(message, **kwargs)

    else:

        def optprint(message: str, **kwargs):
            pass

    return optprint
