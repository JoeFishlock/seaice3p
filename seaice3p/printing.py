from typing import Callable


def get_printer(verbosity_level: int) -> Callable[[str], None]:
    if verbosity_level >= 1:

        def optprint(message: str, **kwargs):
            print(message, **kwargs)

    else:

        def optprint(message: str, **kwargs):
            pass

    return optprint
