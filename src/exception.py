import sys

class CustomerException(Exception):
    def __init__(self, error: Exception, error_detail: object = None):
        if error_detail is not None and hasattr(error_detail, "exc_info"):
            _, _, tb = error_detail.exc_info()
            filename = tb.tb_frame.f_code.co_filename if tb else "<unknown>"
            line = tb.tb_lineno if tb else 0
            message = f"{type(error).__name__} in {filename}:{line} - {error}"
        else:
            message = str(error)
        super().__init__(message)
