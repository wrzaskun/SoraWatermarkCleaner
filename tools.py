import inspect
import os
'''
The function prints the debug message along with the file and line number.
You can use this function anywhere in your code to print debug information with context.
'''

def debug_print(message=""):
    # Gets the filename, line where `debug_print` was called.
    caller_frame = inspect.currentframe().f_back
    line_number = caller_frame.f_lineno
    function_name = caller_frame.f_code.co_name
    file_name = os.path.basename(caller_frame.f_code.co_filename)
    print(f"DEBUG : [{file_name}:{line_number} in {function_name}] {message}")