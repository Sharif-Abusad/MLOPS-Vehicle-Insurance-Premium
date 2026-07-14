import sys
import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including file name, line number and the error message.

    :param error: The exception that occured.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    # Extract traceback details (exception information)
    _, _, exc_tb = error_detail.exc_info()

    # Get the file name where the exception occured
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a formatted error message string with the file name, line number and actual error
    line_no = exc_tb.tb_lineno
    error_message = f"Error occurred in python script: [{file_name}] at line number [{line_no}]: {str(error)}"

    # Log the error for better tracking 
    logging.error(error_message)

    return error_message


class MyException(Exception):
    """
    Custom exception class for handling errors in vehicle insurance application.
    """
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the MyException with a detailed error message.

        :param error_message: A string describing the error.
        :param error_detail: The sys module to access traceback details.
        """
        # Call the base class constructor with the error message
        super().__init__(error_message)

        # Format the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message