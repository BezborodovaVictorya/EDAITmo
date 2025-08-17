class DataFileNotFoundError(FileNotFoundError):
    pass

class ParquetReadError(IOError):
    pass

class SchemaValidationError(ValueError):
    pass

class CurrencyConversionError(RuntimeError):
    pass
