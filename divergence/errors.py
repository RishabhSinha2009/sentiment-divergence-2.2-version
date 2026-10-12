class DataSourceError(RuntimeError):
    """A data provider failed or returned nothing.

    The pipeline stops on this instead of skipping ahead, because quietly
    continuing with a hole in the data would make every later result
    misleading.
    """
