if __package__ is None or __package__ == '':
    # uses current directory visibility
    from serve.app import create_app  # pragma: no cover
else:
    # uses current package visibility
    from .app import create_app


app = create_app()  # pragma: no cover
