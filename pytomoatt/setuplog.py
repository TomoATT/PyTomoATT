import logging


class SetupLog():
    def __init__(self):
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        ch.setFormatter(formatter)

        self.SrcReclog = logging.getLogger('SrcRec')
        if not self.SrcReclog.handlers:
            self.SrcReclog.setLevel(logging.INFO)
            self.SrcReclog.addHandler(ch)

        self.Modellog = logging.getLogger('Model')
        if not self.Modellog.handlers:
            self.Modellog.setLevel(logging.INFO)
            self.Modellog.addHandler(ch)

        self.Outputlog = logging.getLogger('Output')
        if not self.Outputlog.handlers:
            self.Outputlog.setLevel(logging.INFO)
            self.Outputlog.addHandler(ch)