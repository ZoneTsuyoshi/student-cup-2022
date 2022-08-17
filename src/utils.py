
class Config:
    def __init__(self, d: dict):
        for k,v in d.items():
            setattr(self, k, v)