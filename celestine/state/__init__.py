from .equilibrium_state import EQMState


def get_model(cfg):
    MODEL_CHOICES = {"EQM": EQMState}
    return MODEL_CHOICES[cfg.model]
