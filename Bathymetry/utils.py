import logging
from matplotlib import cm, colors as colors_matplotlib


def default_logger(name='GestorBatimetrico') -> logging.Logger:
    """
    Create a default logger with a stream handler if no handlers exist.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def colors_by_beriles(zmin, step_beriles=None, cmap='seismic'):
    def get_colors_cmap(name_cmap, ncolors=None):
        colors = []
        cmap = cm.get_cmap(name_cmap)
        if ncolors is None:
            ncolors = cmap.N
        for i in range(0, cmap.N, int(cmap.N / ncolors)):
            rgba = cmap(i)
            colors.append(colors_matplotlib.rgb2hex(rgba))

        return colors

    if step_beriles is None:
        step_beriles = int(abs(zmin) / 10)
    if step_beriles > abs(zmin):
        step_beriles = int(abs(zmin) / 2)
    if step_beriles == 0:
        step_beriles = 1
    beriles = [i for i in range(-1000000, 0, int(step_beriles)) if i > zmin - step_beriles]
    beriles.append(0)
    if len(beriles) > 128:
        beriles = beriles[-128:]
    beriles = np.concatenate((beriles, -np.array(beriles[-2::-1])))
    ncolors = len(beriles)

    colors = get_colors_cmap(cmap, ncolors=ncolors)
    colors = [colors[i] for i in range(0, len(colors), int(len(colors) / ncolors))]

    return beriles, colors